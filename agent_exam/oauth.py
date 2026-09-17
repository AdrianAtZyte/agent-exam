from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

import click

from .config import McpHttpServer
from .errors import UsageError

if TYPE_CHECKING:
    from collections.abc import Callable

    from .config import Config

_TIMEOUT = 15

# Some CDNs refuse urllib's default user agent outright.
_HEADERS = {"User-Agent": f"agent-exam/{version('agent-exam')}"}


def post(url: str, body: dict, where: str, *, as_json: bool = False) -> dict:
    """POST *body* to *url*, as a form unless *as_json*, and return the JSON
    it answers with. Failures become a :py:class:`UsageError` naming *where*.
    """
    if not url.startswith(("http://", "https://")):
        raise UsageError(f"{where}: {url} must be an http:// or https:// URL")
    if as_json:
        data = json.dumps(body).encode()
        content_type = "application/json"
    else:
        data = urllib.parse.urlencode(body).encode()
        content_type = "application/x-www-form-urlencoded"
    # Scheme checked above; ruff's S310 still flags Request() itself.
    request = urllib.request.Request(  # noqa: S310
        url,
        data=data,
        headers={**_HEADERS, "Content-Type": content_type},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:  # noqa: S310
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        raise UsageError(f"{where}: request to {url} failed: {exc}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise UsageError(f"{where}: request to {url} failed: {exc}") from exc


def _get_json(url: str) -> dict | None:
    request = urllib.request.Request(url, headers=_HEADERS)  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:  # noqa: S310
            payload = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _store_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "agent-exam" / "mcp-oauth.json"


def _load_store() -> dict[str, dict]:
    try:
        return json.loads(_store_path().read_text())
    except (OSError, ValueError):
        return {}


def _save_login(url: str, entry: dict) -> None:
    path = _store_path()
    store = _load_store()
    store[url] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, indent=1))
    path.chmod(0o600)


def has_login(url: str) -> bool:
    """Whether a login is stored for the server at *url*."""
    return url in _load_store()


def expires_in(payload: dict) -> int | None:
    """The lifetime in seconds the token response *payload* declares for the
    token it carries."""
    try:
        return int(payload["expires_in"])
    except (KeyError, TypeError, ValueError):
        return None


def refresh_login(url: str, name: str, where: str) -> tuple[str, int | None]:
    """Return a fresh access token from the login stored for the server at
    *url*, and its lifetime, rotating the stored refresh token when the
    server issues one.
    """
    hint = f"run `agent-exam mcp login {name}`"
    entry = _load_store().get(url)
    if entry is None:
        raise UsageError(f"{where}: no stored login for {url}; {hint}")
    body = {
        "grant_type": "refresh_token",
        "refresh_token": entry["refresh_token"],
        "client_id": entry["client_id"],
        "resource": url,
    }
    if entry.get("client_secret"):
        body["client_secret"] = entry["client_secret"]
    try:
        payload = post(entry["token_endpoint"], body, where)
    except UsageError as exc:
        raise UsageError(f"{exc}; {hint}") from exc
    token = payload.get("access_token")
    if not token:
        raise UsageError(
            f"{where}: token response from {entry['token_endpoint']} has no "
            f"access_token; {hint}"
        )
    if payload.get("refresh_token"):
        _save_login(url, {**entry, "refresh_token": payload["refresh_token"]})
    return token, expires_in(payload)


def _origin_and_path(url: str) -> tuple[str, str]:
    parts = urllib.parse.urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}", parts.path.rstrip("/")


def _resource_metadata(url: str) -> dict:
    origin, path = _origin_and_path(url)
    for candidate in (
        f"{origin}/.well-known/oauth-protected-resource{path}",
        f"{origin}/.well-known/oauth-protected-resource",
    ):
        payload = _get_json(candidate)
        if payload:
            return payload
    return {}


def _auth_server_metadata(issuer: str, where: str) -> dict:
    origin, path = _origin_and_path(issuer)
    for candidate in (
        f"{origin}/.well-known/oauth-authorization-server{path}",
        f"{origin}/.well-known/openid-configuration{path}",
        f"{issuer}/.well-known/openid-configuration",
        f"{issuer}/.well-known/oauth-authorization-server",
    ):
        payload = _get_json(candidate)
        if payload and {"authorization_endpoint", "token_endpoint"} <= payload.keys():
            return payload
    raise UsageError(f"{where}: no OAuth metadata found for {issuer}")


class _Callback(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        url = urllib.parse.urlsplit(self.path)
        if url.path == "/callback":
            self.server.params = dict(urllib.parse.parse_qsl(url.query))  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"<p>Logged in. You can close this tab.</p>")

    def log_message(self, *args: object) -> None:
        pass


def _register_client(
    metadata: dict, issuer: str, redirect_uri: str, scope: str, where: str
) -> tuple[str, str | None]:
    endpoint = metadata.get("registration_endpoint")
    if not endpoint:
        raise UsageError(
            f"{where}: {issuer} offers no dynamic client registration; set "
            "client_id to a client registered by hand"
        )
    body = {
        "client_name": "agent-exam",
        "redirect_uris": [redirect_uri],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    }
    if scope:
        body["scope"] = scope
    registration = post(endpoint, body, where, as_json=True)
    if not registration.get("client_id"):
        raise UsageError(f"{where}: registration at {endpoint} returned no client_id")
    return registration["client_id"], registration.get("client_secret")


def login(
    cfg: Config,
    name: str,
    *,
    open_url: Callable[[str], object] = webbrowser.open,
    timeout: float = 300,
) -> None:
    """Log in to server *name* in the browser and store the login for runs.

    Runs the authorization code flow the MCP specification describes:
    metadata discovery from the server's URL, dynamic client registration
    unless the server's ``oauth`` names a ``client_id``, PKCE, and a
    listener on ``localhost`` for the redirect, which *open_url* is given
    up to *timeout* seconds to reach.
    """
    server = cfg.mcp_servers.get(name)
    if server is None:
        raise UsageError(f"mcp_servers.{name}: no such server")
    where = f"mcp_servers.{name}.oauth"
    if (
        not isinstance(server, McpHttpServer)
        or server.oauth is None
        or server.oauth.client_secret is not None
    ):
        raise UsageError(
            f"{where}: login needs an http or sse server whose oauth block has no "
            "client_secret"
        )
    resource = _resource_metadata(server.url)
    issuer = (
        resource.get("authorization_servers") or [_origin_and_path(server.url)[0]]
    )[0]
    metadata = _auth_server_metadata(issuer, where)
    scope = server.oauth.scope or " ".join(resource.get("scopes_supported") or ())

    callback = http.server.HTTPServer(("localhost", 0), _Callback)
    callback.params = None  # type: ignore[attr-defined]
    callback.timeout = timeout
    redirect_uri = f"http://localhost:{callback.server_address[1]}/callback"
    with callback:
        client_id, client_secret = server.oauth.client_id, None
        if client_id is None:
            client_id, client_secret = _register_client(
                metadata, issuer, redirect_uri, scope, where
            )
        verifier = secrets.token_urlsafe(64)
        challenge = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode()).digest()
        ).rstrip(b"=")
        state = secrets.token_urlsafe(16)
        params = {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "code_challenge": challenge.decode(),
            "code_challenge_method": "S256",
            "state": state,
            "resource": server.url,
        }
        if scope:
            params["scope"] = scope
        authorize_url = (
            f"{metadata['authorization_endpoint']}?{urllib.parse.urlencode(params)}"
        )
        click.echo(f"Log in at:\n\n  {authorize_url}\n")
        open_url(authorize_url)
        deadline = time.monotonic() + timeout
        while callback.params is None and time.monotonic() < deadline:  # type: ignore[attr-defined]
            callback.handle_request()
    result = callback.params  # type: ignore[attr-defined]
    if result is None:
        raise UsageError(f"{where}: no login within {timeout:g} seconds")
    if result.get("state") != state:
        raise UsageError(f"{where}: the login callback carried an unexpected state")
    if "error" in result:
        raise UsageError(
            f"{where}: login failed: {result.get('error_description') or result['error']}"
        )
    body = {
        "grant_type": "authorization_code",
        "code": result.get("code", ""),
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "code_verifier": verifier,
        "resource": server.url,
    }
    if client_secret:
        body["client_secret"] = client_secret
    payload = post(metadata["token_endpoint"], body, where)
    if not payload.get("refresh_token"):
        raise UsageError(
            f"{where}: {issuer} issued no refresh token, so the login could not "
            "be reused by unattended runs"
        )
    entry = {
        "token_endpoint": metadata["token_endpoint"],
        "client_id": client_id,
        "refresh_token": payload["refresh_token"],
    }
    if client_secret:
        entry["client_secret"] = client_secret
    _save_login(server.url, entry)
    click.echo(f"Logged in to {name}; login stored in {_store_path()}")
