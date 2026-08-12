=============================
Evals that touch the network
=============================

The requests an eval makes go out for real. What agent-exam controls is the
environment each attempt runs in: the variables the agent inherits, how many
attempts reach a remote resource at once, and which commands it may run
unattended.

Environment variables
=====================

Every attempt inherits the environment you run agent-exam in, so whatever the
skill reads from it — a token, a key, an endpoint — is already in place.
``setup.env`` adjusts it per task: a string sets or overrides a variable,
``null`` removes it.

.. code-block:: yaml

    setup:
      env:
        API_TOKEN: null

Removing a variable is the only way to exercise what a skill does when it is
absent, since a run otherwise sees your own environment as it is.

Concurrency groups
==================

When parallel attempts against one remote resource would interfere with each
other — racing on its state, or tripping its rate limits — declare a
concurrency group on the task:

.. code-block:: yaml

    concurrency_group: remote_api

and cap it in :file:`evals/config.yaml`:

.. code-block:: yaml

    concurrency_groups:
      remote_api: 1

Tasks in that group then serialize within a run, so each one sees a clean
state. The cap applies within a run and not across runs, so two people running
the same tasks at the same moment can still collide — rerun if you see
suspicious failures.

.. _permission-mode:

Letting the agent run shell commands unattended
===============================================

Network-using evals typically need the agent to run shell commands without a
human approving each one. These settings are harness-specific, which is why
they sit under a per-harness block rather than at the task's top level.

**Default: allow only the specific tools the task needs.**

For Claude Code, use ``allowed_tools``:

.. code-block:: yaml

    claude_code:
      allowed_tools:
        - "Bash(gh*)"        # the CLI this eval needs

Patterns use Claude Code's own ``--allowed-tools`` syntax: ``Bash(curl*)``,
``WebFetch(domain:example.com)``, and so on.

For OpenCode, use ``permission`` with pattern-to-action mappings. ``"ask"``
auto-rejects under ``opencode run``, so it is the headless equivalent of
blocking a tool:

.. code-block:: yaml

    opencode:
      permission:
        bash:
          "*": "ask"      # block all bash by default
          "gh *": "allow" # except the CLI this eval needs

Codex CLI has no per-command allowlist. agent-exam runs ``codex exec``
non-interactively with ``--ask-for-approval never``, so the shell tool always
runs unattended and nothing pauses for confirmation. What a command may *do* is
governed entirely by Codex's sandbox, and network is a sub-capability of the
``workspace-write`` sandbox. So there is no ``gh`` to allow — you just enable
network:

.. code-block:: yaml

    codex_cli:
      network_access: true

Leave it unset and the ``workspace-write`` default disables network: the
command still runs, but its requests fail, with no prompt. See Codex's `agent
approvals and security
<https://developers.openai.com/codex/agent-approvals-security>`_ documentation
for the upstream model, and :ref:`codex-cli-task-block` for the remaining
fields.

A task can carry blocks for several harnesses side by side. Only the block for
the harness actually running is used:

.. code-block:: yaml

    claude_code:
      allowed_tools:
        - "Bash(gh*)"
    opencode:
      permission:
        bash:
          "*": "ask"
          "gh *": "allow"
    codex_cli:
      network_access: true

For harnesses with per-tool allowlists, anything outside the allowed list is
blocked or hidden. For Codex, the sandbox and ``network_access`` are the
confinement boundary.

**Last resort, Claude Code only:** ``permission_mode: bypassPermissions``. Use
it only when the path under test genuinely needs unattended shell that cannot
be enumerated as patterns, and you have verified what those commands will
touch:

.. code-block:: yaml

    claude_code:
      permission_mode: bypassPermissions

Set it per task, inside the ``claude_code:`` block, rather than globally.

Agent-decided fetches
=====================

Tools like ``WebFetch`` are agent-initiated: whether and where they are called
is part of the behavior under evaluation, so they are neither blocked nor
mocked. Pin the expected behavior with ``tool_called``, ``tool_not_called`` or
``tool_count`` assertions. If usage drifts — a skill suddenly fetching on every
run — the assertions surface it.
