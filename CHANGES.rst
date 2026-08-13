=======
Changes
=======

0.1.1 (unreleased)
==================

-   Fixed a crash on Windows consoles, where printing reports, diffs or
    trajectories raised ``UnicodeEncodeError`` on the arrows and box characters
    they contain.

-   Judges now see more of each tool result, and are told that the elisions in
    an abridged trajectory hide content that was really there.

    Tool results were trimmed hard enough that the values a criterion asks
    about often fell inside the elided part, and judges read that absence as
    the agent not having done the work.

0.1.0 (2026-08-12)
==================

Initial release.
