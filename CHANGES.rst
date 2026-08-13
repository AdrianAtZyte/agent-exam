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

-   Judges now see the full tool input of each call, and keep the trajectory
    being graded when a long one has to be cut down.

    Tool inputs are the action under evaluation, such as a command line or the
    body of a written file, and 600 characters cut most of them off mid-value.
    A trajectory over the size limit now drops the bodies of subagent runs
    first, leaving a note of how many turns were omitted, and only truncates
    the parent turns if it still does not fit; the final report at the end of
    a run no longer disappears because a subagent filled up the budget.

0.1.0 (2026-08-12)
==================

Initial release.
