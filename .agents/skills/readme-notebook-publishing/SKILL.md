---
name: readme-notebook-publishing
description: Refresh BetterMDPTools README content, static README visuals, or committed notebook outputs for GitHub and package-index presentation.
---

# Publish the README and notebooks

Read the repository [agent guide](../../../AGENTS.md),
[Repository orientation](../repository-orientation/SKILL.md),
[Generated documentation](../generated-documentation/SKILL.md), and
[Testing and validation](../testing-validation/SKILL.md) first.

## Establish the source of truth

- Trace recent relevant history, public exports, and user-facing API changes
  before deciding which notebooks, README claims, or visuals are affected.
- Execute and validate notebooks against the editable Poetry checkout, not an
  installed release. Notebook tooling may be installed into the local virtual
  environment when absent, but do not add it to project metadata unless the task
  requires a maintained dependency.
- Keep README requirements synchronized with `[project]` and optional extras in
  `pyproject.toml`.

## Preserve useful notebook output

Execute every affected notebook from its first cell through its last cell with
errors fatal. Never use an allow-errors mode, skip a failing cell, or clear
outputs after execution.

Retain plots and meaningful printed results. Avoid committing environment-local
install logs, progress-bar redraws, timestamped third-party logs, or execution
timing metadata. Prefer preventing that output during a clean re-execution, such
as disabling progress displays in the runner environment or configuring a
notebook's optional tool logging. Removing timing metadata after a successful
run is acceptable because it does not remove cell results or execution counts.

Verify each notebook has no error outputs, every code cell other than an
intentional empty cell has an execution count, and expected plotting cells have
saved image output. Inspect source and metadata diffs separately from binary
output changes.

## Publish README visuals

Generate visuals through the documented public API and commit them under
`docs/assets/`. Use descriptive alt text and an absolute raw repository URL in
`README.md` that resolves from both GitHub and package indexes. For newly added
assets, prefer an immutable commit-pinned URL so the image renders during PR
review and remains stable after publication. An absolute URL targeting `master`
alone does not provide that immutability.

Build the wheel and source distribution after a README change. Inspect their
packaged long-description content and verify its README image references resolve
independently of installed package contents. Exercise a representative public
workflow from clean wheel and source-distribution installs outside the repository
so imports cannot accidentally resolve to the editable checkout.

Before handoff, inspect notebooks for absolute local paths, ANSI control
sequences, timestamps, unexpectedly large text output, errors, and unrelated
metadata churn. Report the exact notebooks executed, plots retained, tooling
used, and any failures without masking them.
