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

- Trace recent history and public exports before deciding which notebooks or
  claims are affected.
- Execute notebooks against the editable Poetry checkout, not an installed
  release. Notebook tooling may be installed into the local virtual environment
  when absent, but do not add it to project metadata unless the task requires a
  maintained dependency.
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
`docs/assets/`. Use descriptive alt text and a stable absolute raw repository URL
in `README.md`, because package indexes cannot resolve repository-relative image
paths.

Build the wheel and source distribution after a README change. Inspect their
long-description content and confirm referenced local assets exist. Exercise a
representative public workflow from clean installs of both artifacts.

Before handoff, inspect notebooks for absolute local paths, ANSI control
sequences, timestamps, unexpectedly large text output, errors, and unrelated
metadata churn. Report the exact notebooks executed, plots retained, tooling
used, and any failures without masking them.
