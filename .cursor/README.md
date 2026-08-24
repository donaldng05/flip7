# Cursor Project Guidance

This directory contains guidance shared with agents working in this repository.

## Rules

Rules under `rules/` use `.mdc` frontmatter:

- `project-context.mdc` always applies.
- Other rules use file globs and apply only to matching work.

Rules state durable conventions and point to repository documentation rather
than duplicating it.

## Skills

Skills under `skills/<skill-name>/SKILL.md` are explicit workflows. They are
not automatically invoked by default. Each skill may link to a one-level-deep
reference file for detailed checklists or templates.

This repository now includes baseline-agent and evaluation work; update this
guidance as the project moves into learned training. Keep it consistent with
`docs/engineering-standards.md` and `CONTRIBUTING.md`.
