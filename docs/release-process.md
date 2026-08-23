# Release Process

## Dry run

Use the manually dispatched release workflow to build and upload distributions
as a workflow artifact without publishing a GitHub release.

Locally, run:

```powershell
uv sync --locked
uv build
```

Inspect the files in `dist/` and verify installation in a fresh environment
before creating a tag.

## Tagged release

1. Confirm the working tree is clean and all pull request checks are passing.
2. Update `CHANGELOG.md` and the project version in `pyproject.toml`.
3. Create and push an annotated tag such as `v0.1.0`.
4. Confirm the release workflow reruns validation and attaches the wheel and
   source distribution to the GitHub release.
5. Review generated release notes and the attached artifacts.

The workflow does not deploy a service or launch training. Publishing to a
package index or model registry requires a separate decision, protected
environment, and trusted-publishing configuration.

## Correction and rollback

Do not rewrite or force-push a published tag. Correct a bad release by fixing
the source, incrementing the version, and publishing a new patch release.
