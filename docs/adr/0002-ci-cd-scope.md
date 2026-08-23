# ADR 0002: CI/CD Scope

- Status: Accepted
- Date: 2026-08-23

## Context

The repository has no application service or model registry yet. It still needs
automated quality checks and a safe way to publish versioned research software.

## Decision

Use GitHub Actions for pull-request and main-branch CI. The initial CD workflow
builds and attaches validated Python distributions to GitHub releases created
from semantic version tags. Production service deployment, package-index
publishing, and model-registry automation are deferred until their hosting and
credential requirements are explicitly decided.

## Consequences

The project gets useful release automation without inventing cloud
infrastructure. Future deployment workflows must remain independent of the
research training and evaluation workflows.
