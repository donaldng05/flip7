# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

- Establish the reproducible project foundation.
- Specify Flip 7 rules and implement the deterministic core engine.
- Expose the engine as a PettingZoo AEC environment with a Gymnasium wrapper.

## Release policy

Releases use semantic version tags in the form `vMAJOR.MINOR.PATCH`. A release
is created only after CI passes on the commit being tagged. The release
workflow attaches the source distribution and wheel to the GitHub release.
