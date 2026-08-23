$ErrorActionPreference = "Stop"

function Invoke-QualityCommand {
    param([string[]]$Arguments)

    & uv run @Arguments
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Invoke-QualityCommand @("ruff", "format", "--check", ".")
Invoke-QualityCommand @("ruff", "check", ".")
Invoke-QualityCommand @("pyright")
Invoke-QualityCommand @("pytest", "--cov=flip7", "--cov-report=term-missing", "--cov-report=xml")
