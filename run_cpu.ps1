param([ValidateSet('audit','benchmark','experiment','finalize','predict','report','test')][string]$Action = 'predict')
$projectRoot = $PSScriptRoot
$localPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$workspacePython = Join-Path (Split-Path (Split-Path $projectRoot -Parent) -Parent) 'work\cnx-venv\Scripts\python.exe'
if (Test-Path -LiteralPath $localPython) { $taskPython = $localPython }
elseif (Test-Path -LiteralPath $workspacePython) { $taskPython = $workspacePython }
else { throw 'Install the project environment using README.md first.' }
Push-Location -LiteralPath $projectRoot
try {
    if ($Action -eq 'test') { & $taskPython -m pytest -q }
    else { & $taskPython -m cnx500.cli $Action }
    if ($LASTEXITCODE -ne 0) { throw "CNX 500 command failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
