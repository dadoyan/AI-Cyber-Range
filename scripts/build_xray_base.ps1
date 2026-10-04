param(
    [string]$ArchivePath
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$source = Join-Path $root 'xray_base'

if (-not (Test-Path -LiteralPath $source)) {
    if (-not $ArchivePath) { throw 'Provide -ArchivePath when xray_base is not present.' }
    $archive = (Resolve-Path -LiteralPath $ArchivePath).Path
    Expand-Archive -LiteralPath $archive -DestinationPath $source
}

$required = @(
    'Dockerfile', 'adv_cli.py', 'attack_core.py', 'helper.py',
    'models\efficientnet_b0.pth', 'models\efficientnet_b0_robust.pth'
)
foreach ($relative in $required) {
    if (-not (Test-Path -LiteralPath (Join-Path $source $relative))) {
        throw "The exercise archive is missing $relative"
    }
}
$images = Get-ChildItem -LiteralPath (Join-Path $source 'labeled-chest-xray-images\chest_xray\test') -Filter '*.jpeg' -File -Recurse
if ($images.Count -ne 624) {
    throw "Expected 624 test images; found $($images.Count)"
}

$docker = Get-Command docker.exe -ErrorAction SilentlyContinue
if (-not $docker) {
    $fallback = Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\resources\bin\docker.exe'
    if (-not (Test-Path -LiteralPath $fallback)) {
        throw 'Docker Desktop CLI was not found'
    }
    $dockerPath = $fallback
} else {
    $dockerPath = $docker.Source
}

& $dockerPath build -t adv_demo -f (Join-Path $source 'Dockerfile') $source
if ($LASTEXITCODE -ne 0) {
    throw "Docker build failed with exit code $LASTEXITCODE"
}
Write-Output 'Built adv_demo from the supplied exercise archive.'
