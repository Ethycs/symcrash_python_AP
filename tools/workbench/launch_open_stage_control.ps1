param(
    [string]$Layout = "tools/workbench/symcrash-workbench.json",
    [string]$SendTarget = "127.0.0.1:7400",
    [int]$OscPort = 7500,
    [switch]$InstallOnly
)

$ErrorActionPreference = "Stop"

$oscVersion = "1.30.4"
$oscArchiveUrl = "https://openstagecontrol.ammd.net/packages/open-stage-control_1.30.4_win32-x64.zip"
$oscArchiveSha256 = "29EEB376C8E4A20F0B4B2A51E123A5080BE65B14697111DCCD67D007DD992B4B"
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$installRoot = Join-Path $repoRoot ".pixi\open-stage-control\$oscVersion"
$archivePath = Join-Path $repoRoot ".pixi\open-stage-control\open-stage-control-$oscVersion.zip"
$layoutPath = (Resolve-Path -LiteralPath (Join-Path $repoRoot $Layout)).Path

$oscExecutable = Get-ChildItem -LiteralPath $installRoot -File -Recurse `
    -Filter "open-stage-control.exe" -ErrorAction SilentlyContinue |
    Select-Object -First 1

if ($null -eq $oscExecutable) {
    New-Item -ItemType Directory -Path $installRoot -Force | Out-Null
    New-Item -ItemType Directory -Path (Split-Path -Parent $archivePath) `
        -Force | Out-Null

    $downloadRequired = $true
    if (Test-Path -LiteralPath $archivePath) {
        $existingHash = (Get-FileHash -LiteralPath $archivePath `
            -Algorithm SHA256).Hash
        $downloadRequired = $existingHash -ne $oscArchiveSha256
    }
    if ($downloadRequired) {
        Write-Host "[workbench-ui] downloading Open Stage Control $oscVersion (resumable)"
        & curl.exe --fail --location --retry 3 --continue-at - `
            --output $archivePath $oscArchiveUrl
        if ($LASTEXITCODE -ne 0) {
            throw "Open Stage Control download failed with exit code $LASTEXITCODE"
        }
    }

    $archiveHash = (Get-FileHash -LiteralPath $archivePath `
        -Algorithm SHA256).Hash
    if ($archiveHash -ne $oscArchiveSha256) {
        throw "Open Stage Control archive checksum mismatch: $archiveHash"
    }

    Write-Host "[workbench-ui] extracting to $installRoot"
    Expand-Archive -LiteralPath $archivePath -DestinationPath $installRoot -Force
    $oscExecutable = Get-ChildItem -LiteralPath $installRoot -File -Recurse `
        -Filter "open-stage-control.exe" | Select-Object -First 1
    if ($null -eq $oscExecutable) {
        throw "The Open Stage Control archive contained no executable"
    }
}

Write-Host "[workbench-ui] $($oscExecutable.FullName)"
if ($InstallOnly) {
    exit 0
}

# This machine may set ELECTRON_RUN_AS_NODE globally for development. Open
# Stage Control is an Electron app, so inheriting that variable makes its exe
# behave like node.exe and reject all application switches. Clear it only for
# this child process and restore the caller's environment afterward.
$previousElectronRunAsNode = $env:ELECTRON_RUN_AS_NODE
try {
    Remove-Item Env:ELECTRON_RUN_AS_NODE -ErrorAction SilentlyContinue
    & $oscExecutable.FullName --load $layoutPath --send $SendTarget `
        --osc-port $OscPort
    $oscExitCode = $LASTEXITCODE
} finally {
    if ($null -ne $previousElectronRunAsNode) {
        $env:ELECTRON_RUN_AS_NODE = $previousElectronRunAsNode
    }
}
exit $oscExitCode
