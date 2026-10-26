# System PowerShell bootstrap. No Python, uv, administrator, PATH or registry change required.
$ErrorActionPreference = 'Stop'
$LauncherArgs = @($args)
$NoPause = $LauncherArgs -contains '--no-pause'
$Offline = $LauncherArgs -contains '--offline'
$ExitCode = 0
$Boot = $null
function Fail([int]$Code, [string]$Message) {
    $script:ExitCode = $Code
    throw $Message
}
function Get-ArchiveSha256([string]$Path) {
    # .NET is available in Windows PowerShell 5.1 without module autoloading.
    $Algorithm = [System.Security.Cryptography.SHA256]::Create()
    $Stream = $null
    try {
        $Stream = [System.IO.File]::OpenRead($Path)
        return [System.BitConverter]::ToString($Algorithm.ComputeHash($Stream)).Replace('-', '').ToLowerInvariant()
    } finally {
        if ($null -ne $Stream) { $Stream.Dispose() }
        $Algorithm.Dispose()
    }
}
function Safe-Path([string]$Path) {
    $Current = [System.IO.Path]::GetFullPath($Path)
    while ($Current) {
        if (Test-Path -LiteralPath $Current) {
            $Item = Get-Item -LiteralPath $Current -Force
            if ($Item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
                Fail 12 'Symlink/junction runtime/output paths are not accepted; existing files preserved.'
            }
        }
        $Parent = [System.IO.Path]::GetDirectoryName($Current)
        if ($Parent -eq $Current) { break }
        $Current = $Parent
    }
}
try {
    foreach ($Value in $LauncherArgs) {
        if ($Value -notin @('--offline', '--no-pause', '--help', '-h')) { Fail 2 'Unknown option. Use --offline, --no-pause, or --help.' }
    }
    if (($LauncherArgs -contains '--help') -or ($LauncherArgs -contains '-h')) {
        Write-Output 'AFQMC course launcher: Run_Windows.cmd [--offline] [--no-pause]'
        Write-Output 'First run configures private Python 3.12.10; later runs can be offline.'
    } else {
        if (-not [Environment]::Is64BitOperatingSystem -or -not [Environment]::Is64BitProcess -or $env:PROCESSOR_ARCHITECTURE -ne 'AMD64') {
            Fail 14 'Requires native Windows x64 and 64-bit PowerShell.'
        }
        $Root = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
        $Runtime = Join-Path $Root '.course_runtime'
        foreach ($Path in @($Runtime, (Join-Path $Runtime 'bootstrap'), (Join-Path $Runtime 'cache'), (Join-Path $Runtime 'python'), (Join-Path $Runtime 'envs'), (Join-Path $Root 'course_validation'))) { Safe-Path $Path }
        [System.IO.Directory]::CreateDirectory((Join-Path $Runtime 'bootstrap')) | Out-Null
        $Archive = Join-Path $Runtime 'uv-0.13.5-windows-x64.zip'
        $Expected = 'a38b8d1902b7cfbc76b423ba1bb456f623f42064cdc72628b6075b89afd1019a'
        Safe-Path $Archive
        if (-not (Test-Path -LiteralPath $Archive -PathType Leaf)) {
            if ($Offline) { Fail 10 'OFFLINE: verified uv cache is missing. Run once online in this extracted directory.' }
            $Part = Join-Path $Runtime ('uv-download-' + [guid]::NewGuid().ToString('N') + '.part')
            Write-Output 'Downloading official uv 0.13.5 (first setup)...'
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
            try { Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/astral-sh/uv/releases/download/0.13.5/uv-x86_64-pc-windows-msvc.zip' -OutFile $Part -TimeoutSec 180 }
            catch { Fail 13 'uv download failed. Check connection and retry; existing files preserved.' }
            if ((Get-ArchiveSha256 $Part) -ne $Expected) { Fail 11 'Downloaded uv SHA-256 mismatch; untrusted bytes were not executed.' }
            [System.IO.File]::Move($Part, $Archive)
        }
        if ((Get-ArchiveSha256 $Archive) -ne $Expected) { Fail 11 'Cached uv SHA-256 mismatch; cache preserved and never executed. Restore a verified cache or use a new extraction.' }
        $Boot = Join-Path (Join-Path $Runtime 'bootstrap') ('run-' + [guid]::NewGuid().ToString('N'))
        [System.IO.Directory]::CreateDirectory($Boot) | Out-Null
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $Zip = [System.IO.Compression.ZipFile]::OpenRead($Archive)
        try {
            $Entry = $Zip.GetEntry('uv.exe')
            if ($null -eq $Entry) { $Entry = $Zip.GetEntry('uv-x86_64-pc-windows-msvc/uv.exe') }
            if ($null -eq $Entry) { Fail 11 'Verified uv archive lacks expected executable.' }
            $Uv = Join-Path $Boot 'uv.exe'
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile($Entry, $Uv, $false)
        } finally { $Zip.Dispose() }
        foreach ($Variable in @(Get-ChildItem Env:)) {
            if ($Variable.Name -match '^(UV_|PYTHON|PIP_)|^(VIRTUAL_ENV|CONDA_PREFIX)$') { Remove-Item -LiteralPath ('Env:' + $Variable.Name) }
        }
        $env:UV_NO_CONFIG='1'; $env:UV_NO_PROGRESS='1'
        $env:UV_PYTHON_INSTALL_BIN='0'; $env:UV_PYTHON_INSTALL_REGISTRY='0'; $env:UV_PYTHON_NO_REGISTRY='1'
        $env:UV_CACHE_DIR=Join-Path $Runtime 'cache'; $env:UV_PYTHON_INSTALL_DIR=Join-Path $Runtime 'python'
        $env:UV_PYTHON_BIN_DIR=Join-Path $Runtime 'bin'; $env:UV_TOOL_DIR=Join-Path $Runtime 'tools'; $env:UV_TOOL_BIN_DIR=Join-Path $Runtime 'bin'
        $env:UV_DEFAULT_INDEX='https://pypi.org/simple'; $env:UV_PYTHON_PREFERENCE='only-managed'
        $env:UV_HTTP_TIMEOUT='60'; $env:UV_HTTP_RETRIES='2'
        if ($Offline) { $env:UV_OFFLINE='1' }
        Set-Location -LiteralPath $Root
        Write-Output 'Checking private CPython 3.12.10...'
        & $Uv run --no-project --no-config --python 3.12.10 -- python -I -X utf8 (Join-Path $Root 'scripts/start_course.py') --uv $Uv @LauncherArgs
        $ExitCode = $LASTEXITCODE
        if ($ExitCode -ne 0) { Write-Output "Launcher failed (exit $ExitCode). Check setup/network/cache and local summary; existing results are preserved." }
    }
} catch {
    if ($ExitCode -eq 0) { $ExitCode = 12 }
    Write-Output "COURSE ERROR ${ExitCode}: $($_.Exception.Message)"
} finally {
    if ($Boot -and (Test-Path -LiteralPath $Boot)) { Remove-Item -LiteralPath $Boot -Recurse -Force }
    if (-not $NoPause) { [void](Read-Host "Exit code: $ExitCode. Press Return to close") }
}
exit $ExitCode
