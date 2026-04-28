param(
    [string] $Version = "",
    [string] $InstallDir = "",
    [switch] $Force,
    [switch] $CheckOnly,
    [switch] $NoPersistEnv,
    [switch] $HumanDownload
)

$ErrorActionPreference = "Stop"

function Test-VulkanSdkDir {
    param([AllowEmptyString()][string] $Path)

    if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
    if (-not (Test-Path -LiteralPath $Path)) { return $false }

    $header = Join-Path $Path "Include\vulkan\vulkan.h"
    $bin = Join-Path $Path "Bin"
    $vulkaninfo = Join-Path $bin "vulkaninfo.exe"
    $layerJson = Join-Path $bin "VkLayer_khronos_validation.json"

    return (Test-Path -LiteralPath $header) -and (Test-Path -LiteralPath $bin) -and (
        (Test-Path -LiteralPath $vulkaninfo) -or (Test-Path -LiteralPath $layerJson)
    )
}

function Get-InstalledVulkanSdkDir {
    if (-not [string]::IsNullOrWhiteSpace($env:VULKAN_SDK)) {
        if (Test-VulkanSdkDir -Path $env:VULKAN_SDK) { return $env:VULKAN_SDK }
    }

    $candidates = @(
        "C:\VulkanSDK",
        "$env:ProgramFiles\VulkanSDK",
        "$env:ProgramFiles(x86)\VulkanSDK",
        "$env:LOCALAPPDATA\VulkanSDK",
        "$env:LOCALAPPDATA\VulkanSDK\VulkanSDK"
    ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique

    foreach ($root in $candidates) {
        if (-not (Test-Path -LiteralPath $root)) { continue }
        $versionDirs = Get-ChildItem -LiteralPath $root -Directory -ErrorAction SilentlyContinue |
            Sort-Object -Property Name -Descending
        foreach ($dir in $versionDirs) {
            if (Test-VulkanSdkDir -Path $dir.FullName) { return $dir.FullName }
        }
    }

    return ""
}

function Get-LatestVulkanSdkVersion {
    # Official API (platform-specific text endpoint).
    # See: https://vulkan.lunarg.com/sdk/latest/windows.txt
    $api = "https://vulkan.lunarg.com/sdk/latest/windows.txt"
    $raw = (Invoke-WebRequest -Uri $api -UseBasicParsing).Content
    $match = [Regex]::Match([string]$raw, "\b\d+\.\d+\.\d+\.\d+\b")
    if ($match.Success) { return $match.Value }
    throw "Unable to parse latest Vulkan SDK version from $api. Response was: $raw"
}

function Get-IsAdmin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Ensure-EnvVar {
    param(
        [Parameter(Mandatory = $true)][string] $Name,
        [Parameter(Mandatory = $true)][string] $Value,
        [ValidateSet("Process", "User")][string] $Scope = "Process"
    )

    if ($Scope -eq "Process") {
        Set-Item -Path "Env:$Name" -Value $Value
        return
    }

    [Environment]::SetEnvironmentVariable($Name, $Value, "User")
}

function Add-ToPath {
    param(
        [Parameter(Mandatory = $true)][string] $Dir,
        [ValidateSet("Process", "User")][string] $Scope = "Process"
    )

    if ([string]::IsNullOrWhiteSpace($Dir) -or -not (Test-Path -LiteralPath $Dir)) { return }

    if ($Scope -eq "Process") {
        $parts = ($env:Path -split ";") | Where-Object { $_ -ne "" }
        if ($parts -notcontains $Dir) {
            $env:Path = ($parts + $Dir) -join ";"
        }
        return
    }

    $existing = [Environment]::GetEnvironmentVariable("Path", "User")
    $parts = @()
    if (-not [string]::IsNullOrWhiteSpace($existing)) {
        $parts = ($existing -split ";") | Where-Object { $_ -ne "" }
    }
    if ($parts -notcontains $Dir) {
        [Environment]::SetEnvironmentVariable("Path", (($parts + $Dir) -join ";"), "User")
    }
}

if (-not $Force) {
    $existing = Get-InstalledVulkanSdkDir
    if (-not [string]::IsNullOrWhiteSpace($existing)) {
        Write-Host "Vulkan SDK detected at: $existing"
        if ($CheckOnly) { exit 0 }
        # Still ensure current session env is usable.
        $bin = Join-Path $existing "Bin"
        Ensure-EnvVar -Name "VULKAN_SDK" -Value $existing -Scope "Process"
        Ensure-EnvVar -Name "VK_LAYER_PATH" -Value $bin -Scope "Process"
        Add-ToPath -Dir $bin -Scope "Process"
        exit 0
    }
}

if ($CheckOnly) {
    Write-Error "Vulkan SDK not found."
    exit 1
}

$resolvedVersion = $Version
if ([string]::IsNullOrWhiteSpace($resolvedVersion)) {
    $resolvedVersion = Get-LatestVulkanSdkVersion
}

$downloadVersionPath = $resolvedVersion
if ([string]::IsNullOrWhiteSpace($Version)) {
    # Use the supported "latest" download endpoint when the user didn't pin a version.
    $downloadVersionPath = "latest"
}

$downloadUrl = "https://sdk.lunarg.com/sdk/download/$downloadVersionPath/windows/vulkan_sdk.exe"
if ($HumanDownload) {
    $downloadUrl = "$downloadUrl?Human=true"
}
$tmpDir = Join-Path $env:TEMP "voxel-engine"
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
$installerPath = Join-Path $tmpDir "vulkan_sdk-$resolvedVersion.exe"

Write-Host "Downloading Vulkan SDK $resolvedVersion ..."
Invoke-WebRequest -Uri $downloadUrl -OutFile $installerPath -UseBasicParsing

if (-not (Test-Path -LiteralPath $installerPath)) {
    throw "Download failed: $installerPath"
}

$admin = Get-IsAdmin
if ([string]::IsNullOrWhiteSpace($InstallDir)) {
    if ($admin) {
        $InstallDir = "C:\VulkanSDK\$resolvedVersion"
    } else {
        $InstallDir = Join-Path $env:LOCALAPPDATA "VulkanSDK\$resolvedVersion"
    }
}

Write-Host "Installing Vulkan SDK to: $InstallDir"

$args = @(
    "--accept-licenses",
    "--default-answer",
    "--confirm-command", "install",
    "--root", $InstallDir
)

if (-not $admin) {
    # Without elevation, avoid system-wide modifications.
    $args += "copy_only=1"
}

$proc = Start-Process -FilePath $installerPath -ArgumentList $args -Wait -PassThru
if ($proc.ExitCode -ne 0) {
    throw "Vulkan SDK installer failed with exit code $($proc.ExitCode)"
}

if (-not (Test-VulkanSdkDir -Path $InstallDir)) {
    throw "Install completed, but expected SDK layout not found at: $InstallDir"
}

Write-Host "Vulkan SDK installed at: $InstallDir"

$binDir = Join-Path $InstallDir "Bin"
Ensure-EnvVar -Name "VULKAN_SDK" -Value $InstallDir -Scope "Process"
Ensure-EnvVar -Name "VK_LAYER_PATH" -Value $binDir -Scope "Process"
Add-ToPath -Dir $binDir -Scope "Process"

if (-not $NoPersistEnv) {
    Ensure-EnvVar -Name "VULKAN_SDK" -Value $InstallDir -Scope "User"
    Ensure-EnvVar -Name "VK_LAYER_PATH" -Value $binDir -Scope "User"
    Add-ToPath -Dir $binDir -Scope "User"
}

Write-Host "Done. Open a new terminal to pick up persisted environment variables."

