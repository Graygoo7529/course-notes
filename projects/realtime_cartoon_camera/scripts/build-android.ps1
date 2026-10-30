param([switch]$UseProxy)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
. "$PSScriptRoot/android-env.ps1"
if ($UseProxy) {
    $env:CARGO_HTTP_PROXY = 'http://127.0.0.1:7897'
    # Use Windows' trusted roots, including the local proxy CA, without disabling TLS.
    $env:JAVA_TOOL_OPTIONS = '-Dhttp.proxyHost=127.0.0.1 -Dhttp.proxyPort=7897 -Dhttps.proxyHost=127.0.0.1 -Dhttps.proxyPort=7897 -Dhttp.nonProxyHosts=localhost|127.* -Djavax.net.ssl.trustStoreType=Windows-ROOT -Djavax.net.ssl.trustStore=NONE -Djdk.tls.client.protocols=TLSv1.2 -Dhttps.protocols=TLSv1.2'
}
foreach ($asset in @('mobilenetv3_features.onnx', 'face_landmarker.task', 'cartoon_config.json')) {
    if (!(Test-Path -LiteralPath "models/android/$asset")) {
        throw "Missing models/android/$asset. Run: python python/export_android.py"
    }
}
& npm.cmd run tauri android build -- --debug --apk --target aarch64
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$apk = Get-ChildItem 'apps/desktop/src-tauri/gen/android/app/build/outputs/apk' -Recurse -Filter '*debug.apk' |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (!$apk) { throw 'ARM64 APK was not found in the build outputs.' }
New-Item -ItemType Directory -Force 'dist-android' | Out-Null
Copy-Item -LiteralPath $apk.FullName -Destination 'dist-android/CartoonCamera-arm64-debug.apk'
$hash = (Get-FileHash 'dist-android/CartoonCamera-arm64-debug.apk' -Algorithm SHA256).Hash.ToLower()
"$hash  CartoonCamera-arm64-debug.apk" | Set-Content 'dist-android/SHA256SUMS.txt' -Encoding ASCII
Write-Host 'APK: dist-android/CartoonCamera-arm64-debug.apk'
