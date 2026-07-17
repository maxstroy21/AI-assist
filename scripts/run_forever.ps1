# Держит ассистента запущенным: перезапускает при падении.
# Штатный выход (/quit, код 0) и ошибка конфигурации (код 2) цикл останавливают.
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$log = Join-Path $root "data\service.log"
New-Item -ItemType Directory -Force -Path (Join-Path $root "data") | Out-Null

while ($true) {
    "[$(Get-Date -Format o)] запуск ассистента" | Add-Content $log
    & "$root\.venv\Scripts\python.exe" -m sba 2>> $log
    $code = $LASTEXITCODE
    "[$(Get-Date -Format o)] ассистент завершился с кодом $code" | Add-Content $log
    if ($code -eq 0) { break }
    if ($code -eq 2) {
        "[$(Get-Date -Format o)] ошибка конфигурации - исправьте config и перезапустите" | Add-Content $log
        break
    }
    Start-Sleep -Seconds 5
}
