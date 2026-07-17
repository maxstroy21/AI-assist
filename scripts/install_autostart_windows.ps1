# Регистрирует автозапуск ассистента при входе в Windows (Task Scheduler).
# Запуск: powershell -ExecutionPolicy Bypass -File scripts\install_autostart_windows.ps1
# Удаление: Unregister-ScheduledTask -TaskName "SecondBrainAgent" -Confirm:$false
$runner = Join-Path $PSScriptRoot "run_forever.ps1"
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$runner`""
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit (New-TimeSpan -Days 3650)
Register-ScheduledTask -TaskName "SecondBrainAgent" -Action $action -Trigger $trigger `
    -Settings $settings -Description "Second Brain Agent (локальный AI-ассистент)" -Force
Write-Host "Готово. Ассистент будет запускаться при входе в систему."
Write-Host "Запустить прямо сейчас: Start-ScheduledTask -TaskName SecondBrainAgent"
