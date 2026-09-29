$ErrorActionPreference = 'Stop'
$taskName = 'TSFM-Prospective-CPU'
$project = Split-Path -Parent $PSScriptRoot
$scheduledScript = Join-Path $PSScriptRoot 'run_scheduled.ps1'
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($null -ne $existing) {
    $old = Export-ScheduledTask -TaskName $taskName
    if ($old -notmatch [regex]::Escape($scheduledScript)) {
        throw "Task '$taskName' already exists with a different action; leaving it untouched."
    }
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}
$sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$start = '2026-10-13T09:00:00'
$end = '2027-09-14T23:59:00'
$months = '<January/><February/><March/><April/><May/><June/><July/><August/><September/><October/><November/><December/>'
$xml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>CPU-only prospective TSFM study; one forward-only run per scheduled cutoff.</Description><URI>\$taskName</URI></RegistrationInfo>
  <Triggers><CalendarTrigger><StartBoundary>$start</StartBoundary><EndBoundary>$end</EndBoundary><Enabled>true</Enabled><ScheduleByMonthDayOfWeek><Weeks><Week>2</Week></Weeks><DaysOfWeek><Tuesday/></DaysOfWeek><Months>$months</Months></ScheduleByMonthDayOfWeek></CalendarTrigger></Triggers>
  <Principals><Principal id="Author"><UserId>$sid</UserId><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>
  <Settings><MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy><DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries><StopIfGoingOnBatteries>false</StopIfGoingOnBatteries><AllowHardTerminate>true</AllowHardTerminate><StartWhenAvailable>true</StartWhenAvailable><Enabled>true</Enabled><Hidden>false</Hidden><RunOnlyIfIdle>false</RunOnlyIfIdle><WakeToRun>false</WakeToRun><ExecutionTimeLimit>PT8H</ExecutionTimeLimit><Priority>7</Priority></Settings>
  <Actions Context="Author"><Exec><Command>powershell.exe</Command><Arguments>-NoProfile -ExecutionPolicy Bypass -File &quot;$scheduledScript&quot;</Arguments><WorkingDirectory>$project</WorkingDirectory></Exec></Actions>
</Task>
"@
Register-ScheduledTask -TaskName $taskName -Xml $xml | Out-Null
$registered = Get-ScheduledTask -TaskName $taskName
Export-ScheduledTask -TaskName $taskName | Set-Content -LiteralPath (Join-Path $project 'schedule\registered_task.xml') -Encoding Unicode
[pscustomobject]@{TaskName=$taskName; State=$registered.State; UserSid=$sid; FirstCutoff=$start; LastCutoff='2027-09-14T09:00:00'; StartWhenAvailable=$true; TaskPath=$scheduledScript} | ConvertTo-Json -Compress
