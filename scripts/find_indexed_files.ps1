<#
.SYNOPSIS
    Быстрый поиск файлов по системному индексу Windows.

.DESCRIPTION
    Ищет указанные имена файлов через COM-интерфейс Microsoft.Windows.Search.QueryHelper.
    Возвращает абсолютные пути (System.ItemPathDisplay) из SYSTEMINDEX.

.PARAMETER Names
    Список имён файлов с расширениями (например: zoom.exe, chrome.exe, notepad.exe)

.EXAMPLE
    PS> .\find_indexed_files.ps1 zoom.exe
    C:\Users\Echo\AppData\Local\Zoom\bin\Zoom.exe

.EXAMPLE
    PS> .\find_indexed_files.ps1 zoom.exe chrome.exe
    C:\Users\Echo\AppData\Local\Zoom\bin\Zoom.exe
    C:\Program Files\Google\Chrome\Application\chrome.exe
#>

param (
    [Parameter(Mandatory=$true, ValueFromRemainingArguments=$true)]
    [string[]] $Names
)

# --- Основная функция поиска через индекс ---
function Find-IndexedFile($fileName) {
    try {
        $queryHelper = New-Object -ComObject "Microsoft.Windows.Search.QueryHelper"
        $queryHelper.ConnectionString = "provider=Search.CollatorDSO;extended properties='Application=Windows';"

        # Формируем SQL-запрос к индексу
        $queryHelper.Query = "SELECT System.ItemPathDisplay FROM SYSTEMINDEX WHERE System.FileName LIKE '$fileName'"

        $recordset = $queryHelper.CreateRecordset()
        $results = @()

        while (-not $recordset.EOF) {
            $path = $recordset.Fields.Item("System.ItemPathDisplay").Value
            if ($path) { $results += $path }
            $recordset.MoveNext()
        }

        return $results
    }
    catch {
        Write-Error "Ошибка при поиске '$fileName': $_"
    }
}

# --- Основная логика ---
foreach ($name in $Names) {
    $found = Find-IndexedFile $name
    if ($found -and $found.Count -gt 0) {
        foreach ($path in $found) {
            Write-Output $path
        }
    } else {
        Write-Warning "Файл '$name' не найден в индексе."
    }
}
