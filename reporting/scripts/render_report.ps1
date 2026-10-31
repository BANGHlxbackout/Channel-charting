param(
    [Parameter(Mandatory=$true)][string]$Docx,
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [Parameter(Mandatory=$true)][string]$PythonPath
)

$ErrorActionPreference = 'Stop'
$docxPath = (Resolve-Path -LiteralPath $Docx).Path
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$outPath = (Resolve-Path -LiteralPath $OutputDir).Path
$pdfPath = Join-Path $outPath ([IO.Path]::GetFileNameWithoutExtension($docxPath) + '.pdf')
if (Test-Path -LiteralPath $pdfPath) { throw "Refusing to overwrite $pdfPath" }
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $document = $word.Documents.Open($docxPath, $false, $true)
    try { $document.ExportAsFixedFormat($pdfPath, 17) }
    finally { $document.Close($false) }
}
finally { $word.Quit() }

$pythonCode = @'
import sys
from pathlib import Path
import pypdfium2 as pdfium
pdf = pdfium.PdfDocument(sys.argv[1])
out = Path(sys.argv[2])
for i, page in enumerate(pdf):
    page.render(scale=1.5).to_pil().save(out / f"page-{i+1}.png")
print(f"pages={len(pdf)}")
'@
& $PythonPath -c $pythonCode $pdfPath $outPath
if ($LASTEXITCODE -ne 0) { throw 'PDF rasterization failed' }
