import os
import sys
import subprocess
import json

def convert_pptx_to_pdf(input_pptx, output_pdf):
    input_pptx = os.path.abspath(input_pptx)
    output_pdf = os.path.abspath(output_pdf)
    print(f"Converting {input_pptx} -> {output_pdf}")

    # 1. Try LibreOffice / soffice CLI
    for cmd in ['soffice', 'libreoffice', 'C:\\Program Files\\LibreOffice\\program\\soffice.exe']:
        try:
            res = subprocess.run([cmd, '--headless', '--convert-to', 'pdf', '--outdir', os.path.dirname(output_pdf), input_pptx],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            if res.returncode == 0 and os.path.exists(output_pdf):
                print("Converted successfully via LibreOffice!")
                return True
        except Exception:
            pass

    # 2. Try PowerPoint via comtypes
    try:
        import comtypes.client
        powerpoint = comtypes.client.CreateObject("Powerpoint.Application")
        powerpoint.Visible = 1
        deck = powerpoint.Presentations.Open(input_pptx)
        deck.SaveAs(output_pdf, 32) # 32 = ppSaveAsPDF
        deck.Close()
        powerpoint.Quit()
        if os.path.exists(output_pdf):
            print("Converted successfully via MS PowerPoint (comtypes)!")
            return True
    except Exception as e:
        print("comtypes PowerPoint export failed:", e)

    # 3. Try PowerPoint via win32com
    try:
        import win32com.client
        powerpoint = win32com.client.Dispatch("PowerPoint.Application")
        deck = powerpoint.Presentations.Open(input_pptx)
        deck.SaveAs(output_pdf, 32)
        deck.Close()
        powerpoint.Quit()
        if os.path.exists(output_pdf):
            print("Converted successfully via MS PowerPoint (win32com)!")
            return True
    except Exception as e:
        print("win32com PowerPoint export failed:", e)

    # 4. Try PowerShell COM script
    ps_cmd = f"""
$pptx = "{input_pptx.replace('\\', '\\\\')}"
$pdf = "{output_pdf.replace('\\', '\\\\')}"
$ppt = New-Object -ComObject PowerPoint.Application
$pres = $ppt.Presentations.Open($pptx, 1, 0, 0)
$pres.SaveAs($pdf, 32)
$pres.Close()
$ppt.Quit()
"""
    try:
        res = subprocess.run(["powershell", "-Command", ps_cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        if os.path.exists(output_pdf):
            print("Converted successfully via PowerShell COM script!")
            return True
    except Exception as e:
        print("PowerShell export failed:", e)

    return False

if __name__ == '__main__':
    pptx_in = sys.argv[1] if len(sys.argv) > 1 else 'LunaProof_SIH2026_Maximus2_185903_Official_v2.pptx'
    pdf_out = sys.argv[2] if len(sys.argv) > 2 else 'LunaProof_SIH2026_Maximus2_185903_Official_v2.pdf'
    success = convert_pptx_to_pdf(pptx_in, pdf_out)
    if not success:
        print("PDF export failed across all engines!")
        sys.exit(1)
