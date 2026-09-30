import os
import aspose.slides as slides
from pypdf import PdfReader

pptx_file = "LunaProof_SIH2026_Maximus2_185903_Official_v2.pptx"
pdf_file = "LunaProof_SIH2026_Maximus2_185903_Official_v2.pdf"
pages_dir = "assets/pdf_pages"
os.makedirs(pages_dir, exist_ok=True)

print(f"Opening {pptx_file}...")
with slides.Presentation(pptx_file) as pres:
    print(f"Saving to {pdf_file}...")
    pres.save(pdf_file, slides.export.SaveFormat.PDF)
    
    print("Rendering slides to PNG...")
    for idx, slide in enumerate(pres.slides):
        img_file = os.path.join(pages_dir, f"slide_{idx+1}.png")
        bmp = slide.get_image(1.5, 1.5)
        bmp.save(img_file)
        print(f"  Rendered slide {idx+1} -> {img_file}")

# Verify PDF properties
reader = PdfReader(pdf_file)
num_pages = len(reader.pages)
file_size_mb = os.path.getsize(pdf_file) / (1024 * 1024)

print("\n=== PDF VERIFICATION RESULTS ===")
print(f"Page Count: {num_pages} (Required: 6)")
print(f"File Size: {file_size_mb:.2f} MB (Required: < 10 MB)")
assert num_pages == 6, f"Expected 6 pages, got {num_pages}"
assert file_size_mb < 10.0, f"Expected < 10 MB, got {file_size_mb:.2f} MB"
print("SUCCESS: PDF verification passed!")
