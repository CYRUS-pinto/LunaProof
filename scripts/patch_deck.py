import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

OFFICIAL_PPTX = "c:/Users/Cyrus/Downloads/SIH2026-IDEA-Presentation-Format.pptx"
OUTPUT_PPTX = "LunaProof_SIH2026_Maximus2_185903_Official_v2.pptx"

def patch_deck(values=None):
    if values is None:
        values = {
            "gap_180_SIFT": "{{gap_180_SIFT}}",
            "gap_180_PC": "{{gap_180_PC}}",
            "n_tiles": "{{n_tiles}}",
            "pairs_total": "{{pairs_total}}",
            "SIFT_pct": "{{SIFT_pct}}",
            "PCSIFT_pct": "{{PCSIFT_pct}}",
            "LOFTR_pct": "{{LOFTR_pct}}",
            "LEARNED_pct": "{{LEARNED_pct}}",
            "worst_method": "{{worst_method}}",
            "worst_pct": "{{worst_pct}}",
            "showcase_err": "{{showcase_err}}",
            "gate_success_pct": "{{gate_success_pct}}"
        }

    prs = Presentation(OFFICIAL_PPTX)

    # Remove Slide 7 if present to strictly enforce 6 slides
    if len(prs.slides) > 6:
        rId = prs.slides._sldIdLst[6].rId
        prs.part.drop_rel(rId)
        del prs.slides._sldIdLst[6]

    # --- Slide 1: Title Page
    slide1 = prs.slides[0]
    for shape in slide1.shapes:
        if shape.has_text_frame:
            text = shape.text
            if "Problem Statement ID" in text or "Team ID" in text:
                tf = shape.text_frame
                tf.clear()
                p0 = tf.paragraphs[0]
                p0.text = "Problem Statement ID: SIH26166"
                p0.font.bold = True
                p0.font.size = Pt(16)
                
                p1 = tf.add_paragraph()
                p1.text = "Problem Statement Title: Lunar Image Registration Under Extreme Illumination Variations"
                p1.font.size = Pt(14)
                
                p2 = tf.add_paragraph()
                p2.text = "Theme: Space Technology  |  PS Category: Software"
                p2.font.size = Pt(14)
                
                p3 = tf.add_paragraph()
                p3.text = "Team ID: 185903  |  Team Name: Maximus2"
                p3.font.bold = True
                p3.font.size = Pt(16)

    # Set team name on all footer ovals
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame and "Your Team Name" in shape.text:
                shape.text_frame.text = "Maximus2 (185903)"

    # --- Slide 2: Idea Title & Proposed Solution
    slide2 = prs.slides[1]
    for shape in slide2.shapes:
        if shape.has_text_frame:
            if "IDEA TITLE" in shape.text:
                shape.text_frame.text = "LunaProof: Multi-Modal Illumination-Invariant Lunar Image Registration"
            elif "Proposed Solution" in shape.text:
                tf = shape.text_frame
                tf.clear()
                p = tf.paragraphs[0]
                p.text = "Proposed Solution & Core Technical Innovation"
                p.font.bold = True
                p.font.size = Pt(16)
                
                bullets = [
                    "Physics-Based Lunar Rendering: Lommel-Seeliger scattering + ray-marched cast shadows for true illumination modeling.",
                    "Illumination-Invariant Phase Congruency: Kovesi log-Gabor filters extract edge structures invariant to sun-angle flips.",
                    "Multi-Matcher Framework: Benchmarking SIFT, PC-SIFT, LoFTR (zero-shot), and custom PatchNet descriptors.",
                    "Ground-Truth Checkpoint Metrology: Measures sub-pixel error against exact ground truth at held-out checkpoints (not fitted matches).",
                    "Autonomous Refusal Gate: Observes match count, inlier ratio, and spatial coverage to refuse un-reliable registrations."
                ]
                for b in bullets:
                    p_b = tf.add_paragraph()
                    p_b.text = "• " + b
                    p_b.font.size = Pt(13)

    # --- Slide 3: Technical Approach
    slide3 = prs.slides[2]
    for shape in slide3.shapes:
        if shape.has_text_frame:
            if "Technologies to be used" in shape.text or "Methodology" in shape.text:
                tf = shape.text_frame
                tf.clear()
                p0 = tf.paragraphs[0]
                p0.text = "Technical Architecture & Pipeline"
                p0.font.bold = True
                p0.font.size = Pt(16)
                
                bullets = [
                    "Tech Stack: Python, PyTorch, OpenCV, NumPy, SciPy, Kornia, Google Colab, GitHub.",
                    "Dataset: NASA LRO LOLA LDEM_64 elevation tiles (64 PPD, ~474 m/px) split strictly by lunar region.",
                    "Pipeline: DEM Tile -> Lommel-Seeliger + Cast Shadows -> Log-Gabor Phase Map -> Feature Matcher -> MAGSAC++ -> Refusal Gate."
                ]
                for b in bullets:
                    p_b = tf.add_paragraph()
                    p_b.text = "• " + b
                    p_b.font.size = Pt(12)

    # Insert pipeline diagram if exists
    if os.path.exists('assets/flow_pipeline.png'):
        slide3.shapes.add_picture('assets/flow_pipeline.png', Inches(0.8), Inches(4.2), Inches(8.4), Inches(2.2))

    # --- Slide 4: Feasibility and Viability
    slide4 = prs.slides[3]
    for shape in slide4.shapes:
        if shape.has_text_frame:
            if "Analysis of the feasibility" in shape.text:
                tf = shape.text_frame
                tf.clear()
                p0 = tf.paragraphs[0]
                p0.text = "Rigorous Held-Out Benchmark Results & Failure Analysis"
                p0.font.bold = True
                p0.font.size = Pt(16)
                
                bullets = [
                    f"Dataset Feasibility: Real lunar topography from {values['n_tiles']} non-overlapping lunar regions ({values['pairs_total']} test pairs).",
                    f"SIFT Baseline: Succeeded in {values['SIFT_pct']}% of test pairs (breaks under sun angle flips).",
                    f"Phase-Congruency SIFT: Succeeded in {values['PCSIFT_pct']}% of test pairs.",
                    f"LoFTR (Zero-Shot): Succeeded in {values['LOFTR_pct']}% of test pairs.",
                    f"Learned PatchNet Descriptor: Succeeded in {values['LEARNED_pct']}% of test pairs.",
                    f"Refusal Gate Accuracy: Accepted pairs were correct {values['gate_success_pct']}% of the time.",
                    f"Honest Failure Analysis: At 60°-120° sun gaps, {values['worst_method']} drops to {values['worst_pct']}%. Median error showcase: {values['showcase_err']} px."
                ]
                for b in bullets:
                    p_b = tf.add_paragraph()
                    p_b.text = "• " + b
                    p_b.font.size = Pt(12)

    # --- Slide 5: Impact and Benefits
    slide5 = prs.slides[4]
    for shape in slide5.shapes:
        if shape.has_text_frame:
            if "Potential impact" in shape.text:
                tf = shape.text_frame
                tf.clear()
                p0 = tf.paragraphs[0]
                p0.text = "ISRO Mission Impact & Multi-Scale Roadmap"
                p0.font.bold = True
                p0.font.size = Pt(16)
                
                bullets = [
                    "Multi-Modal Co-Registration: Enables robust registration between Chandrayaan-2 OHRC (0.25m), TMC-2 (5m), and IIRS (~80m).",
                    "Safety-Critical Refusal: Prevents corrupted GIS mapping by automatically rejecting ill-conditioned feature matches.",
                    "Scale Cascade Roadmap: IIRS 80m -> TMC-2 5m -> LRO NAC 1m -> OHRC 0.25m (hop ratios 16x, 5x, 4x)."
                ]
                for b in bullets:
                    p_b = tf.add_paragraph()
                    p_b.text = "• " + b
                    p_b.font.size = Pt(13)

    # Insert scale cascade diagram if exists
    if os.path.exists('assets/flow_cascade.png'):
        slide5.shapes.add_picture('assets/flow_cascade.png', Inches(1.0), Inches(4.3), Inches(8.0), Inches(2.2))

    # --- Slide 6: Research and References
    slide6 = prs.slides[5]
    for shape in slide6.shapes:
        if shape.has_text_frame:
            if "Details / Links" in shape.text or "RESEARCH" in shape.text:
                tf = shape.text_frame
                tf.clear()
                p0 = tf.paragraphs[0]
                p0.text = "Primary Research Citations & Repository"
                p0.font.bold = True
                p0.font.size = Pt(16)
                
                bullets = [
                    "Kovesi, P. (1999). Image Features from Phase Congruency. Videre: J. Comput. Vis. Res., 1(3), 1-26.",
                    "Sun, J., et al. (2021). LoFTR: Detector-Free Local Feature Matching with Transformers. CVPR 2021.",
                    "Baranjague et al. (2020). USAC: A Universal Framework for Random Sample Consensus. IEEE TPAMI.",
                    "NASA LRO LOLA LDEM_64 Global Lunar Digital Elevation Model (NASA PDS Geosciences Node).",
                    "ISRO Chandrayaan-2 OHRC, TMC-2 & IIRS Payload Specifications.",
                    "GitHub Repository & Executable Notebook: github.com/pintocyrus/LunaProof"
                ]
                for b in bullets:
                    p_b = tf.add_paragraph()
                    p_b.text = "• " + b
                    p_b.font.size = Pt(12)

    prs.save(OUTPUT_PPTX)
    print(f"Saved patched deck to {OUTPUT_PPTX}")

if __name__ == '__main__':
    patch_deck()
