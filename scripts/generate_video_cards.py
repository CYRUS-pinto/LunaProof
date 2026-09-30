import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs('video/scenes', exist_ok=True)

def generate_card(scene_num, title, speaker, duration, screen_desc, bullet_points, filename, values=None):
    if values is None:
        values = {
            "gap_180_SIFT": "0.0",
            "gap_180_PC": "100.0",
            "n_tiles": "15",
            "pairs_total": "112",
            "SIFT_pct": "28.6",
            "PCSIFT_pct": "88.4",
            "LOFTR_pct": "73.2",
            "LEARNED_pct": "98.2",
            "worst_method": "SIFT",
            "worst_pct": "0.0",
            "showcase_err": "0.685",
            "gate_success_pct": "93.6"
        }

    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)  # 1920x1080
    ax.set_facecolor('#0f172a')
    fig.patch.set_facecolor('#0f172a')

    # Top Header Banner
    banner = patches.Rectangle((0, 0.82), 1, 0.18, transform=ax.transAxes, color='#1e293b')
    ax.add_patch(banner)
    
    ax.text(0.05, 0.91, f"SCENE {scene_num}: {title.upper()}", color='#38bdf8', weight='bold',
            fontsize=26, va='center', transform=ax.transAxes)
    ax.text(0.95, 0.91, f"Speaker: {speaker} | Time: {duration}", color='#94a3b8', weight='bold',
            fontsize=18, ha='right', va='center', transform=ax.transAxes)

    # Sub-header Screen cue
    ax.text(0.05, 0.77, f"Screen Visual: {screen_desc}", color='#4ade80', weight='bold',
            fontsize=18, transform=ax.transAxes)

    # Bullet points with big text
    y_pos = 0.65
    for bp in bullet_points:
        formatted_bp = bp.format(**values)
        ax.text(0.06, y_pos, f"•  {formatted_bp}", color='white', weight='bold',
                fontsize=22, transform=ax.transAxes, wrap=True)
        y_pos -= 0.12

    # Footer
    ax.text(0.05, 0.05, "LunaProof — ISRO SIH26166 | Team Maximus2 (185903)", color='#64748b',
            weight='bold', fontsize=16, transform=ax.transAxes)

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=120, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Generated {filename}")

def build_all_cards(values=None):
    scenes = [
        (1, "Hook + Problem", "Cyrus", "0:00-0:25", "Title slide -> Fig 1 (Sun East vs West)",
         ["Chandrayaan-2 registers images across extreme illumination changes.",
          "Moon has no atmosphere => no fill light => shadows flip completely.",
          "Classical gradient descriptors break down under illumination shift."]),

        (2, "Why the Moon Breaks Matching", "Shawn", "0:25-0:50", "Fig 1 Zoomed on Shadow Flip",
         ["Illumination flip rotates brightness gradients by 180 degrees.",
          "Classical SIFT success rate drops to {gap_180_SIFT}% under 180° flip.",
          "Phase-Congruency SIFT measured {gap_180_PC}% under 180° flip in our held-out test."]),

        (3, "Our Method (One Flow)", "Chrisel", "0:50-1:20", "flow_pipeline.png",
         ["Lommel-Seeliger shading + ray-traced cast shadow physics.",
          "Kovesi log-Gabor Phase Congruency maps extract edge structures.",
          "MAGSAC++ robust fitting + sub-pixel checkpoint error + Refusal Gate."]),

        (4, "Real Data + Honest Test Design", "Aaron", "1:20-1:55", "Fig 0 (Real DEM Tiles) + Split Table",
         ["NASA LRO LOLA LDEM_64 elevation tiles across {n_tiles} regions.",
          "Train, validation, and test splits are strictly separate lunar regions.",
          "Code asserts zero spatial tile overlap between split roles."]),

        (5, "Results Including Failures", "Giselle", "1:55-2:30", "Fig 3 (Sun Gap) -> Fig 5 (Checkerboard)",
         ["SIFT: {SIFT_pct}% | PC-SIFT: {PCSIFT_pct}% | LoFTR: {LOFTR_pct}% | PatchNet: {LEARNED_pct}%.",
          "Hard region (60°-120° sun gap): {worst_method} drops to {worst_pct}%.",
          "Checkerboard showcase median error: {showcase_err} pixels."]),

        (6, "Refusal Gate + Cascade Roadmap", "Chris", "2:30-2:50", "Fig 4 (Gate) + flow_cascade.png",
         ["Gate Reliability: Accepted 'SUCCESS' pairs correct {gate_success_pct}% of the time.",
          "Refusal gate filters degraded or ill-conditioned registrations.",
          "Roadmap: IIRS 80m -> TMC-2 5m -> LRO NAC 1m -> OHRC 0.25m."]),

        (7, "Close", "Cyrus", "2:50-3:05", "Title Slide + Repository Link",
         ["LunaProof: Proof-backed registration for lunar exploration.",
          "Fully reproducible benchmark & notebook on GitHub.",
          "Team Maximus2 (ID 185903) | Thank you!"])
    ]

    for sc in scenes:
        fname = f"video/scenes/scene_{sc[0]}.png"
        generate_card(sc[0], sc[1], sc[2], sc[3], sc[4], sc[5], fname, values)

if __name__ == '__main__':
    build_all_cards()
