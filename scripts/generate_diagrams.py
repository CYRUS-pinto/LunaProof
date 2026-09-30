import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as patheffects
import numpy as np

os.makedirs('assets', exist_ok=True)

def generate_flow_pipeline():
    fig, ax = plt.subplots(figsize=(14, 3.5), dpi=300)
    ax.set_facecolor('#0F172A')
    fig.patch.set_facecolor('#0F172A')
    ax.axis('off')

    steps = [
        ("DEM / Image", "LOLA LDEM 474m/px\nReal Lunar Relief", "#1E293B", "#38BDF8"),
        ("Physics Render", "Lommel-Seeliger Shading\n+ Cast Shadow Marching", "#1E293B", "#38BDF8"),
        ("Phase Congruency", "Kovesi Log-Gabor\nIllumination-Invariant Edges", "#1E293B", "#818CF8"),
        ("Feature Matching", "SIFT / LoFTR / Learned\nPatchNet Descriptor", "#1E293B", "#C084FC"),
        ("MAGSAC++", "Robust Homography\nTransformation Matrix H", "#1E293B", "#F472B6"),
        ("Checkpoint & Gate", "Held-Out Checkpoints\nRefusal Gate: SUCCESS/REFUSED", "#1E293B", "#34D399")
    ]

    x_starts = np.linspace(0.05, 0.82, len(steps))
    box_w = 0.12
    box_h = 0.55

    for i, (title, desc, fill_col, border_col) in enumerate(steps):
        x = x_starts[i]
        y = 0.22
        rect = patches.FancyBboxPatch((x, y), box_w, box_h, boxstyle="round,pad=0.03,rounding_size=0.04",
                                      facecolor=fill_col, edgecolor=border_col, linewidth=2, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + box_w/2, y + box_h*0.7, title, color="#F8FAFC", fontsize=10, fontweight='bold', ha='center', va='center', transform=ax.transAxes)
        ax.text(x + box_w/2, y + box_h*0.32, desc, color="#94A3B8", fontsize=7.5, ha='center', va='center', transform=ax.transAxes)

        if i < len(steps) - 1:
            ax.annotate('', xy=(x_starts[i+1] - 0.005, y + box_h/2), xytext=(x + box_w + 0.005, y + box_h/2),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle="->", color="#64748B", lw=2, mutation_scale=15))

    ax.set_title("LunaProof: Physics-Informed Cross-Illumination Registration Pipeline", color="#F8FAFC", fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('assets/flow_pipeline.png', dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.savefig('assets/flow_pipeline.svg', facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

def generate_flow_cascade():
    fig, ax = plt.subplots(figsize=(13, 3.5), dpi=300)
    ax.set_facecolor('#0F172A')
    fig.patch.set_facecolor('#0F172A')
    ax.axis('off')

    nodes = [
        ("IIRS (~80 m)", "Hyperspectral Base\n(roadmap)", "#1E293B", "#F59E0B", "16x hop"),
        ("TMC-2 (5 m)", "Stereo / Terrain Context\n(roadmap)", "#1E293B", "#3B82F6", "5x hop"),
        ("LRO NAC (1 m)", "Optional Bridge Layer\n(roadmap)", "#1E293B", "#6366F1", "4x hop"),
        ("OHRC (0.25 m)", "High-Res Target\n(roadmap)", "#1E293B", "#10B981", "")
    ]

    x_positions = [0.08, 0.33, 0.58, 0.81]
    box_w = 0.15
    box_h = 0.55

    for i, (title, desc, fill_col, border_col, hop_label) in enumerate(nodes):
        x = x_positions[i]
        y = 0.22
        rect = patches.FancyBboxPatch((x, y), box_w, box_h, boxstyle="round,pad=0.03,rounding_size=0.04",
                                      facecolor=fill_col, edgecolor=border_col, linewidth=2, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + box_w/2, y + box_h*0.7, title, color="#F8FAFC", fontsize=10, fontweight='bold', ha='center', va='center', transform=ax.transAxes)
        ax.text(x + box_w/2, y + box_h*0.32, desc, color="#94A3B8", fontsize=8, ha='center', va='center', transform=ax.transAxes)

        if i < len(nodes) - 1:
            next_x = x_positions[i+1]
            ax.annotate('', xy=(next_x - 0.005, y + box_h/2), xytext=(x + box_w + 0.005, y + box_h/2),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle="->", color="#38BDF8", lw=2, mutation_scale=15))
            ax.text((x + box_w + next_x)/2, y + box_h/2 + 0.08, hop_label, color="#38BDF8", fontsize=8, fontweight='bold', ha='center', va='center', transform=ax.transAxes)

    ax.set_title("Multi-Modal Resolution Cascade Architecture (Roadmap)", color="#F8FAFC", fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('assets/flow_cascade.png', dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.savefig('assets/flow_cascade.svg', facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

def generate_techstack():
    fig, ax = plt.subplots(figsize=(14, 2.2), dpi=300)
    ax.set_facecolor('#0F172A')
    fig.patch.set_facecolor('#0F172A')
    ax.axis('off')

    techs = [
        ("Python 3.13", "#3776AB"),
        ("PyTorch", "#EE4C2C"),
        ("OpenCV", "#5C3EE8"),
        ("NumPy", "#013243"),
        ("SciPy", "#8CAAE6"),
        ("Google Colab", "#F9AB00"),
        ("GitHub", "#24292E"),
        ("Jupyter", "#F37626"),
        ("ISRO Chandrayaan-2 data", "#FF9933"),
        ("NASA LRO LOLA", "#0B3D91")
    ]

    n = len(techs)
    x_positions = np.linspace(0.02, 0.98 - 0.08, n)
    box_w = 0.082
    box_h = 0.55

    for i, (name, col) in enumerate(techs):
        x = x_positions[i]
        y = 0.22
        rect = patches.FancyBboxPatch((x, y), box_w, box_h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                      facecolor="#1E293B", edgecolor=col, linewidth=2, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + box_w/2, y + box_h/2, name, color="#F8FAFC", fontsize=7.5, fontweight='bold', ha='center', va='center', wrap=True, transform=ax.transAxes)

    ax.set_title("Technology Stack & Data Sources", color="#F8FAFC", fontsize=11, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig('assets/techstack.png', dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.savefig('assets/techstack.svg', facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

def generate_why_sift_fails():
    if os.path.exists('results/fig1_sun_flip_physics.png'):
        shutil.copy('results/fig1_sun_flip_physics.png', 'assets/why_sift_fails.png')
        print("Copied results/fig1_sun_flip_physics.png to assets/why_sift_fails.png")
    else:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
        ax.set_facecolor('#0F172A')
        ax.text(0.5, 0.5, "Why SIFT Fails under Sun Flips\n(Shadow Inversion & Gradient Reversal)", color="#F8FAFC", fontsize=12, ha='center', va='center')
        ax.axis('off')
        plt.savefig('assets/why_sift_fails.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    generate_flow_pipeline()
    generate_flow_cascade()
    generate_techstack()
    generate_why_sift_fails()
    print("Diagram assets generated cleanly in assets/")
