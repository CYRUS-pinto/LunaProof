import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

os.makedirs('assets', exist_ok=True)

# --------------------------------------------------------- 1. Pipeline Flow Diagram
def draw_flow_pipeline():
    fig, ax = plt.subplots(figsize=(14, 3.5), dpi=300)
    ax.set_facecolor('#0f172a')
    fig.patch.set_facecolor('#0f172a')
    
    boxes = [
        ("DEM / Image", "NASA LOLA\nLDEM_64 (474m)", "#1e293b", "#38bdf8"),
        ("Physics Render", "Lommel-Seeliger +\nCast Shadows", "#1e293b", "#38bdf8"),
        ("Phase Congruency", "Log-Gabor Illum-\nInvariant Map", "#1e293b", "#38bdf8"),
        ("Multi-Matcher", "SIFT / LoFTR /\nLearned Descriptor", "#1e293b", "#38bdf8"),
        ("MAGSAC++", "Robust Fitting &\nHomography", "#1e293b", "#38bdf8"),
        ("Checkpoint & Gate", "Ground Truth Error\n+ Refusal Gate", "#1e293b", "#4ade80")
    ]
    
    n = len(boxes)
    for i, (title, sub, bg, border) in enumerate(boxes):
        x = i * 2.2 + 0.5
        y = 0.8
        w, h = 1.8, 1.4
        
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                  fc=bg, ec=border, lw=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h*0.65, title, color="white", weight="bold", 
                ha="center", va="center", fontsize=11)
        ax.text(x + w/2, y + h*0.3, sub, color="#94a3b8", ha="center", 
                va="center", fontsize=8.5)
        
        if i < n - 1:
            ax.annotate("", xy=(x + w + 0.4, y + h/2), xytext=(x + w + 0.05, y + h/2),
                        arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=2.5))

    ax.set_xlim(0, n * 2.2 + 0.3)
    ax.set_ylim(0, 3.0)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('assets/flow_pipeline.png', dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.savefig('assets/flow_pipeline.svg', facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print("Generated flow_pipeline.png and .svg")

# --------------------------------------------------------- 2. Scale Cascade Flow Diagram
def draw_flow_cascade():
    fig, ax = plt.subplots(figsize=(12, 3.5), dpi=300)
    ax.set_facecolor('#0f172a')
    fig.patch.set_facecolor('#0f172a')
    
    nodes = [
        ("IIRS", "80 m", "(roadmap)", "#3b82f6"),
        ("TMC-2", "5 m", "(roadmap)", "#8b5cf6"),
        ("LRO NAC", "1 m (bridge)", "(roadmap)", "#ec4899"),
        ("OHRC", "0.25 m", "(roadmap)", "#f43f5e")
    ]
    
    hops = ["16x hop", "5x hop", "4x hop"]
    
    for i, (name, res, note, color) in enumerate(nodes):
        x = i * 2.8 + 0.6
        y = 0.8
        w, h = 2.0, 1.4
        
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                  fc='#1e293b', ec=color, lw=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h*0.7, name, color="white", weight="bold", 
                ha="center", va="center", fontsize=12)
        ax.text(x + w/2, y + h*0.4, res, color="#cbd5e1", ha="center", 
                va="center", fontsize=9.5)
        ax.text(x + w/2, y + h*0.18, note, color="#f43f5e", weight="bold", 
                ha="center", va="center", fontsize=8)
        
        if i < len(nodes) - 1:
            ax.annotate("", xy=(x + w + 0.75, y + h/2), xytext=(x + w + 0.05, y + h/2),
                        arrowprops=dict(arrowstyle="->", color="#64748b", lw=2, linestyle='--'))
            ax.text(x + w + 0.4, y + h/2 + 0.25, hops[i], color="#94a3b8", 
                    ha="center", va="center", fontsize=8.5, weight="bold")

    ax.set_xlim(0, len(nodes) * 2.8 + 0.2)
    ax.set_ylim(0, 3.0)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('assets/flow_cascade.png', dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.savefig('assets/flow_cascade.svg', facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print("Generated flow_cascade.png and .svg")

# --------------------------------------------------------- 3. Techstack Badges
def draw_techstack():
    fig, ax = plt.subplots(figsize=(14, 2.5), dpi=300)
    ax.set_facecolor('#0f172a')
    fig.patch.set_facecolor('#0f172a')
    
    techs = [
        ("Python", "#3776AB"),
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
    for i, (name, color) in enumerate(techs):
        x = (i % 5) * 2.6 + 0.4
        y = 1.4 if i < 5 else 0.3
        w, h = 2.3, 0.8
        
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", 
                                  fc=color, ec="white", lw=1)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, name, color="white", weight="bold", 
                ha="center", va="center", fontsize=9)

    ax.set_xlim(0, 5 * 2.6 + 0.3)
    ax.set_ylim(0, 2.5)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('assets/techstack.png', dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.savefig('assets/techstack.svg', facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print("Generated techstack.png and .svg")

if __name__ == '__main__':
    draw_flow_pipeline()
    draw_flow_cascade()
    draw_techstack()
