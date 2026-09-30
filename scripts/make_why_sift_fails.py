import os
import matplotlib.pyplot as plt
import numpy as np

def make_stub():
    os.makedirs('assets', exist_ok=True)
    if os.path.exists('results/fig1_sun_flip_physics.png'):
        from PIL import Image
        img = Image.open('results/fig1_sun_flip_physics.png')
        img.save('assets/why_sift_fails.png')
        print("Copied results/fig1_sun_flip_physics.png to assets/why_sift_fails.png")
    else:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
        ax.set_facecolor('#1e293b')
        fig.patch.set_facecolor('#0f172a')
        ax.text(0.5, 0.5, "Why SIFT Fails\n(Awaiting results/fig1_sun_flip_physics.png from Prompt B)",
                color='white', weight='bold', fontsize=14, ha='center', va='center')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig('assets/why_sift_fails.png', dpi=300, facecolor=fig.get_facecolor())
        plt.close()
        print("Generated stub assets/why_sift_fails.png")

if __name__ == '__main__':
    make_stub()
