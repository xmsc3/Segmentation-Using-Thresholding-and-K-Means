import cv2
import os
import matplotlib.pyplot as plt

def load_image(filepath):
    img = cv2.imread(filepath, 0)
    if img is None:
        raise FileNotFoundError(f"Error: Could not load image from {filepath}")
    return img

def save_single_image(img, folder, filename, title=None, cmap='gray'):
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap=cmap)
    
    if title:
        plt.title(title)
    plt.axis('off')
    
    save_path = os.path.join(folder, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f">> Saved: {filename}")

def save_histogram(img, folder, filename, title):
    plt.figure(figsize=(8, 5))
    plt.hist(img.ravel(), bins=256, range=[0, 256], color='black', alpha=0.7)
    plt.title(title)
    plt.xlabel("Intensity Level"); plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(folder, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f">> Saved: {filename}")