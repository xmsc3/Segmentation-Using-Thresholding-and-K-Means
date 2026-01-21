import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

def load_image(filepath):
    img = cv2.imread(filepath, 0)
    if img is None:
        raise FileNotFoundError(f"Error: Could not load image from {filepath}")
    return img

def create_geometric_figure(img):
    S = 150 
    img_sq = cv2.resize(img, (S, S), interpolation=cv2.INTER_CUBIC)
    
    apothem = (S / 2) * (1 + np.sqrt(2))
    canvas_size = int((apothem + S) * 2)
    center_pt = canvas_size // 2
    
    final_canvas = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255
    
    center_img_size = int(apothem * 2) + 4 
    img_center = cv2.resize(img, (center_img_size, center_img_size), interpolation=cv2.INTER_CUBIC)
    
    c_offset = center_pt - (center_img_size // 2)
    final_canvas[c_offset:c_offset+center_img_size, c_offset:c_offset+center_img_size] = img_center

    base_layer = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    base_mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    
    img_petal_base = cv2.rotate(img_sq, cv2.ROTATE_180)
    
    dist_to_petal_center = apothem + (S / 2)
    y_pos = int(center_pt - dist_to_petal_center - (S / 2))
    x_pos = int(center_pt - (S / 2))
    
    y_pos = max(0, y_pos)
    x_pos = max(0, x_pos)
    
    h, w = img_petal_base.shape
    h_end = min(y_pos + h, canvas_size)
    w_end = min(x_pos + w, canvas_size)
    
    base_layer[y_pos:h_end, x_pos:w_end] = img_petal_base[:h_end-y_pos, :w_end-x_pos]
    base_mask[y_pos:h_end, x_pos:w_end] = 255

    for angle in range(0, 360, 45):
        M = cv2.getRotationMatrix2D((center_pt, center_pt), angle, 1.0)
        
        rotated_petal = cv2.warpAffine(base_layer, M, (canvas_size, canvas_size), flags=cv2.INTER_LINEAR)
        rotated_mask = cv2.warpAffine(base_mask, M, (canvas_size, canvas_size), flags=cv2.INTER_NEAREST)
        
        np.copyto(final_canvas, rotated_petal, where=(rotated_mask > 0))

    return final_canvas

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