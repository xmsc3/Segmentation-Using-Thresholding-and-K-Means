import matplotlib.pyplot as plt
import os
from functions import *


def main():
    image_path = 'images/image_4.jpeg'
    output_dir = 'results'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    try:
        img = load_image(image_path)
    except FileNotFoundError as e:
        print(e)
        return

    # PHASE 1:  LOAD & ANALYZE IMAGE
    print("--- Phase 1: Original Image Analysis ---")
    
    plt.figure()
    plt.hist(img.ravel(), bins=256, range=[0, 256], color='black', alpha=0.7)
    plt.title("Original Histogram")
    plt.show()
    
    save_histogram(img, output_dir, '01_histogram_original.png', "Original Histogram")
    save_single_image(img, output_dir, '00_original_image.png', "Original Image", cmap='gray')

    try:
        val_str = input(">> Enter Manual Threshold value (0-255): ")
        manual_T = int(val_str)
    except ValueError:
        manual_T = 127
        print("Invalid input. Defaulting to 127.")

    # PHASE 2: BASIC SEGMENTATION METHODS
    print("--- Phase 2: Basic Segmentation Methods ---")

    # 1. Global Manual
    img_manual = apply_global_threshold(img, manual_T)
    save_single_image(img_manual, output_dir, f'02_manual_threshold_T{manual_T}.png', f"Manual Threshold (T={manual_T})")

    # 2. Global Otsu
    otsu_val, img_otsu = apply_otsu_threshold(img)
    save_single_image(img_otsu, output_dir, '03_otsu_threshold.png', f"Otsu (T={int(otsu_val)})")

    # 3. Local Adaptive
    img_adaptive = apply_adaptive_threshold(img, block_size=11, C=2)
    save_single_image(img_adaptive, output_dir, '04_adaptive_threshold.png', "Adaptive Threshold")

    # 4. K-Means
    img_kmeans = apply_kmeans(img, k=3)
    save_single_image(img_kmeans, output_dir, '05_kmeans_clustering.png', "K-Means (K=3)")

    # 5. Canny Edges
    edges = apply_canny_edge(img, low=100, high=200)
    save_single_image(edges, output_dir, '06_canny_edges.png', "Canny Edges", cmap='gray')

    # PHASE 3: EQUALIZATION
    print("--- Phase 3: Preprocessing Impact (Histogram Equalization) ---")

    img_eq = apply_histogram_equalization(img)
    save_single_image(img_eq, output_dir, '07_image_equalized.png', "Equalized Image")
    save_histogram(img_eq, output_dir, '08_histogram_equalized.png', "Equalized Histogram")

    print(">> Analyze the Equalized Histogram.")
    plt.figure()
    plt.hist(img_eq.ravel(), bins=256, range=[0, 256], color='black', alpha=0.7)
    plt.title("Equalized Histogram")
    plt.show()

    try:
        val_str_eq = input(">> Enter New Manual Threshold for Equalized Image (0-255): ")
        manual_T_eq = int(val_str_eq)
    except ValueError:
        manual_T_eq = 127

    # 1. Global Manual Equalized
    img_manual_eq = apply_global_threshold(img_eq, manual_T_eq)
    save_single_image(img_manual_eq, output_dir, f'09_manual_eq_T{manual_T_eq}.png', f"Global Manual Equalized (T={manual_T_eq})")

    # 2. Global Otsu Equalized
    otsu_val_eq, otsu_eq = apply_otsu_threshold(img_eq)
    save_single_image(otsu_eq, output_dir, '10_otsu_equalized.png', f"Global Otsu Equalized (T={int(otsu_val_eq)})")
    # 3. Local Adaptive Equalized
    adaptive_eq = apply_adaptive_threshold(img_eq, block_size=11, C=2)
    save_single_image(adaptive_eq, output_dir, '11_adaptive_equalized.png', "Local Adaptive (Equalized)")

    # 4. K-Means Equalized
    kmeans_eq = apply_kmeans(img_eq, k=3)
    save_single_image(kmeans_eq, output_dir, '12_kmeans_equalized.png', "K-Means (Equalized)")

    # 5. Canny Equalized
    edges_eq = apply_canny_edge(img_eq, low=100, high=200)
    save_single_image(edges_eq, output_dir, '13_canny_edges_equalized.png', "Canny Edges (Equalized)")

    print(f"\nAll images saved in folder: '{output_dir}'.")

if __name__ == "__main__":
    main()