import cv2
import numpy as np

def apply_global_threshold(img, threshold_value):
    _, binary_img = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_img

def apply_otsu_threshold(img):
    val, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"Otsu calculated threshold: T={val}")
    return val, binary_img

def apply_adaptive_threshold(img, block_size=11, C=2):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)

def apply_kmeans(img, k=3):
    pixel_values = img.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()].reshape(img.shape)
    return segmented_image

def apply_canny_edge(img, low=100, high=200):
    return cv2.Canny(img, low, high)

def apply_histogram_equalization(img):
    return cv2.equalizeHist(img)