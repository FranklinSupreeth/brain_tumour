"""
Feature extraction pipeline — must exactly mirror the training notebook,
otherwise the scaler/model input shape and distribution will not match.

Pipeline: read image -> resize 128x128 -> grayscale
          -> HOG + LBP + GLCM -> concatenate into one feature vector
"""

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops

# Must match training notebook exactly
IMG_SIZE = 128
RADIUS = 2
N_POINTS = 8 * RADIUS


def preprocess_image_from_array(image_bgr):
    """Resize + grayscale an already-loaded BGR image (as cv2.imread returns)."""
    image = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def extract_hog(image):
    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True
    )
    return features


def extract_lbp(image):
    lbp = local_binary_pattern(image, N_POINTS, RADIUS, method='uniform')
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, N_POINTS + 3),
        range=(0, N_POINTS + 2)
    )
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist


def extract_glcm(image):
    glcm = graycomatrix(
        image,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    asm = graycoprops(glcm, 'ASM')[0, 0]
    return np.array([contrast, dissimilarity, homogeneity, energy, correlation, asm])


def extract_features_from_array(image_bgr):
    """Full pipeline for a single image already loaded into memory (BGR array)."""
    image = preprocess_image_from_array(image_bgr)
    hog_features = extract_hog(image)
    lbp_features = extract_lbp(image)
    glcm_features = extract_glcm(image)
    features = np.concatenate([hog_features, lbp_features, glcm_features])
    return features


def extract_features_from_path(image_path):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Could not read image at {image_path}")
    return extract_features_from_array(image_bgr)
