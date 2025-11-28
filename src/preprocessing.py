import cv2
import numpy as np
from pathlib import Path

def load_image(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(str(p))
    if img is None:
        raise IOError(f"OpenCV could not read the file: {path}")
    return img

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def resize_keep_aspect(img, max_side=1600):
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    return img

def apply_clahe(gray, clip_limit=3.0, tile_grid=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray)

def denoise_nlmeans(gray, h=10):
    return cv2.fastNlMeansDenoising(gray, None, h, 7, 21)

def adaptive_thresh(gray, block_size=15, c=10):
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block_size, c)

def deskew(gray):
    coords = np.column_stack(np.where(gray < 250))
    if coords.shape[0] < 10:
        return gray, 0.0
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    h, w = gray.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle

def preprocess_for_ocr(image_path):
    img = load_image(image_path)
    img = resize_keep_aspect(img)
    gray = to_gray(img)
    gray = apply_clahe(gray)
    gray = denoise_nlmeans(gray)
    gray, angle = deskew(gray)
    bin_img = adaptive_thresh(gray)
    return {
        "original": img,
        "gray": gray,
        "binarized": bin_img,
        "deskew_angle": angle
    }

