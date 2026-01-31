import cv2
import os

class Router:
    def route(self, image_path):
        # Logic đơn giản: Ảnh nhỏ hoặc trong tập CIFAR -> SIMPLE
        # Ảnh to -> COMPLEX
        try:
            img = cv2.imread(image_path)
            if img is None: return "SIMPLE" # Fallback
            h, w, _ = img.shape
            if h < 64 or w < 64:
                return "SIMPLE"
            return "COMPLEX"
        except Exception:
            return "SIMPLE"