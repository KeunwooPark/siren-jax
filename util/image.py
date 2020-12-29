import cv2

def gradient(img):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    return (sobel_x + sobel_y) / 2
