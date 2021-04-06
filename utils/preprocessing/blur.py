import cv2

def medianBlur(image):
  return cv2.medianBlur(image, 3)

def averageBlur(image):
  return cv2.blur(image, (3,3))

def gaussianBlur(image):
  return cv2.GaussianBlur(image, (3,3), 0)

def bilateralBlur(image):
  return cv2.bilateralFilter(image, 5, 75, 75)