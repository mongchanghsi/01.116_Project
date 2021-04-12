import cv2

def threshold(image):
  # First arg is the source image in grayscale
  # Second arg is the threshold
  # Third arg is the maxVal that the pixel should be assigned to if it exceeded the threshold value
  # Fourth arg is the style of thresholding
  return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

def adaptiveMeanThreshold(image):
  image = cv2.GaussianBlur(image, (3,3),0)
  return cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)

def adaptiveGaussianThreshold(image):
  image = cv2.GaussianBlur(image, (3,3),0)
  return cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

def gaussianBlur_threshold(image):
  image = cv2.GaussianBlur(image, (3,3),0)
  return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]