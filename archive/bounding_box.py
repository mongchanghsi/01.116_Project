import os
import pytesseract
import argparse
import cv2

from utils.preprocessing.blur import medianBlur, averageBlur, gaussianBlur, bilateralBlur
from utils.preprocessing.threshold import threshold, adaptiveMeanThreshold, adaptiveGaussianThreshold, gaussianBlur_threshold

# Set Arguments Parser
ap = argparse.ArgumentParser()
ap.add_argument("-i1", "--image1", required=True, help="path to 1st image that will be processed by OCR / tesseract")
ap.add_argument("-i2", "--image2", required=True, help="path to 1st image that will be processed by OCR / tesseract")
args = vars(ap.parse_args())

# The image is loaded into memory – Python kernel
image1 = cv2.imread(args["image1"])
image2 = cv2.imread(args["image2"])

# Convert to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Preprocessing function - produces 2 versions of using Gaussian Blur only and Gaussian Blur and Binary Threshold
def preprocess(input1, input2):
  print('Undergoing Gaussian Blur Preprocessing')
  output_GB_1, output_GB_2 = medianBlur(input1), medianBlur(input2)

  print('Undergoing Gaussian Blurring with Threshold Preprocessing')
  output_GBT_1, output_GBT_2 = gaussianBlur_threshold(input1), gaussianBlur_threshold(input2)

  return output_GB_1, output_GB_2, output_GBT_1, output_GBT_2

# Start of the script
gray_GB_1, gray_GB_2, gray_GBT_1, gray_GBT_2 = preprocess(gray1, gray2)

# Save the processed images into the disk
filename1 = "./processed_images/{}_GB_1_Box.png".format(os.getpid())
filename2 = "./processed_images/{}_GB_2_Box.png".format(os.getpid())
filename3 = "./processed_images/{}_GBT_1_Box.png".format(os.getpid())
filename4 = "./processed_images/{}_GBT_2_Box.png".format(os.getpid())

cv2.imwrite(filename1, gray_GB_1)
cv2.imwrite(filename2, gray_GB_2)
cv2.imwrite(filename3, gray_GBT_1)
cv2.imwrite(filename4, gray_GBT_2)

image_1 = cv2.imread(filename1)
image_2 = cv2.imread(filename2)
image_3 = cv2.imread(filename3)
image_4 = cv2.imread(filename4)

gray_GB_1_box = pytesseract.image_to_boxes(image_1)
gray_GB_2_box = pytesseract.image_to_boxes(image_2)
gray_GBT_1_box = pytesseract.image_to_boxes(image_3)
gray_GBT_2_box = pytesseract.image_to_boxes(image_4)

h1,w1,c1 = o1.shape
h2,w2,c2 = o2.shape
h3,w3,c3 = o3.shape
h4,w4,c4 = o4.shape

for b in gray_GB_1_box.splitlines():
    b = b.split(' ')
    box_image1 = cv2.rectangle(image_1, (int(b[1]), h1 - int(b[2])), (int(b[3]), h1 - int(b[4])), (0, 255, 0), 2)

for b in gray_GB_2_box.splitlines():
    b = b.split(' ')
    box_image2 = cv2.rectangle(image_2, (int(b[1]), h2 - int(b[2])), (int(b[3]), h2 - int(b[4])), (0, 255, 0), 2)

for b in gray_GBT_1_box.splitlines():
    b = b.split(' ')
    box_image3 = cv2.rectangle(image_3, (int(b[1]), h3 - int(b[2])), (int(b[3]), h3 - int(b[4])), (0, 255, 0), 2)

for b in gray_GBT_2_box.splitlines():
    b = b.split(' ')
    box_image4 = cv2.rectangle(image_4, (int(b[1]), h4 - int(b[2])), (int(b[3]), h4 - int(b[4])), (0, 255, 0), 2)

cv2.imwrite(filename1, box_image1)
cv2.imwrite(filename2, box_image2)
cv2.imwrite(filename3, box_image3)
cv2.imwrite(filename4, box_image4)