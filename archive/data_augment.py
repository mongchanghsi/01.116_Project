import argparse
import cv2
import numpy
import imutils

# Set Arguments Parser
ap = argparse.ArgumentParser()
ap.add_argument("-i1", "--image1", required=True, help="path to 1st image that will be processed by OCR / tesseract")
ap.add_argument("-i2", "--image2", required=True, help="path to 2nd image that will be processed by OCR / tesseract")
ap.add_argument("-d", "--degree", type=str, default=5, help="rotation in degree")
args = vars(ap.parse_args())

# The image is loaded into memory – Python kernel
image1 = cv2.imread(args["image1"])
image2 = cv2.imread(args["image2"])

image1 = imutils.rotate_bound(image1, int(args["degree"]))
image2 = imutils.rotate_bound(image2, int(args["degree"]))

# write the new rotated image to disk 
filename1 = "./test_images/image_augment_test/top_{}.png".format(args["degree"])
filename2 = "./test_images/image_augment_test/back_{}.png".format(args["degree"])
cv2.imwrite(filename1, image1)
cv2.imwrite(filename2, image2)