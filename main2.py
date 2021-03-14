# Tutorial from https://www.tensorscience.com/ocr/optical-character-recognition-ocr-with-python-and-tesseract-4-an-introduction
import os
from PIL import Image
import pytesseract
import argparse
import cv2

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pprint import pprint

from utils.isDate import isDate
from utils.isDiopter import isDiopter
from utils.isBrand import isBrand, isSimilarBrand
from utils.isModel import isModel
from utils.isSerial import isSerial, isSerial_2

# Google Sheet Set up
client = gspread.service_account(filename='credentials.json')
sheet = client.open('Mock_EMR').sheet1
sheet_data = sheet.get_all_records()

# Set Arguments Parser
ap = argparse.ArgumentParser()
ap.add_argument("-i1", "--image1", required=True, help="path to 1st image that will be processed by OCR / tesseract")
ap.add_argument("-i2", "--image2", required=True, help="path to 1st image that will be processed by OCR / tesseract")
ap.add_argument("-p", "--preprocess", type=str, default="blur", help="preprocessing method that is applied to the image")
args = vars(ap.parse_args())

# The image is loaded into memory – Python kernel
image1 = cv2.imread(args["image1"])
image2 = cv2.imread(args["image2"])

# Convert to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# preprocess the image
if args["preprocess"] == "thresh": gray1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
if args["preprocess"] == "thresh": gray2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# blur the image to remove noise
elif args["preprocess"] == "blur": gray1 = cv2.medianBlur(gray1, 3)
elif args["preprocess"] == "blur": gray2 = cv2.medianBlur(gray2, 3)

# write the new grayscale image to disk 
filename1 = "./processed_images/{}_1.png".format(os.getpid())
filename2 = "./processed_images/{}_2.png".format(os.getpid())
cv2.imwrite(filename1, gray1)
cv2.imwrite(filename2, gray2)

# load the image as a PIL/Pillow image, apply OCR 
text1 = pytesseract.image_to_string(Image.open(filename1))
text2 = pytesseract.image_to_string(Image.open(filename2))

information1 = text1.split()
information2 = text2.split()
info = information1 + information2
print(info)

metadata = {'brand': '', 'model': '', 'expirydate': '', 'serialnumber': '', 'diopter': ''}

# Round 1 of looping gather any information
for i in info:
  if isDate(i):
    metadata['expirydate'] = i
  if isDiopter(i):
    metadata['diopter'] = i
  if isBrand(i):
    metadata['brand'] = i.upper()

# Round 2 is using the identified brand to narrow down on the model to find
if metadata['brand'] != '':
  for i in info:
    if isModel(i, metadata['brand']):
      metadata['model'] = i
    if isSerial(i, metadata['brand']):
      metadata['serialnumber'] = i

# Insert a part 2 check, where assuming that the serial number had broken up into 2
if metadata['serialnumber'] == '':
  for i in range(len(info)-2):
    if isSerial_2(info[i], info[i+1], metadata['brand']):
      combined_serial = info[i] + info[i+1]
      metadata['serialnumber'] = combined_serial

print(metadata)

shouldUpdate = True
for i in metadata.values():
  if i == '':
    shouldUpdate = False
  
if shouldUpdate:
  entry_number = len(sheet_data) + 2
  # row = list(metadata.values())
  # sheet.insert_row(row, entry_number)
  print('Updated Google Sheets')
else:
  print('Missing Information, did not update Google Sheets')
