# Tutorial from https://www.tensorscience.com/ocr/optical-character-recognition-ocr-with-python-and-tesseract-4-an-introduction
import os
from PIL import Image
import pytesseract
import argparse
import cv2
import time

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pprint import pprint

from utils.isDate import isDate
from utils.isDiopter import isDiopter, processDiopter
from utils.isBrand import isBrand, brandSimilarity
from utils.isModel import isModel, modelSimilarity
from utils.isSerial import isSerial, isSerial_2, checkSerialSize
from utils.isBatch import isBatch, batchSimilarity

from utils.preprocessing.blur import medianBlur, averageBlur, gaussianBlur, bilateralBlur
from utils.preprocessing.threshold import threshold, adaptiveMeanThreshold, adaptiveGaussianThreshold, gaussianBlur_threshold

# Google Sheet Set up
client = gspread.service_account(filename='credentials.json')
sheet = client.open('Mock_EMR').sheet1
sheet_data = sheet.get_all_records()

# Set Arguments Parser
ap = argparse.ArgumentParser()
ap.add_argument("-i1", "--image1", required=True, help="path to 1st image that will be processed by OCR / tesseract")
ap.add_argument("-i2", "--image2", required=True, help="path to 2nd image that will be processed by OCR / tesseract")
ap.add_argument("-p", "--preprocess", type=str, default="combi", help="preprocessing method that is applied to the image")
args = vars(ap.parse_args())

# The image is loaded into memory – Python kernel
image1 = cv2.imread(args["image1"])
image2 = cv2.imread(args["image2"])

# Convert to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

def preprocess(type_of_preprocess, input1, input2):
  if type_of_preprocess == "thresh":
    print('Undergoing Threshold Preprocessing')
    output1, output2 = adaptiveGaussianThreshold(input1), adaptiveGaussianThreshold(input2)
  elif type_of_preprocess == "blur":
    print('Undergoing Gaussian Blur Preprocessing')
    output1 = medianBlur(input1)
    output2 = medianBlur(input2)
  elif type_of_preprocess == "combi":
    print('Undergoing Gaussian Blurring with Threshold Preprocessing')
    output1, output2 = gaussianBlur_threshold(input1), gaussianBlur_threshold(input2)
  else:
    output1, output2 = gray1, gray2
  return output1, output2

# Start of the script
start = time.time()
gray1, gray2 = preprocess(args["preprocess"], gray1, gray2)

# write the new grayscale image to disk 
filename1 = "./processed_images/{}_1.png".format(os.getpid())
filename2 = "./processed_images/{}_2.png".format(os.getpid())
cv2.imwrite(filename1, gray1)
cv2.imwrite(filename2, gray2)

# load the image as a PIL/Pillow image, apply OCR
text1 = pytesseract.image_to_string(Image.open(filename1))
text2 = pytesseract.image_to_string(Image.open(filename2))

image123 = cv2.imread(filename1)
image1234 = cv2.imread(filename2)
box1 = pytesseract.image_to_boxes(image123)
box2 = pytesseract.image_to_boxes(image1234)
print(type(box1))
# h1,w1,c1 = image123.shape
# h2,w2,c2 = image1234.shape

# for b in box1.splitlines():
#     b = b.split(' ')
#     box_image1 = cv2.rectangle(image123, (int(b[1]), h1 - int(b[2])), (int(b[3]), h1 - int(b[4])), (0, 255, 0), 2)

# cv2.imshow('img', box_image1)
# # cv2.waitKey(0)

# for b in box2.splitlines():
#     b = b.split(' ')
#     box_image2 = cv2.rectangle(image1234, (int(b[1]), h2 - int(b[2])), (int(b[3]), h2 - int(b[4])), (0, 255, 0), 2)

# cv2.imshow('img2', box_image2)
# cv2.waitKey(0)

information1 = text1.split()
information2 = text2.split()
info = information1 + information2
print(info)

metadata = {'brand': '', 'model': '', 'batch': '','expirydate': '', 'serialnumber': '', 'diopter': ''}

# Round 1 of looping gather any information
for i in info:
  if isDate(i):
    metadata['expirydate'] = i
  if isDiopter(i):
    metadata['diopter'] = processDiopter(i)
  if isBrand(i):
    metadata['brand'] = i.upper()

# Brand Similarity Score
if metadata['brand'] == '':
  similarity_brand_score = {}
  for i in info:
    if len(i) > 5:
      most_similar_brand, score = brandSimilarity(i)
      if score > 0:
        if most_similar_brand in similarity_brand_score.keys():
          if similarity_brand_score[most_similar_brand] < score:
            similarity_brand_score[most_similar_brand] = score
        else:
          similarity_brand_score[most_similar_brand] = score
  if (similarity_brand_score != {}):
    print(f'Brand Similarity Score - {similarity_brand_score}')
    metadata['brand'] = (max(similarity_brand_score, key=similarity_brand_score.get))

# Round 2 is using the identified brand to narrow down on the model to find
if metadata['brand'] != '':
  for i in info:
    if isModel(i, metadata['brand']):
      metadata['model'] = i
    if isSerial(i, metadata['brand']):
      metadata['serialnumber'] = i

  # Round 3a - Don't have a model, run a similarity score to get the model
  if metadata['model'] == '':
    similarity_model_score = {}
    for i in info:
      # Maybe need remove i.isalnum()
      if i.isalnum() and i.isalpha() == False and i.isdigit() == False and len(i) > 4 and len(i) < 12:
        most_similar_model, score = modelSimilarity(i, metadata['brand'])
        if score > 0:
          if most_similar_model in similarity_model_score.keys():
            if similarity_model_score[most_similar_model] < score:
              similarity_model_score[most_similar_model] = score
          else:
            similarity_model_score[most_similar_model] = score
    if (similarity_model_score != {}):
      print(f'Model Similarity Score - {similarity_model_score}')
      metadata['model'] = (max(similarity_model_score, key=similarity_model_score.get))

# Round 3b is using Model to detect the Batch Number
if metadata['model'] != '':
  for i in info:
    if isBatch(i, metadata['model']):
      metadata['batch'] = i
  if metadata['batch'] == '':
    similarity_batch_score = {}
    for i in info:
      # Maybe need remove i.isalnum()
      if len(i) >= 9:
        most_similar_batch, score = batchSimilarity(i, metadata['model'])
        if score > 0:
          if most_similar_batch in similarity_batch_score.keys():
            if similarity_batch_score[most_similar_batch] < score:
              similarity_batch_score[most_similar_batch] = score
          else:
            similarity_batch_score[most_similar_batch] = score
    print(f'Batch Similarity Score - {similarity_batch_score}')
    if (similarity_batch_score != {}):
      batch_name = (max(similarity_batch_score, key=similarity_batch_score.get))
      autocorrect_batch_name = metadata['model'] + batch_name[len(metadata['model']):]
      if (checkSerialSize(autocorrect_batch_name, metadata['brand']) == False):
        metadata['batch'] = autocorrect_batch_name + '0'
      else:
        metadata['batch'] = autocorrect_batch_name

# Insert a part 2 check, where assuming that the serial number had broken up into 2
if metadata['serialnumber'] == '':
  for i in range(len(info)-2):
    if isSerial_2(info[i], info[i+1], metadata['brand']):
      combined_serial = info[i] + info[i+1]
      metadata['serialnumber'] = combined_serial

print(f'Final Output after OCR - {metadata}')

shouldUpdate = True
for i in metadata.values():
  if i == '':
    shouldUpdate = False
  
if shouldUpdate:
  entry_number = len(sheet_data) + 2
  row = list(metadata.values())
  sheet.insert_row(row, entry_number)
  print('Updated Google Sheets')
else:
  print('Missing Information, did not update Google Sheets')

# End of the script
end = time.time()
print('Time taken:', round(end-start,2), 'seconds')
