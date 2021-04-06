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
from utils.isDiopter import isDiopter, processDiopter
from utils.isBrand import isBrand, brandSimilarity
from utils.isModel import isModel, modelSimilarity
from utils.isSerial import isSerial, isSerial_2
from utils.isBatch import isBatch, batchSimilarity

from utils.preprocessing.blur import medianBlur, averageBlur, gaussianBlur, bilateralBlur
from utils.preprocessing.threshold import threshold

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

def preprocess(type_of_preprocess, input1, input2):
  if type_of_preprocess == "thresh":
    print('Undergoing Threshold Preprocessing')
    output1, output2 = threshold(input1), threshold(input2)
  elif type_of_preprocess == "blur":
    print('Undergoing Gaussian Blur Preprocessing')
    output1 = medianBlur(input1)
    output2 = medianBlur(input2)
  else:
    output1, output2 = gray1, gray2
  return output1, output2

gray1, gray2 = preprocess(args["preprocess"], gray1, gray2)

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
      if i.isalnum() and len(i) > 4 and len(i) < 11:
        most_similar_model, score = modelSimilarity(i, metadata['brand'])
        if score > 0:
          if most_similar_model in similarity_model_score.keys():
            if similarity_model_score[most_similar_model] < score:
              similarity_model_score[most_similar_model] = score
          else:
            similarity_model_score[most_similar_model] = score
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
      if len(i) >= 9 and i.isalnum():
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
      metadata['batch'] = metadata['model'] + batch_name[len(metadata['model']):]


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
