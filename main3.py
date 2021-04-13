# Tutorial from https://www.tensorscience.com/ocr/optical-character-recognition-ocr-with-python-and-tesseract-4-an-introduction
import os
from PIL import Image
import pytesseract
import argparse
import cv2
import time
import numpy
import subprocess

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pprint import pprint

from utils.isDate import isDate
from utils.isDiopter import isDiopter, processDiopter
from utils.isBrand import isBrand, brandSimilarity
from utils.isModel import isModel, modelSimilarity, modelFindBrand
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
  output_GB_1, output_GB_2 = gaussianBlur(input1), gaussianBlur(input2)

  print('Undergoing Gaussian Blurring with Threshold Preprocessing')
  output_GBT_1, output_GBT_2 = gaussianBlur_threshold(input1), gaussianBlur_threshold(input2)

  return output_GB_1, output_GB_2, output_GBT_1, output_GBT_2

# Start of the script
start = time.time()
gray_GB_1, gray_GB_2, gray_GBT_1, gray_GBT_2 = preprocess(gray1, gray2)

# Save the processed images into the disk
filename1 = "./processed_images/{}_GB_1.png".format(os.getpid())
filename2 = "./processed_images/{}_GB_2.png".format(os.getpid())
filename3 = "./processed_images/{}_GBT_1.png".format(os.getpid())
filename4 = "./processed_images/{}_GBT_2.png".format(os.getpid())

# # Script to convert to 300DPI
# gray_GB_1 = cv2.cvtColor(gray_GB_1, cv2.COLOR_BGR2RGB)
# gray_GB_1 = Image.fromarray(gray_GB_1)

# gray_GB_2 = cv2.cvtColor(gray_GB_2, cv2.COLOR_BGR2RGB)
# gray_GB_2 = Image.fromarray(gray_GB_2)

# gray_GBT_1 = cv2.cvtColor(gray_GBT_1, cv2.COLOR_BGR2RGB)
# gray_GBT_1 = Image.fromarray(gray_GBT_1)

# gray_GBT_2 = cv2.cvtColor(gray_GBT_2, cv2.COLOR_BGR2RGB)
# gray_GBT_2 = Image.fromarray(gray_GBT_2)

# gray_GB_1.save(filename1, dpi=(300,300))
# gray_GB_2.save(filename2, dpi=(300,300))
# gray_GBT_1.save(filename3, dpi=(300,300))
# gray_GBT_2.save(filename4, dpi=(300,300))

cv2.imwrite(filename1, gray_GB_1)
cv2.imwrite(filename2, gray_GB_2)
cv2.imwrite(filename3, gray_GBT_1)
cv2.imwrite(filename4, gray_GBT_2)

# Load the image as a PIL/Pillow image, apply OCR
text_GB_1 = pytesseract.image_to_string(Image.open(filename1))
text_GB_2 = pytesseract.image_to_string(Image.open(filename2))
text_GBT_1 = pytesseract.image_to_string(Image.open(filename3))
text_GBT_2 = pytesseract.image_to_string(Image.open(filename4))

information_GB_1 = text_GB_1.split()
information_GB_2 = text_GB_2.split()
information_GBT_1 = text_GBT_1.split()
information_GBT_2 = text_GBT_2.split()
info_GB = information_GB_1 + information_GB_2
info_GBT = information_GBT_1 + information_GBT_2
print(info_GB)
print(info_GBT)

metadata_GB = {'brand': '', 'model': '', 'batch': '','expirydate': '', 'serialnumber': '', 'diopter': ''}
metadata_GBT = {'brand': '', 'model': '', 'batch': '','expirydate': '', 'serialnumber': '', 'diopter': ''}

def dataExtraction(metadata, info):
  # Round 1 of looping gather any information
  for i in info:
    if isDate(i):
      metadata['expirydate'] = i
    if isDiopter(i):
      metadata['diopter'] = processDiopter(i)
    if isBrand(i):
      metadata['brand'] = i.upper()

  # If brand is not detected, perform brand similarity score
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
  
  # If the brand is ALCON, convert it to ACRYSOF
  if metadata['brand'] == 'ALCON': metadata['brand'] = 'ACRYSOF'

  # If brand is still not detected after similarity score, search for model -> brand
  # May need to do similarity score
  if metadata['brand'] == '':
    for i in info:
      _brand, _model = modelFindBrand(i)
      if _brand != '' and _model != '':
        metadata['brand'] = _brand
        metadata['model'] = _model

  # Using identified brand, detect the model and the serial number
  if metadata['brand'] != '':
    for i in info:
      if isModel(i, metadata['brand']):
        metadata['model'] = i
      if isSerial(i, metadata['brand']):
        metadata['serialnumber'] = i

    # If model is not detected, perform model similarity score
    if metadata['model'] == '':
      similarity_model_score = {}
      for i in info:
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

  # Using identified model, detect the batch
  if metadata['model'] != '':
    for i in info:
      if isBatch(i, metadata['model']):
        metadata['batch'] = i
    if metadata['batch'] == '':
      similarity_batch_score = {}
      for i in info:
        if len(i) >= 9 and len(i) < 18:
          most_similar_batch, score = batchSimilarity(i, metadata['model'])
          if score > 36:
            if most_similar_batch in similarity_batch_score.keys():
              if similarity_batch_score[most_similar_batch] < score:
                similarity_batch_score[most_similar_batch] = score
            else:
              similarity_batch_score[most_similar_batch] = score
      print(f'Batch Similarity Score - {similarity_batch_score}')
      if (similarity_batch_score != {}):
        batch_name = (max(similarity_batch_score, key=similarity_batch_score.get))
        autocorrect_batch_name = metadata['model'] + batch_name[len(metadata['model']):]
        metadata['batch'] = autocorrect_batch_name
        # if (checkSerialSize(autocorrect_batch_name, metadata['brand']) == False):
        #   metadata['batch'] = autocorrect_batch_name + '0'
        # else:
        #   metadata['batch'] = autocorrect_batch_name

  # If serial number is still not detected, perform a secondary check if the serial number has been broken up into two
  if metadata['serialnumber'] == '':
    for i in range(len(info)-2):
      if isSerial_2(info[i], info[i+1], metadata['brand']):
        combined_serial = info[i] + info[i+1]
        metadata['serialnumber'] = combined_serial

  # print(f'Final Output after OCR - {metadata}')
  return metadata

metadata_GB = dataExtraction(metadata_GB, info_GB)
metadata_GBT = dataExtraction(metadata_GBT, info_GBT)

print('Gaussian Blur only:', metadata_GB)
print('Gaussian Blur + Threshold:', metadata_GBT)

def combineData(metadata_GB, metadata_GBT):
  metadata_final = {'brand': '', 'model': '', 'batch': '','expirydate': '', 'serialnumber': '', 'diopter': ''}
  for i in metadata_final.keys():
    if metadata_GB[i] == metadata_GBT[i]:
      metadata_final[i] = metadata_GB[i]
    elif metadata_GB[i] == '' and metadata_GBT[i] != '':
      metadata_final[i] = metadata_GBT[i]
    elif metadata_GB[i] != '' and metadata_GBT[i] == '':
      metadata_final[i] = metadata_GB[i]
    else:
      if i == 'batch':
        if isBatch(metadata_GBT[i], metadata_GBT['model']):
          metadata_final[i] = metadata_GBT[i]
        elif isBatch(metadata_GB[i], metadata_GB['model']):
          metadata_final[i] = metadata_GB[i]
      if i == 'diopter':
        if len(metadata_GB[i]) == 4 or len(metadata_GB[i]) == 5:
          metadata_final[i] = metadata_GB[i]
        elif len(metadata_GBT[i]) == 4 or len(metadata_GBT[i]) == 5:
          metadata_final[i] = metadata_GBT[i]
      elif i == 'serialnumber':
        metadata_final[i] = metadata_GB[i]
      elif i == 'expirydate':
        pass
  return metadata_final

# Combine 
metadata_final = combineData(metadata_GB, metadata_GBT)
print('Final Metadata Extracted:', metadata_final)

shouldUpdate = True
for i in metadata_final.values():
  if i == '':
    shouldUpdate = False
  
if shouldUpdate:
  entry_number = len(sheet_data) + 2
  row = list(metadata_final.values())
  sheet.insert_row(row, entry_number)
  print('Updated Google Sheets')
else:
  print('Missing Information, did not update Google Sheets')

# End of the script
end = time.time()
print('Time taken:', round(end-start,2), 'seconds')
