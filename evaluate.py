import os
from PIL import Image
import pytesseract
import argparse
import cv2

from pprint import pprint

from utils import csv_utils

# Set Arguments Parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default="/home/hwlee96/SUTD/01.116/project/Data", help="path to dataset images")
ap.add_argument("-p", "--preprocess", type=str, default="blur", help="preprocessing method that is applied to the image")
args = vars(ap.parse_args())

DATASET_PATH = args["dataset"]

def preprocess(image):
    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # preprocess the image
    if args["preprocess"] == "thresh": 
        image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # blur the image to remove noise
    elif args["preprocess"] == "blur": 
        image = cv2.medianBlur(image, 3)

    return image


def evaluate_acc(image_path, prediction, ground_truth): #TODO: Implement "augmentations" (e.g. lower/upper case, date delimeters)
    accurate_count = 0
    for item in prediction.split():
        if item in ground_truth.values():
            accurate_count += 1
    
    if accurate_count == len(ground_truth):
        print("===============================================================================")
        print("OCR SUCCESS: " + image_path)
        print("-------------------------------------------------------------------------------")
        print("===============================================================================")        
        return True
    else:

        print("===============================================================================")
        print("OCR FAILED: " + image_path)
        print("-------------------------------------------------------------------------------")
        print("Ground Truth: {}".format(ground_truth))
        print("Prediction: {}".format(prediction.split())) 
        print("===============================================================================")
        print("") 
        return False

def main():
    failed_test_images = []
    for brand_dirname in os.listdir(DATASET_PATH):
        brand_dirpath = os.path.join(DATASET_PATH, brand_dirname)
        if os.path.isdir(brand_dirpath) and brand_dirname != "Others":
            for model_dirname in os.listdir(brand_dirpath):
                model_dirpath = os.path.join(brand_dirpath, model_dirname)
                for view in os.listdir(model_dirpath):
                    if view in csv_utils.VIEWS.keys(): # To exclude "front" directory
                        view_path = os.path.join(model_dirpath, view)
                        for image_name in os.listdir(view_path):
                            image_path = os.path.join(view_path, image_name)
                            image = cv2.imread(image_path)
                            image = preprocess(image)
                            prediction = pytesseract.image_to_string(Image.fromarray(image))
                            ground_truth = csv_utils.extract_ground_truth(brand_dirname, model_dirname, view)
                            if not evaluate_acc(image_path, prediction, ground_truth):
                                failed_test_images.append(image_path)
                                
    return failed_test_images

main()