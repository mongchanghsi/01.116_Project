import os
from PIL import Image
import pytesseract
import argparse
import cv2
import copy 
import json
from pprint import pprint

from utils import json_utils
from utils import csv_utils
from utils.isDate import isDate
from utils.isDiopter import isDiopter, processDiopter
from utils.isBrand import isBrand, brandSimilarity
from utils.isModel import isModel, modelSimilarity
from utils.isSerial import isSerial, isSerial_2
from utils.isBatch import isBatch, batchSimilarity

from utils.preprocessing.blur import medianBlur, averageBlur, gaussianBlur, bilateralBlur
from utils.preprocessing.threshold import threshold, adaptiveMeanThreshold, adaptiveGaussianThreshold, gaussianBlur_threshold

import evaluate_utils

# Set Arguments Parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset-dir", required=True, help="dir of dataset images")
ap.add_argument("-p", "--preprocess", type=str, choices=["thresh", "blur", "all", "none"], help="preprocessing method that is applied to the image")
ap.add_argument("-v", "--verbose", choices=[1,0], type=int, default=0)

args = vars(ap.parse_args())

DATASET_PATH =  args["dataset_dir"]
SIG_FIG = 3

EVAL_LOG_FILEPATH = "logs/{}_{}.json".format(args["dataset_dir"], args["preprocess"])
EVAL_CSV_FILEPATH = "logs/{}_{}.csv".format(args["dataset_dir"], args["preprocess"])

csv_field_names = ["Image Path", "Stage", "Number of Words", "Number of Accurate Words", "Word Accuracy",
                    "Number of Characters", "Number of Accurate Characters", "Character Accuracy"]

if not os.path.exists(EVAL_CSV_FILEPATH):
    csv_utils.initialize(EVAL_CSV_FILEPATH, csv_field_names)

if not os.path.exists(EVAL_LOG_FILEPATH):
    f = open(EVAL_LOG_FILEPATH, "w")
    f.write("[]")
    f.close()

IMAGE_PRED_PARAMS = ["Image Path", "Orientation"]

STAGE_PRED_PARAMS = ["Number of Words", "Number of Accurate Words", "Word Accuracy",
                    "Number of Characters", "Number of Accurate Characters", "Character Accuracy"]

NO_OF_STAGES = 3

def preprocess(image, preprocess_type):
    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # preprocess the image
    if preprocess_type == "thresh": 
        image = cv2.GaussianBlur(image, (3,3),0)
        image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # blur the image to remove noise
    elif preprocess_type == "blur": 
        image = cv2.medianBlur(image, 3)
    

    return image

# Preprocessing function - produces 2 versions of using Gaussian Blur only and Gaussian Blur and Binary Threshold
def preprocess_2(image):
  print('Undergoing Gaussian Blur Preprocessing')
  image = gaussianBlur(image)

  print('Undergoing Gaussian Blurring with Threshold Preprocessing')
  image = gaussianBlur_threshold(image)

  return output_GB_1, output_GB_2, output_GBT_1, output_GBT_2


#TODO: Implement "augmentations" (e.g. lower/upper case, date delimeters)
def calc_accuracy(image_path, view, preprocess_type, prediction, ground_truth):
    MIN_CHAR_SIMILARITY = 0.4

    pred_info = {param: None for param in IMAGE_PRED_PARAMS}
    pred_info["Image Path"] = image_path
    pred_info["Orientation"] = view
    pred_info["Preprocess Type"] = preprocess_type
    pred_info["Stages"] = {1: None, 2: None, 3: None}

    stage_info = {param: None for param in STAGE_PRED_PARAMS}
    stage_info["Number of Words"] = len([x for x in ground_truth.values() if x !=None])
    stage_info["Number of Characters"] = len(''.join([x for x in ground_truth.values() if x !=None]))
    gt_predicted = copy.deepcopy(ground_truth)
    for param in gt_predicted:
        gt_predicted[param] = {"GT": gt_predicted[param], "Predicted": None, "Character Similarity": None}
    stage_info["GT - Predicted"] = gt_predicted

    stage_info["GT - Predicted"]["Brand"]["GT"] = stage_info["GT - Predicted"]["Brand"]["GT"].upper()

    autocorrect_words = {}

    for stage_no in pred_info["Stages"]:
        single_item_acc_word_count = 0
        single_item_sim_char_count = 0

        if stage_no == 1:
            print("Stage 1: Raw OCR Output")
            for item in prediction.split():
                exact_flag = False

                # If any predicted item matches the GT exactly
                for param in ground_truth:
                    if evaluate_utils.is_pred_param_not_filled(param, stage_info) and \
                            evaluate_utils.is_exact(item, param, stage_info):
                        stage_info["GT - Predicted"][param]["Predicted"] = item
                        exact_flag = True
                        break

                # isFunctions that have indicate 100% character accuracy TODO: refactored will affect the count
                if not exact_flag:
                    if evaluate_utils.is_pred_param_not_filled('Brand', stage_info) and isBrand(item):
                        item = item.upper()
                        stage_info["GT - Predicted"]['Brand']["Predicted"] = item

                        if evaluate_utils.is_pred_param_not_filled('Model', stage_info) and \
                                isModel(item, stage_info["GT - Predicted"]['Brand']['GT']):
                            stage_info["GT - Predicted"]['Model']["Predicted"] = item
                    else: 
                        continue

                    # TODO: Verify if the outputs of is<> functions have the correct length
                    # TODO: delete element from output list once it is used as valid prediction

        # TODO: implement serial_2 (need to loop pair values)
        if stage_no == 2: 
            print("Stage 2: Extracted Similar Words")

            # is__ functions that do not necessarily mean 100% character accuracy
            for item in prediction.split():

                if evaluate_utils.is_pred_param_not_filled('Expiry Date', stage_info) and isDate(item):
                    stage_info["GT - Predicted"]['Expiry Date']["Predicted"] = item

                elif evaluate_utils.is_pred_param_not_filled("Diopter", stage_info) and isDiopter(item):
                    stage_info["GT - Predicted"]['Diopter']["Predicted"] = item

                elif evaluate_utils.is_param_pred_gt_exact("Brand", stage_info) and \
                        evaluate_utils.is_param_pred_gt_exact("Serial Number", stage_info) and \
                        isSerial(item, stage_info["GT - Predicted"]['Brand']['GT']):                    

                    stage_info["GT - Predicted"]['Serial Number']["Predicted"] = item

                elif evaluate_utils.is_param_pred_gt_exact("Model", stage_info) and \
                        evaluate_utils.is_param_pred_gt_exact("Batch Number", stage_info) and \
                        isBatch(item, stage_info["GT - Predicted"]['Model']['GT']):

                    stage_info["GT - Predicted"]['Batch Number']["Predicted"] = item
                    
                else: 
                    pass             

            for param in stage_info["GT - Predicted"]:
                if evaluate_utils.is_pred_param_not_filled(param, stage_info):
                    param_similarity_score = evaluate_utils.similarity_score(ground_truth, prediction.split(), param)
                    
                    if param_similarity_score != {} and \
                            max([similar_pred['Score'] for similar_pred in param_similarity_score.values()]) >= MIN_CHAR_SIMILARITY*100:
                       
                        next_similar_pred = max(param_similarity_score, 
                                                key=lambda similar_pred:param_similarity_score[similar_pred]["Score"])

                        stage_info["GT - Predicted"][param]["Predicted"] = param_similarity_score[next_similar_pred]["Original Value"]

                        autocorrect_words[param] = {"Next Similar Pred": next_similar_pred}
                                    
        if stage_no == 3:
            print("Stage 3: Autocorrect Functions")

            # Autocorrect diopter
            if evaluate_utils.is_param_pred_gt_exact("Diopter", stage_info):
                stage_info["GT - Predicted"]["Diopter"]["Predicted"] = processDiopter(stage_info["GT - Predicted"]["Diopter"]["Predicted"])
                    
            for param in autocorrect_words:
                stage_info["GT - Predicted"][param]["Predicted"] = autocorrect_words[param]["Next Similar Pred"]

        stage_info, pred_info, single_item_acc_word_count, single_item_sim_char_count = evaluate_utils.stage_eval(stage_no, stage_info, 
                                                                                        pred_info, single_item_acc_word_count, single_item_sim_char_count)
        content = csv_utils.stage_result(image_path, stage_no, stage_info)
        csv_utils.write_to_csv(EVAL_CSV_FILEPATH, content)

        print("----------------------")
    
    json_utils.append_new_entry(EVAL_LOG_FILEPATH, pred_info)

    return pred_info

def is_empty_gt(extracted_gt):
    for param in extracted_gt:
        if extracted_gt[param] != None:
            return False
    return True

def main():
    preprocess_ds_info = {}
    ds_info = { 1: {"Total Number of Words": 0, 
                    "Total Number of Accurate Words": 0,
                    "Total Number of Characters": 0, 
                    "Total Number of Accurate Characters": 0},
                2: {"Total Number of Words": 0, 
                    "Total Number of Accurate Words": 0,
                    "Total Number of Characters": 0, 
                    "Total Number of Accurate Characters": 0},
                3: {"Total Number of Words": 0, 
                    "Total Number of Accurate Words": 0,
                    "Total Number of Characters": 0, 
                    "Total Number of Accurate Characters": 0}
                }
    empty_gts = []

    preprocess_types = [args["preprocess"]] if args["preprocess"] != "all" else ["thresh", "blur"]
    print("Starting evaluation...")

    for preprocess_type in preprocess_types:
        preprocess_ds_info[preprocess_type] = ds_info

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

                            preprocess_pred = []

                            for preprocess_type in preprocess_types:
                                preprocessed_image = preprocess(image, preprocess_type)
                                prediction = pytesseract.image_to_string(Image.fromarray(preprocessed_image))
                                
                                ground_truth = csv_utils.extract_ground_truth(CSV_FILE_PATH, brand_dirname, model_dirname, view)
                                
                                if is_empty_gt(ground_truth):
                                    empty_gts.append(ground_truth)
                                    continue

                                print("=============== Evaluating: {} ===============".format(image_path))
                                pred_info = calc_accuracy(image_path, view, preprocess_type, prediction, ground_truth)
                                preprocess_pred.append(evaluate_utils.extract_stage_3_info(pred_info))                           
                                
                                # if len(preprocess_pred) == 2:
                                #     print("************ Stage 4: Combine side and back images ************")
                                #     combined = evaluate_utils.combineData(preprocess_pred[0], preprocess_pred[1], ground_truth)
                                #     print("**********************************")

                                for stage_no in pred_info["Stages"]:
                                    dp_total_acc_word_count = pred_info["Stages"][stage_no]["Number of Accurate Words"]
                                    dp_total_word_count = pred_info["Stages"][stage_no]["Number of Words"]
                                    dp_total_acc_char_count = pred_info["Stages"][stage_no]["Number of Accurate Characters"]
                                    dp_total_char_count = pred_info["Stages"][stage_no]["Number of Characters"]

                                    print("STAGE {} - WORD-LEVEL - Total number of accurate words predicted / Total number of words = Word accuracy - {} / {} = {}".format(
                                        stage_no, dp_total_acc_word_count, dp_total_word_count, round(dp_total_acc_word_count/dp_total_word_count, SIG_FIG)))
                                    print("STAGE {} - CHAR-LEVEL - Total number of accurate chars predicted / Total number of chars = Char accuracy - {} / {} = {}".format(
                                        stage_no, dp_total_acc_char_count, dp_total_char_count, round(dp_total_acc_char_count/dp_total_char_count, SIG_FIG)))
                                    
                                    preprocess_ds_info[preprocess_type][stage_no]["Total Number of Accurate Words"]  += dp_total_acc_word_count
                                    preprocess_ds_info[preprocess_type][stage_no]["Total Number of Words"]  += dp_total_word_count
                                    preprocess_ds_info[preprocess_type][stage_no]["Total Number of Accurate Characters"]  += dp_total_acc_char_count
                                    preprocess_ds_info[preprocess_type][stage_no]["Total Number of Characters"]  += dp_total_char_count

                                print(len("=============== Evaluating: {} ===============".format(image_path))*"=")
                                print("")
    
    for stage_no in ds_info:
        overall_word_acc = ds_info[stage_no]["Total Number of Accurate Words"] / ds_info[stage_no]["Total Number of Words"]
        overall_char_acc = ds_info[stage_no]["Total Number of Accurate Characters"] /  ds_info[stage_no]["Total Number of Characters"]
        print("Stage {}: Total Word Accuracy - {} - Total Character Accuracy - {}".format(stage_no, overall_word_acc, overall_char_acc))

    print("Check CSV entries against empty ground truths extracted:")
    print(empty_gt for empty_gt in empty_gts)


def test_one():
    brand = "Tecnis"
    view = "back"
    model = "ZXR00"
    image_path = "/mnt/c/Users/user/OneDrive - Singapore University of Technology and Design/Term 7/01.116/1D Project/01.116_IHIS_Project/Data/good/Tecnis/ZXR00/back/3.jpeg"
    preprocess_types = [args["preprocess"]] if args["preprocess"] != "all" else ["thresh", "blur"]
    combined_predictions = ''
    for preprocess_type in preprocess_types:
        image = cv2.imread(image_path)
        preprocessed_image = preprocess(image, preprocess_type)
        prediction = pytesseract.image_to_string(Image.fromarray(preprocessed_image))
        combined_predictions += prediction

    ground_truth = csv_utils.extract_ground_truth(CSV_FILE_PATH, brand, model, view)
    
    pred_info = calc_accuracy(
                    image_path, 
                    view, 
                   combined_predictions,
                    ground_truth
                    )

    return pred_info

def test_all():

    image_path = "/mnt/c/Users/user/OneDrive - Singapore University of Technology and Design/Term 7/01.116/1D Project/01.116_IHIS_Project/Data/good/Tecnis/ZCT300/back/IMG_1648.png"
    preprocess_types = [args["preprocess"]] if args["preprocess"] != "all" else ["thresh", "blur"]
    combined_predictions = ''
    for preprocess_type in preprocess_types:
        image = cv2.imread(image_path)

        preprocessed_image = preprocess(image, preprocess_type)
        prediction = pytesseract.image_to_string(Image.fromarray(preprocessed_image))
        combined_predictions += prediction

    pred_info = calc_accuracy(
                    image_path, 
                    "back", 
                    combined_predictions,
                    csv_utils.extract_ground_truth(CSV_FILE_PATH, "Tecnis", "ZCT300", "back")
                    )
main()

# test_one()