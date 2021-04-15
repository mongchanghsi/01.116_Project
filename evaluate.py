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
from utils.preprocessing.threshold import threshold

# Set Arguments Parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset-dir", required=True, help="dir of dataset images")
ap.add_argument("-p", "--preprocess", type=str, choices=["thresh", "blur", "all", "none"], help="preprocessing method that is applied to the image")
ap.add_argument("-v", "--verbose", choices=[1,0], type=int, default=0)

args = vars(ap.parse_args())

DATASET_PATH = "/mnt/c/Users/user/OneDrive - Singapore University of Technology and Design/Term 7/01.116/1D Project/01.116_IHIS_Project/Data/" + args["dataset_dir"]
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
        # image = cv2.medianBlur(image, 3)
        image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # blur the image to remove noise
    elif preprocess_type == "blur": 
        image = cv2.medianBlur(image, 3)
    
    return image

 #TODO: Implement "augmentations" (e.g. lower/upper case, date delimeters)
def calc_accuracy(image_path, view, prediction, ground_truth):
    MIN_CHAR_SIMILARITY = 0.4

    pred_info = {param: None for param in IMAGE_PRED_PARAMS}
    pred_info["Image Path"] = image_path
    pred_info["Orientation"] = view
    pred_info["Stages"] = {1: None, 2: None, 3: None}

    stage_info = {param: None for param in STAGE_PRED_PARAMS}
    stage_info["Number of Words"] = len([x for x in ground_truth.values() if x !=None])
    stage_info["Number of Characters"] = len(''.join([x for x in ground_truth.values() if x !=None]))
    gt_predicted = copy.deepcopy(ground_truth)
    for param in gt_predicted:
        gt_predicted[param] = {"GT": gt_predicted[param], "Predicted": None, "Character Similarity": None}
    stage_info["GT - Predicted"] = gt_predicted

    stage_info["GT - Predicted"]["Brand"]["GT"] = stage_info["GT - Predicted"]["Brand"]["GT"].upper()

    single_item_acc_word_count = 0
    single_item_sim_char_count = 0

    autocorrect_words = {}

    for stage_no in pred_info["Stages"]:
        # print("Start of stage")
        # pprint(stage_info)
        # print()
        if stage_no == 1:
            print("Stage 1: Raw OCR Output")
            for item in prediction.split():
                exact_flag = False
                for param in ground_truth:
                    if stage_info["GT - Predicted"][param]["GT"] != None and \
                            stage_info["GT - Predicted"][param]["Predicted"] == None and \
                            item == stage_info["GT - Predicted"][param]["GT"]:
                        stage_info["GT - Predicted"][param]["Predicted"] = item
                        stage_info["GT - Predicted"][param]["Character Similarity"] = 100.0
                        single_item_acc_word_count += 1
                        single_item_sim_char_count += len(ground_truth[param])
                        exact_flag = True
                        break

                # isFunctions that have indicate 100% character accuracy
                if not exact_flag:
                    if stage_info["GT - Predicted"]['Expiry Date']["GT"] != None and \
                            stage_info["GT - Predicted"]['Expiry Date']["Predicted"] == None and \
                            isDate(item):
                        
                        stage_info["GT - Predicted"]['Expiry Date']["Predicted"] = item
                        stage_info["GT - Predicted"]['Expiry Date']["Character Similarity"] = 100.0
                        gt_temp = stage_info["GT - Predicted"]['Expiry Date']['GT']
                        # print("Expiry Date")
                        # print("Original Prediction: " + item)
                        # print("Ground Truth: " + stage_info["GT - Predicted"]['Expiry Date']['GT'])
                        # print("Revised Prediction: " + item)

                    elif stage_info["GT - Predicted"]['Brand']["GT"] != None and \
                            stage_info["GT - Predicted"]['Brand']["Predicted"] == None and \
                            isBrand(item):
                        
                        item = item.upper()
                        stage_info["GT - Predicted"]['Brand']["Predicted"] = item
                        stage_info["GT - Predicted"]['Brand']["Character Similarity"] = 100.0
                        gt_temp = stage_info["GT - Predicted"]['Brand']['GT']
                        # print("Brand")
                        # print("Original Prediction: " + item)
                        # print("Ground Truth: " + stage_info["GT - Predicted"]['Brand']['GT'])
                        # print("Revised Prediction: " + item)
                        
                        if stage_info["GT - Predicted"]['Model']["GT"] != None and \
                            stage_info["GT - Predicted"]['Model']["Predicted"] == None and \
                            isModel(item, stage_info["GT - Predicted"]['Brand']['GT']):
                        
                            stage_info["GT - Predicted"]['Model']["Predicted"] = item
                            stage_info["GT - Predicted"]['Model']["Character Similarity"] = 100.0
                            gt_temp = stage_info["GT - Predicted"]['Model']['GT']
                            # print("Model")
                            # print("Original Prediction: " + item)
                            # print("Ground Truth: " + stage_info["GT - Predicted"]['Model']['GT'])
                            # print("Revised Prediction: " + item)

                    else: 
                        continue
                    
                    # TODO: Verify if the outputs of is<> functions have the correct length
                    # TODO: delete element from output list once it is used as valid prediction
                    single_item_acc_word_count += 1
                    # print(3123214213)
                    # print(single_item_sim_char_count, len(gt_temp))             
                    single_item_sim_char_count += len(gt_temp)
                    # print()

        if stage_no == 2: # TODO: implement serial_2 (need to loop pair values)
            print("Stage 2: Extracted Similar Words")
            for item in prediction.split():
                # is__ functions that do not necessarily mean 100% character accuracy
                if stage_info["GT - Predicted"]['Diopter']["GT"] != None and \
                        stage_info["GT - Predicted"]['Diopter']["Predicted"] == None and \
                        isDiopter(item):

                    # print("Diopter")
                    # print("Original Prediction: " + item)
                    # print("Ground Truth: " + stage_info["GT - Predicted"]['Diopter']['GT'])
                    # print("Revised Prediction: " + item)
                    # print()

                    stage_info["GT - Predicted"]['Diopter']["Predicted"] = item
                    score, similar_chars_count = char_similarity(item, stage_info["GT - Predicted"]['Diopter']["GT"])
                    stage_info["GT - Predicted"]['Diopter']["Character Similarity"] = score
                    single_item_sim_char_count += similar_chars_count

                elif stage_info["GT - Predicted"]['Brand']['GT'] != None and \
                        stage_info["GT - Predicted"]['Serial Number']["GT"] != None and \
                        stage_info["GT - Predicted"]['Serial Number']["Predicted"] == None and \
                        isSerial(item, stage_info["GT - Predicted"]['Brand']['GT']):                    

                    # print("Serial Number")
                    # print("Original Prediction: " + item)
                    # print("Ground Truth: " + stage_info["GT - Predicted"]['Serial Number']['GT'])
                    # print("Revised Prediction: " + item)
                    # print()

                    stage_info["GT - Predicted"]['Serial Number']["Predicted"] = item
                    score, similar_chars_count = char_similarity(item, stage_info["GT - Predicted"]['Serial Number']["GT"])
                    stage_info["GT - Predicted"]['Serial Number']["Character Similarity"] = score
                    single_item_sim_char_count += similar_chars_count

                elif stage_info["GT - Predicted"]['Model']['GT'] != None and \
                        stage_info["GT - Predicted"]['Batch Number']["GT"] != None and \
                        stage_info["GT - Predicted"]['Batch Number']["Predicted"] == None and \
                        isBatch(item, stage_info["GT - Predicted"]['Model']['GT']):
            
                    print("Batch Number")
                    print("Original Prediction: " + item)
                    print("Ground Truth: " + stage_info["GT - Predicted"]['Batch Number']['GT'])
                    print("Revised Prediction: " + item)
                    print()

                    stage_info["GT - Predicted"]['Batch Number']["Predicted"] = item
                    score, similar_chars_count = char_similarity(item, stage_info["GT - Predicted"]['Batch Number']["GT"])
                    stage_info["GT - Predicted"]['Batch Number']["Character Similarity"] = score
                    single_item_sim_char_count += similar_chars_count

                    
                else: 
                    pass

                # TODO: Verify if the outputs of is<> functions have the correct length
                # TODO: delete element from output list once it is used as valid prediction
                

            for param in stage_info["GT - Predicted"]:
                if stage_info["GT - Predicted"][param]["GT"] != None and \
                        stage_info["GT - Predicted"][param]["Predicted"] == None:
                    param_similarity_score = similarity_score(ground_truth, prediction.split(), param)

                    if param_similarity_score != {} and max([similar_pred['Score'] for similar_pred in param_similarity_score.values()]) >= MIN_CHAR_SIMILARITY*100:
                        
                        next_similar_pred = max(param_similarity_score, 
                                                key=lambda similar_pred:param_similarity_score[similar_pred]["Score"])

                        

                        stage_info["GT - Predicted"][param]["Predicted"] = param_similarity_score[next_similar_pred]["Original Value"]
                        score, similar_chars_count = char_similarity(
                            stage_info["GT - Predicted"][param]["Predicted"], 
                            stage_info["GT - Predicted"][param]["GT"]
                            )

                        autocorrect_words[param] = {"Next Similar Pred": next_similar_pred, 
                                                    "No of chars to subtract after autocorrect": similar_chars_count}
                        
                        # print(score, similar_chars_count)
                        stage_info["GT - Predicted"][param]["Character Similarity"] = score
                        single_item_sim_char_count += similar_chars_count
            
            # char_accuracy = round(single_item_sim_char_count/stage_info["Number of Characters"], SIG_FIG)
            # stage_info["Character Accuracy"] = char_accuracy
            
        if stage_no == 3:
            print("Stage 3: Autocorrect Functions")

            if stage_info["GT - Predicted"]["Diopter"]["GT"] != None and \
                stage_info["GT - Predicted"]["Diopter"]["Predicted"] != None and \
                stage_info["GT - Predicted"]["Diopter"]["Character Similarity"] != 100.0:

                    stage_info["GT - Predicted"]["Diopter"]["Predicted"] = processDiopter(stage_info["GT - Predicted"]["Diopter"]["Predicted"])

                    # if processDiopter(stage_info["GT - Predicted"]["Diopter"]["Predicted"]) == stage_info["GT - Predicted"]["Diopter"]["GT"]:
                        # single_item_sim_char_count -= len(stage_info["GT - Predicted"]["Diopter"]["Predicted"])
                        # stage_info["GT - Predicted"]["Diopter"]["Predicted"] = processDiopter(stage_info["GT - Predicted"]["Diopter"]["Predicted"])
                        # stage_info["GT - Predicted"]["Diopter"]["Character Similarity"] = 100.0
                        # single_item_acc_word_count += 1
                        # single_item_sim_char_count += len(stage_info["GT - Predicted"]["Diopter"]["GT"])

            for param in autocorrect_words:
                stage_info["GT - Predicted"][param]["Predicted"] = autocorrect_words[param]["Next Similar Pred"]
                if stage_info["GT - Predicted"][param]["Predicted"] == stage_info["GT - Predicted"][param]["GT"]:
                    stage_info["GT - Predicted"][param]["Character Similarity"] = 100.0
                    single_item_acc_word_count += 1
                    single_item_sim_char_count -= autocorrect_words[param]["No of chars to subtract after autocorrect"]
                    single_item_sim_char_count += len(stage_info["GT - Predicted"][param]["GT"])
                else:
                    sim_score, sim_chars_count = char_similarity(stage_info["GT - Predicted"][param]["Predicted"], stage_info["GT - Predicted"][param]["GT"])
                    stage_info["GT - Predicted"][param]["Character Similarity"] = sim_score
                    single_item_sim_char_count -= autocorrect_words[param]["No of chars to subtract after autocorrect"]
                    single_item_sim_char_count += sim_chars_count

        
        stage_info["Number of Accurate Words"] = single_item_acc_word_count
        stage_info["Number of Accurate Characters"] = single_item_sim_char_count
        word_accuracy = round(stage_info["Number of Accurate Words"]/stage_info["Number of Words"], SIG_FIG)
        char_accuracy = round(stage_info["Number of Accurate Characters"]/stage_info["Number of Characters"], SIG_FIG)
        stage_info["Word Accuracy"] = word_accuracy
        stage_info["Character Accuracy"] = char_accuracy

        pred_info["Stages"][stage_no] = copy.deepcopy(stage_info)

        content = {
            "Image Path": image_path, 
            "Stage": str(stage_no),
            "Number of Words": stage_info["Number of Words"], 
            "Number of Accurate Words": stage_info["Number of Accurate Words"], 
            "Word Accuracy": stage_info["Word Accuracy"],
            "Number of Characters": stage_info["Number of Characters"], 
            "Number of Accurate Characters": stage_info["Number of Accurate Characters"], 
            "Character Accuracy": stage_info["Character Accuracy"]
        }
        csv_utils.write_to_csv(EVAL_CSV_FILEPATH, content)

        print("----------------------")
    # TODO: Verify if the counts are correct
    
    json_utils.append_new_entry(EVAL_LOG_FILEPATH, pred_info)

    return pred_info

def similarity_score(gt_items: dict, pred_items: list, param: str):
    similarity_score = {}
    most_similar_param, score = None, None
    # print(gt_items, pred_items, param)
    for pred_item in pred_items:

        if param == "Brand" and len(pred_item) > 5:
            most_similar_param, score = brandSimilarity(pred_item)
            # print("brand", pred_item, most_similar_param, score)
        
        elif param == "Model" and pred_item.isalnum() and len(pred_item) > 4 and len(pred_item) < 11 and gt_items['Brand'] != None:
            most_similar_param, score = modelSimilarity(pred_item, gt_items['Brand'])
            # print(gt_items['Brand'])
            # print("model", pred_item, 1, most_similar_param, 2,score)
        
        elif param == "Batch Number" and len(pred_item) >= 9 and pred_item.isalnum() and gt_items['Model'] != None:
            most_similar_param, score = batchSimilarity(pred_item, gt_items['Model'])

        if score != None and score > 0:
            if most_similar_param in similarity_score.keys():
                if similarity_score[most_similar_param]["Score"] < score:
                    similarity_score[most_similar_param]["Score"] = score
                    similarity_score[most_similar_param]["Original Value"] = pred_item
            else:
                similarity_score[most_similar_param]= {}
                similarity_score[most_similar_param]["Score"] = score
                similarity_score[most_similar_param]["Original Value"] = pred_item
    return similarity_score

def char_similarity(pred, gt):
  similar_chars_count = 0
  i, j = 0, 0
  print(pred, gt)
  while i < len(pred) and j < len(gt):
    if pred[i] == gt[i]:
      similar_chars_count += 1
    i += 1
    j += 1
  score = (similar_chars_count / len(gt)) * 100
  return score, similar_chars_count

def is_empty_gt(extracted_gt):
    for param in extracted_gt:
        if extracted_gt[param] != None:
            return False
    return True

def main():
    ds_info = {1: {"Total Number of Words": 0, "Total Number of Accurate Words": 0,
                "Total Number of Characters": 0, "Total Number of Accurate Characters": 0},
            2: {"Total Number of Words": 0, "Total Number of Accurate Words": 0,
                "Total Number of Characters": 0, "Total Number of Accurate Characters": 0},
            3: {"Total Number of Words": 0, "Total Number of Accurate Words": 0,
                "Total Number of Characters": 0, "Total Number of Accurate Characters": 0}}
    empty_gts = []

    preprocess_types = [args["preprocess"]] if args["preprocess"] != "all" else ["thresh", "blur"]
    print("Starting evaluation...")

    for brand_dirname in os.listdir(DATASET_PATH):
        brand_dirpath = os.path.join(DATASET_PATH, brand_dirname)
        if os.path.isdir(brand_dirpath) and brand_dirname != "Others":
            for model_dirname in os.listdir(brand_dirpath):
                model_dirpath = os.path.join(brand_dirpath, model_dirname)
                for view in os.listdir(model_dirpath):
                    if view in csv_utils.VIEWS.keys(): # To exclude "front" directory
                        view_path = os.path.join(model_dirpath, view)
                        for image_name in os.listdir(view_path):
                            combined_predictions = ''
                            image_path = os.path.join(view_path, image_name)
                            image = cv2.imread(image_path)

                            for preprocess_type in preprocess_types:
                                preprocessed_image = preprocess(image, preprocess_type)
                                prediction = pytesseract.image_to_string(Image.fromarray(preprocessed_image))
                                combined_predictions += prediction

                            ground_truth = csv_utils.extract_ground_truth(brand_dirname, model_dirname, view)
                            
                            if is_empty_gt(ground_truth):
                                empty_gts.append(ground_truth)
                                continue

                            print("=============== Evaluating: {} ===============".format(image_path))
                            pred_info = calc_accuracy(image_path, view, combined_predictions, ground_truth)

                            for stage_no in pred_info["Stages"]:

                                dp_total_acc_word_count = pred_info["Stages"][stage_no]["Number of Accurate Words"]
                                dp_total_word_count = pred_info["Stages"][stage_no]["Number of Words"]
                                dp_total_acc_char_count = pred_info["Stages"][stage_no]["Number of Accurate Characters"]
                                dp_total_char_count = pred_info["Stages"][stage_no]["Number of Characters"]

                                print("STAGE {} - WORD-LEVEL - Total number of accurate words predicted / Total number of words = Word accuracy - {} / {} = {}".format(
                                    stage_no, dp_total_acc_word_count, dp_total_word_count, round(dp_total_acc_word_count/dp_total_word_count, SIG_FIG)))
                                print("STAGE {} - CHAR-LEVEL - Total number of accurate chars predicted / Total number of chars = Char accuracy - {} / {} = {}".format(
                                    stage_no, dp_total_acc_char_count, dp_total_char_count, round(dp_total_acc_char_count/dp_total_char_count, SIG_FIG)))
                                
                                ds_info[stage_no]["Total Number of Accurate Words"]  += dp_total_acc_word_count
                                ds_info[stage_no]["Total Number of Words"]  += dp_total_word_count
                                ds_info[stage_no]["Total Number of Accurate Characters"]  += dp_total_acc_char_count
                                ds_info[stage_no]["Total Number of Characters"]  += dp_total_char_count

                            print(len("=============== Evaluating: {} ===============".format(image_path))*"=")
                            print("")
    
    for stage_no in ds_info:
        overall_word_acc = ds_info[stage_no]["Total Number of Accurate Words"] / ds_info[stage_no]["Total Number of Words"]
        overall_char_acc = ds_info[stage_no]["Total Number of Accurate Characters"] /  ds_info[stage_no]["Total Number of Characters"]
        print("Stage {}: Total Word Accuracy - {} - Total Character Accuracy - {}".format(stage_no, overall_word_acc, overall_char_acc))

    print("Check CSV entries against empty ground truths extracted:")
    print(empty_gt for empty_gt in empty_gts)


def test_one():
    image_path = "/mnt/c/Users/user/OneDrive - Singapore University of Technology and Design/Term 7/01.116/1D Project/01.116_IHIS_Project/Data/good/Tecnis/ZXR00/back/3.jpeg"

    pred_info = calc_accuracy(
                    image_path, 
                    "back", 
                    pytesseract.image_to_string(Image.fromarray(preprocess(cv2.imread(image_path)))),
                    csv_utils.extract_ground_truth("Tecnis", "ZXR00", "back")
                    )

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
                    csv_utils.extract_ground_truth("Tecnis", "ZCT300", "back")
                    )
# main()

test_all()