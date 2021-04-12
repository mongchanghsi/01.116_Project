import os
from PIL import Image
import pytesseract
import argparse
import cv2
import copy 

from pprint import pprint

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
ap.add_argument("-d", "--dataset", default="/home/hwlee96/SUTD/01.116/project/Data", help="path to dataset images")
ap.add_argument("-p", "--preprocess", type=str, default="blur", help="preprocessing method that is applied to the image")
ap.add_argument("-v", "--verbose", choices=[1,0], type=int, default=0)

args = vars(ap.parse_args())

DATASET_PATH = args["dataset"]
SIG_FIG = 3

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

 #TODO: Implement "augmentations" (e.g. lower/upper case, date delimeters)
def calc_accuracy(image_path, view, prediction, ground_truth):
    ACC_TYPES = ["word_level", "char_level"]
    MIN_CHAR_SIMILARITY = 0.4
    pred_info = {
                "Stage": None,
                "Image Path": None,
                "Orientation": None,
                "Number of Words": None,
                "Number of Characters": None,
                "Word Accuracy": None,
                "Character Accuracy": None,
                }
    pred_info["Image Path"] = image_path
    pred_info["Orientation"] = view
    pred_info["Number of Words"] = len([x for x in ground_truth.values() if x !=None])
    pred_info["Number of Characters"] = len(''.join([x for x in ground_truth.values() if x !=None]))

    '''
    '{'Batch Number':      {'Character Similarity': None,
                            'GT': None,
                            'Predicted': None},
                'Brand':   {'Character Similarity': None,
                            'GT': 'TECNIS',
                            'Predicted': 'TECNIS'},
                'Diopter': {'Character Similarity': None,
                            'GT': '+1.5',
                            'Predicted': None},
             'Expiry Date': {'Character Similarity': None,
                            'GT': '3/6/2024',
                            'Predicted': '2024-06-03'},
            'Model':       {'Character Similarity': None,
                            'GT': 'ZCT150',
                            'Predicted': None},
        'Serial Number':   {'Character Similarity': None,
                                'GT': '2525232006',
                                'Predicted': None}
                        }
    '''
    gt_predicted = copy.deepcopy(ground_truth)
    for param in gt_predicted:
        gt_predicted[param] = {"GT": gt_predicted[param], "Predicted": None, "Character Similarity": None}
    pred_info["GT - Predicted"] = gt_predicted

    single_item_acc_word_count = 0
    single_item_sim_char_count = 0

    for idx, acc_type in enumerate(ACC_TYPES):
        pred_info["Stage"] = idx + 1
        
        if acc_type == "word_level":
            for item in prediction.split():
                exact_flag = False
                for parameter in ground_truth:
                    if item == ground_truth[parameter]:
                        pred_info["GT - Predicted"][parameter]["Predicted"] = item
                        pred_info["GT - Predicted"][param]["Character Similarity"] = 100.0
                        single_item_acc_word_count += 1
                        single_item_sim_char_count += len(ground_truth[parameter])
                        exact_flag = True
                        break
                
                if not exact_flag:
                    # TODO: review if there are other similarity such functions
                    if isDate(item) and pred_info["GT - Predicted"]['Expiry Date']["Predicted"] == None:

                        print(1)
                        print(pred_info["GT - Predicted"]['Expiry Date']['GT'])
                        print(item)

                        pred_info["GT - Predicted"]['Expiry Date']["Predicted"] = item
                    elif isDiopter(item) and pred_info["GT - Predicted"]['Diopter']["Predicted"] == None:
                        print(2)
                        print(pred_info["GT - Predicted"]['Diopter']['GT'])
                        print(item)
                        item = processDiopter(item)
                        pred_info["GT - Predicted"]['Diopter']["Predicted"] = item
                    elif isBrand(item) and pred_info["GT - Predicted"]['Brand']["Predicted"] == None:

                        print(3)
                        print(pred_info["GT - Predicted"]['Brand']['GT'])
                        print(item)

                        item = item.upper()
                        pred_info["GT - Predicted"]['Brand']["Predicted"] = item
                    else: 
                        continue
                    pred_info["GT - Predicted"][param]["Character Similarity"] = 100.0
                    single_item_acc_word_count += 1
                    # TODO: Verify if the outputs of is<> functions have the correct length
                    single_item_sim_char_count += len(item) 
                    print(item)
                    print()
                
            word_accuracy = round(single_item_acc_word_count/pred_info["Number of Words"], SIG_FIG)
            pred_info["Word Accuracy"] = word_accuracy

        if acc_type == "char_level":
            for param in pred_info["GT - Predicted"]:
                if pred_info["GT - Predicted"][param]["Predicted"] == None:
                    param_similarity_score = similarity_score(ground_truth, prediction.split(), param)
                    # print("{} Similarity Score: {}".format(param, param_similarity_score))
                    if param_similarity_score != {} and max(param_similarity_score.values()) >= MIN_CHAR_SIMILARITY*100:
                        next_similar_pred = max(param_similarity_score, key=param_similarity_score.get)
                        pred_info["GT - Predicted"][param]["Predicted"] = next_similar_pred
                        score, similar_chars_count = char_similarity(
                            pred_info["GT - Predicted"][param]["Predicted"], 
                            pred_info["GT - Predicted"][param]["GT"]
                            )
                        print(score, similar_chars_count)
                        pred_info["GT - Predicted"][param]["Character Similarity"] = score
                        single_item_sim_char_count += similar_chars_count
            
            char_accuracy = round(single_item_sim_char_count/pred_info["Number of Characters"], SIG_FIG)
            pred_info["Character Accuracy"] = char_accuracy

    # TODO: Verify if the counts are correct
    return single_item_acc_word_count, single_item_sim_char_count, pred_info["Number of Words"], pred_info["Number of Characters"]

def similarity_score(gt_items: dict, pred_items: list, param: str):
    similarity_score = {}
    most_similar_param, score = None, None
    # print(gt_items, pred_items, param)
    for pred_item in pred_items:
        if param == "Brand" and len(pred_item) > 5:
            most_similar_param, score = brandSimilarity(pred_item)
            # print("brand", pred_item, most_similar_param, score)

        elif param == "Model" and pred_item.isalnum() and len(pred_item) > 4 and len(pred_item) < 11:
            most_similar_param, score = modelSimilarity(pred_item, gt_items['Brand'])
            # print(gt_items['Brand'])
            # print("model", pred_item, most_similar_param, score)

        if score != None and score > 0:
            if most_similar_param in similarity_score.keys():
                if similarity_score[most_similar_param] < score:
                    similarity_score[most_similar_param] = score
            else:
                similarity_score[most_similar_param] = score
    return similarity_score

def char_similarity(pred, gt):
  similar_chars_count = 0
  i, j = 0, 0

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

# def similarity_word_list(param_gt_pred_dict):
#     pred_list = [gt_pred_dict[param]["Predicted"] for param in param_gt_pred_dict]
#     gt_list = [gt_pred_dict[param]["GT"] for param in param_gt_pred_dict]
#     total_similar_chars = 0
#     total_gt_chars = 0

#     if len(pred_list) == 0:
#         return 0.0

#     for pred, gt in zip(pred_list, gt_list):
#         similar_chars = 0
#         i, j = 0, 0
#         while i < len(pred) and j < len(gt):
#             if pred[i] == gt[i]:
#                 similar_chars += 1
#             i += 1
#             j += 1
#         total_similar_chars += similar_chars
#         total_gt_chars += len(gt)

#     char_similarity = (total_similar_chars / total_gt_chars) * 100
#     return char_similarity

# def increment_total_count(gt, total_count, count_type):
#     for param in gt:
#         if gt[param] != None:
#             if count_type == "word":
#                 total_count += 1
#             elif count_type == "char":
#                 total_count += len(gt[param])
#     return total_count

def main():
    ds_total_word_count = 0
    ds_total_char_count = 0
    ds_total_acc_word_count = 0
    ds_total_char_acc_count = 0
    empty_gts = []
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
                            image_path = os.path.join(view_path, image_name)
                            image = cv2.imread(image_path)
                            image = preprocess(image)
                            prediction = pytesseract.image_to_string(Image.fromarray(image))
                            ground_truth = csv_utils.extract_ground_truth(brand_dirname, model_dirname, view)
                            
                            if is_empty_gt(ground_truth):
                                empty_gts.append(ground_truth)
                                continue

                            print("=============== Evaluating: {} ===============".format(image_path))
                            dp_total_acc_word_count, dp_total_acc_char_count, dp_total_word_count, dp_total_char_count = calc_accuracy(
                                image_path, view, prediction, ground_truth)
                            print("WORD-LEVEL - (Total number of words, Total number of accurate words predicted, Word accuracy) - {}, {}, {}".format(
                                dp_total_word_count, dp_total_acc_word_count, round(dp_total_acc_word_count/dp_total_word_count, SIG_FIG)))
                            print("CHAR-LEVEL - (Total number of chars, Total number of accurate chars predicted, Char accuracy) - {}, {}, {}".format(
                                dp_total_char_count, dp_total_acc_char_count, round(dp_total_acc_char_count/dp_total_char_count, SIG_FIG)))
                            print(len("=============== Evaluating: {} ===============".format(image_path))*"=")
                            print("")
    
                            ds_total_word_count += dp_total_word_count
                            ds_total_char_count += dp_total_char_count
                            ds_total_acc_word_count += dp_total_acc_word_count
                            ds_total_char_acc_count += dp_total_acc_char_count

    overall_word_acc = ds_total_acc_word_count / ds_total_word_count
    overall_char_acc = ds_total_char_acc_count / ds_total_char_count

    print("Total Word Accuracy - {} - Total Character Accuracy - {}".format(overall_word_acc, overall_char_acc))
    print("Check CSV entries against empty ground truths extracted:")
    print(empty_gt for empty_gt in empty_gts)

main()

# image_path = "/home/hwlee96/SUTD/01.116/project/Data/Sensar/AR40E_1/side/6.jpeg"
# calc_accuracy(
#     2, 
#     ["char_level"], 
#     image_path, 
#     "side", 
#     pytesseract.image_to_string(Image.fromarray(preprocess(cv2.imread(image_path)))),
#     csv_utils.extract_ground_truth("Sensar", "AR40E_1", "side"), 
#     0, 
#     0)
