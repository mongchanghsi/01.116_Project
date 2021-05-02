import os
import copy 
from pprint import pprint

from utils.isDate import isDate
from utils.isDiopter import isDiopter, processDiopter
from utils.isBrand import brandSimilarity
from utils.isModel import modelSimilarity
from utils.isSerial import isSerial, isSerial_2
from utils.isBatch import batchSimilarity, isBatch

from utils.preprocessing.blur import medianBlur, averageBlur, gaussianBlur, bilateralBlur
from utils.preprocessing.threshold import threshold

from utils import csv_utils

SIG_FIG = 3

def char_similarity(pred, gt):
  print(pred, gt)
  similar_chars_count = 0
  i, j = 0, 0
  while i < len(pred) and j < len(gt):
    if pred[i] == gt[i]:
      similar_chars_count += 1
    i += 1
    j += 1
  score = (similar_chars_count / len(gt)) * 100
  return score, similar_chars_count

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

def increment_score(param, stage_info, single_item_sim_char_count, single_item_acc_word_count):
    ground_truth = stage_info["GT - Predicted"][param]["GT"]
    predicted = stage_info["GT - Predicted"][param]["Predicted"]
    score, similar_chars_count = char_similarity(predicted, ground_truth)

    stage_info["GT - Predicted"][param]["Character Similarity"] = score
    single_item_sim_char_count += similar_chars_count
    if similar_chars_count == len(ground_truth):    
        single_item_acc_word_count += 1
    
    return stage_info, single_item_sim_char_count, single_item_acc_word_count

def is_pred_param_not_filled(param, stage_info):
    return stage_info["GT - Predicted"][param]["GT"] != None and \
            stage_info["GT - Predicted"][param]["Predicted"] == None

def is_exact(item, param, stage_info):
    return item == stage_info["GT - Predicted"][param]["GT"]

def is_param_pred_gt_exact(param, stage_info):
    ground_truth = stage_info["GT - Predicted"][param]["GT"]
    predicted = stage_info["GT - Predicted"][param]["Predicted"]    
    return ground_truth != None and predicted != None and ground_truth == predicted

# TODO: to improve this - change to autocorrect form
def is_valid_date(pred):
    if "-" in pred:
        dmy = pred.split("-")
    elif "/" in pred:
        dmy = pred.split("/")
    else:
        return False
    
    return len(dmy) == 3

def stage_eval(stage_no, stage_info, pred_info, single_item_acc_word_count, single_item_sim_char_count):
    for param in stage_info["GT - Predicted"]:
        ground_truth = stage_info["GT - Predicted"][param]["GT"]
        predicted = stage_info["GT - Predicted"][param]["Predicted"]
        if ground_truth != None and predicted != None:

            # Bad implentation: Remove this next time
            if param == "Expiry Date" and is_valid_date(predicted):
                
                stage_info["GT - Predicted"][param]["Character Similarity"] = 100.0
                single_item_sim_char_count +=  len(ground_truth)
                single_item_acc_word_count += 1

            else:
                score, similar_chars_count = char_similarity(predicted, ground_truth)

                stage_info["GT - Predicted"][param]["Character Similarity"] = score
                single_item_sim_char_count += similar_chars_count
                if similar_chars_count == len(ground_truth):
                    single_item_acc_word_count += 1

    stage_info["Number of Accurate Words"] = single_item_acc_word_count
    stage_info["Number of Accurate Characters"] = single_item_sim_char_count
    word_accuracy = round(stage_info["Number of Accurate Words"]/stage_info["Number of Words"], SIG_FIG)
    char_accuracy = round(stage_info["Number of Accurate Characters"]/stage_info["Number of Characters"], SIG_FIG)
    stage_info["Word Accuracy"] = word_accuracy
    stage_info["Character Accuracy"] = char_accuracy

    pred_info["Stages"][stage_no] = copy.deepcopy(stage_info)

    return stage_info, pred_info, single_item_acc_word_count, single_item_sim_char_count

def extract_stage_3_info(pred_info):
    result = {}
    for param in csv_utils.PARAMS:
        result[param] = pred_info["Stages"][3]['GT - Predicted'][param]['Predicted']
    return result

# def combineData(metadata_GB, metadata_GBT, ground_truth):
#   metadata_final = {}
#   for param in csv_utils.PARAMS:
#     metadata_final[param] = None
  
#   for i in metadata_final.keys():
#     if metadata_GB[i] == metadata_GBT[i]:
#       metadata_final[i] = metadata_GB[i]
#     elif metadata_GB[i] == None and metadata_GBT[i] != None:
#       metadata_final[i] = metadata_GBT[i]
#     elif metadata_GB[i] != None and metadata_GBT[i] == None:
#       metadata_final[i] = metadata_GB[i]
#     else:
#       if i == 'Batch Number':
#         if isBatch(metadata_GBT[i], metadata_GBT['Model']):
#           metadata_final[i] = metadata_GBT[i]
#         elif isBatch(metadata_GB[i], metadata_GB['Model']):
#           metadata_final[i] = metadata_GB[i]
#       if i == 'Diopter':
#         if len(metadata_GB[i]) == 4 or len(metadata_GB[i]) == 5:
#           metadata_final[i] = metadata_GB[i]
#         elif len(metadata_GBT[i]) == 4 or len(metadata_GBT[i]) == 5:
#           metadata_final[i] = metadata_GBT[i]
#       elif i == 'Serial Number':
#         metadata_final[i] = metadata_GB[i]
#       elif i == 'Expiry Date':
#         pass

#   return metadata_final

# def test(metadata_final, ground_truth):
#     single_item_sim_char_count = 0
#     single_item_sim_word_count = 0
#     for param in metadata_final:
#         ground_truth = ground_truth[param]
#         predicted = metadata_final[param]
#         if ground_truth != None and predicted != None:

#             # Bad implentation: Remove this next time
#             if param == "Expiry Date" and is_valid_date(predicted):
                
#                 stage_info["GT - Predicted"][param]["Character Similarity"] = 100.0
#                 single_item_sim_char_count +=  len(ground_truth)
#                 single_item_acc_word_count += 1

#             else:
#                 score, similar_chars_count = char_similarity(predicted, ground_truth)

#                 stage_info["GT - Predicted"][param]["Character Similarity"] = score
#                 single_item_sim_char_count += similar_chars_count
#                 if similar_chars_count == len(ground_truth):
#                     single_item_acc_word_count += 1

#     return single_item_sim_word_count, single_item_sim_char_count