import csv
import os
import sys

VIEWS = {"back": "Back View Details", "side": "Side View Details"}
PARAMS = ["Brand", "Model", "Diopter", "Serial Number", "Batch Number", "Expiry Date"]

def get_fields_rows(csv_file_path):
    field_to_col_idx = {}
    with open(csv_file_path, 'r', encoding='utf-8-sig') as csv_file:
        reader = csv.reader(csv_file)
        fields = next(reader) # Reads header row as a list
        rows = list(reader)   # Reads all subsequent rows as a list of lists
        # for i in rows:
        #     print(i)
        for column_number, field in enumerate(fields):
            field_to_col_idx[field] = column_number
        return field_to_col_idx, rows

def extract_ground_truth(csv_file_path, brand, model, view):
    ground_truth = {}
    view_col_name = VIEWS[view]
    field_to_col_idx, rows= get_fields_rows(csv_file_path)
    
    # print(field_to_col_idx)
    for row in rows:
        # print("------")
        # print(row)
        # print(row[field_to_col_idx["Model Directory Name"]])
        # print(row[field_to_col_idx["Brand"]])
        if model == row[field_to_col_idx["Model Directory Name"]] and brand == row[field_to_col_idx["Brand"]]:
            detail_type_list = row[field_to_col_idx[view_col_name]].split("_")
            for detail_type in detail_type_list:
                detail_val = row[field_to_col_idx[detail_type]]
                if detail_val != "": # Can remove if all is populated
                    ground_truth[detail_type] = row[field_to_col_idx[detail_type]]

    for param in PARAMS:
        if param not in list(ground_truth.keys()):
            ground_truth[param] = None

    return ground_truth

# print(extract_ground_truth("Sensar", "AAB00", "back"))

def initialize(csvFileName: str, csvFieldNames: list):
    if not os.path.exists(csvFileName):
        with open(csvFileName, mode="w",  newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csvFieldNames)
            writer.writeheader()


def write_to_csv(csvFileName: str, content: dict):
    with open(csvFileName, mode="a",  newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(content.keys()))
        writer.writerow(content)

def stage_result(image_path, stage_no, stage_info):
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
    return content
