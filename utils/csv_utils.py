import csv
import os
import sys

CSV_FILE_PATH = "/mnt/c/Users/user/OneDrive - Singapore University of Technology and Design/Term 7/01.116/1D Project/01.116_IHIS_Project/Data/Combined_2/Sample_Images_Data_Dictionary_05032021.csv"
VIEWS = {"back": "Back View Details", "side": "Side View Details"}
PARAMS = ["Brand", "Model", "Diopter", "Serial Number", "Batch Number", "Expiry Date"]

def get_fields_rows():
    field_to_col_idx = {}
    with open(CSV_FILE_PATH, 'r', encoding='utf-8-sig') as csv_file:
        reader = csv.reader(csv_file)
        fields = next(reader) # Reads header row as a list
        rows = list(reader)   # Reads all subsequent rows as a list of lists
        # for i in rows:
        #     print(i)
        for column_number, field in enumerate(fields):
            field_to_col_idx[field] = column_number
        return field_to_col_idx, rows

def get_brands():
    col_name = "Brand Directory Name"
    field_to_col_idx, rows = get_fields_rows()
    col_vals = []

    for row in rows:
        col_val = row[field_to_col_idx[col_name]]
        if col_val not in col_vals:
            col_vals.append(row[field_to_col_idx[col_name]])

    return col_vals

def extract_ground_truth(brand, model, view):
    ground_truth = {}
    view_col_name = VIEWS[view]
    field_to_col_idx, rows= get_fields_rows()
    
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

# EVAL_CSV_FILEPATH = "logs/good_all.csv"

# csv_field_names = ["Image Path", "Stage", "Number of Words", "Number of Accurate Words", "Word Accuracy",
#                     "Number of Characters", "Number of Accurate Characters", "Character Accuracy"]

# if not os.path.exists(EVAL_CSV_FILEPATH):
#     initialize(EVAL_CSV_FILEPATH, csv_field_names)