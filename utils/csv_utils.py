import csv
import os
import sys

CSV_FILE_PATH = "/home/hwlee96/SUTD/01.116/project/Data/Sample_Images_Data_Dictionary_05032021.csv"
VIEWS = {"back": "Back View Details", "side": "Side View Details"}

def get_fields_rows():
    field_to_col_idx = {}
    with open(CSV_FILE_PATH, 'r', encoding='utf-8-sig') as csv_file:
        reader = csv.reader(csv_file)
        fields = next(reader) # Reads header row as a list
        rows = list(reader)   # Reads all subsequent rows as a list of lists
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
    
    for row in rows:
        if model in row[field_to_col_idx["Model"]] and brand in row[field_to_col_idx["Brand"]]:
            detail_type_list = row[field_to_col_idx[view_col_name]].split("_")
            for detail_type in detail_type_list:
                detail_val = row[field_to_col_idx[detail_type]]
                if detail_val != "": # Can remove if all is populated
                    ground_truth[detail_type] = row[field_to_col_idx[detail_type]]

    return ground_truth
    