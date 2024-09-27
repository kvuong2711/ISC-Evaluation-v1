import csv
import pandas as pd
import time
import numpy as np
from tqdm import tqdm
import os
from bbox_3d_detection_evaluation.nuscenes_eval_core import NuScenesEval



def read_csv_gt(file_path):
    # Read CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Initialize the dictionary to store timestamp and bounding box data
    bbox_dict = {}

    # Iterate over the DataFrame rows
    for index, row in df.iterrows():
        timestamp = row['Time']

        # In the row, get the non-nan values + column names
        non_nan_cnt = 0
        values = []
        col_names = []
        for col_name, value in row.items():
            if col_name == 'Time':
                continue
            if pd.notna(value):
                non_nan_cnt += 1
                values.append(value)
                col_names.append(col_name)

        assert len(values) % 9 == 0, f"Number of values is not a multiple of 9: {len(values)}"
        if timestamp not in bbox_dict:
            bbox_dict[timestamp] = []
        num_bboxes = len(values) // 9
        for i in range(num_bboxes):
            bbox = values[i*9:(i+1)*9]
            # append class in front: 0 for car, 2 for pedestrian
            if "Vehicle" in col_names[i*9]:
                bbox.insert(0, 0)
            else:
                bbox.insert(0, 2)
            bbox_dict[timestamp].append(bbox)

    # print('Ground truth bounding boxes:')
    # print(bbox_dict)

    # exit(0)
    return bbox_dict


def read_csv_pred(file_path):
    # Read CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Initialize the dictionary to store timestamp and bounding box data
    bbox_dict = {}

    # Iterate over the DataFrame rows
    for index, row in df.iterrows():
        timestamp = row['Timestamp']

        # In the row, get the non-nan values + column names
        non_nan_cnt = 0
        values = []
        for col_name, value in row.items():
            if col_name == 'Timestamp':
                continue
            if pd.notna(value):
                non_nan_cnt += 1
                values.append(value)

        if timestamp not in bbox_dict:
            bbox_dict[timestamp] = []
        # num_bboxes = len(values) // 10
        # for i in range(num_bboxes):
        #     bbox = values[i*10:(i+1)*10]
        #     bbox_dict[timestamp].append(bbox)

        bbox_formatted = values[2:8] + np.array([0, 0]).tolist() + [values[8]]
        bbox_formatted.insert(0, int(values[0]))
        bbox_dict[timestamp].append(bbox_formatted)

    # print('Predicted bounding boxes:')
    # print(bbox_dict)

    # exit(0)
    return bbox_dict



class_id_to_name = {0: 'car', 2: 'pedestrian'}


def prepare_eval(bbox_gt_dict, bbox_pred_dict, gt_loc=None, pred_loc=None):


    # selected_timestamp = 1700412338.260972

    # for each timestamp, get the bounding boxes
    timestamp_result_dict = {}

    for timestamp, bbox_info in bbox_pred_dict.items():
        # if timestamp == selected_timestamp:
        # Prepare the ground truth and prediction bounding boxes
        gt_labels = []
        pred_labels = []
        for bbox in bbox_info:
            class_id, x, y, z, xlen, ylen, zlen, xrot, yrot, zrot = bbox
            zrot_rad = np.radians(zrot)
            new_bbox = [class_id_to_name[class_id], x, y, z, xlen, ylen, zlen, zrot_rad, 1.0]
            pred_labels.append(new_bbox)

        for bbox in bbox_gt_dict[timestamp]:
            class_id, x, y, z, xlen, ylen, zlen, xrot, yrot, zrot = bbox
            zrot_rad = np.radians(zrot)
            new_bbox = [class_id_to_name[class_id], x, y, z, xlen, ylen, zlen, zrot_rad, 1.0]
            gt_labels.append(new_bbox)
        
        timestamp_result_dict[timestamp] = (gt_labels, pred_labels)

    # print('Ground truth bounding boxes:')
    # print(gt_labels)

    # print('Predicted bounding boxes:')
    # print(pred_labels)

    # Write the ground truth and prediction bounding boxes to txt files
    os.makedirs(gt_loc, exist_ok=True)
    os.makedirs(pred_loc, exist_ok=True)

    # Write the ground truth and prediction bounding boxes to txt files
    for timestamp, (gt_labels, pred_labels) in timestamp_result_dict.items():
        with open(f'{gt_loc}/{timestamp}.txt', 'w') as f:
            for bbox in gt_labels:
                f.write(', '.join(map(str, bbox)) + '\n')

        with open(f'{pred_loc}/{timestamp}.txt', 'w') as f:
            for bbox in pred_labels:
                f.write(', '.join(map(str, bbox)) + '\n')




if __name__ == '__main__':
    bbox_dict = read_csv_gt('./Run_1253_GT.csv')
    print(f'Number of timestamps in the CSV file: {len(bbox_dict)}')

    bbox_pred_dict = read_csv_pred('./tracked_objects_Run_1253 (3).csv')
    print(f'Number of timestamps in the CSV file: {len(bbox_pred_dict)}')

    gt_loc = './gt_labels'
    pred_loc = './pred_labels'

    prepare_eval(bbox_dict, bbox_pred_dict, gt_loc, pred_loc)

    # Eval
    classes = ['car', 'pedestrian']
    format = 'class x y z l w h r score'
    save_loc = './eval_results'
    os.makedirs(save_loc, exist_ok=True)
    NuScenesEval(pred_loc, gt_loc, format, save_loc,
                classes=classes,
                max_range=0.0,
                min_score=0.0)
    