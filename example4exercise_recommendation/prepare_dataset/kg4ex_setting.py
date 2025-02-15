import argparse
import random
import os

import config

from lib.util.FileManager import FileManager
from lib.util.parse import parse_data_type
from lib.util.data import read_preprocessed_file, write2file, write_json
from lib.dataset.split_seq import dataset_truncate2multi_seq


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="statics2011")
    # setting config
    parser.add_argument("--setting_name", type=str, default="kg4ex_setting")
    parser.add_argument("--data_type", type=str, default="single_concept",
                        choices=("multi_concept", "single_concept", "only_question"))

    args = parser.parse_args()
    params = vars(args)
    objects = {"file_manager": FileManager(config.FILE_MANAGER_ROOT)}

    params["lab_setting"] = {
        "name": params["setting_name"],
        "max_seq_len": 200,
        "min_seq_len": 3
    }
    objects["file_manager"].add_new_setting(params["lab_setting"]["name"], params["lab_setting"])

    parse_data_type(params["dataset_name"], params["data_type"])
    data_uniformed_path = objects["file_manager"].get_preprocessed_path(params["dataset_name"], params["data_type"])
    data_uniformed = read_preprocessed_file(data_uniformed_path)
    dataset_truncated = dataset_truncate2multi_seq(data_uniformed, 3, 200,
                                                   single_concept=params["data_type"] != "multi_concept")
    n = len(dataset_truncated)
    train_user_ids = []
    valid_user_ids = []
    test_user_ids = []
    num1 = n * 0.7
    num2 = n * 0.8
    for i, item_data in enumerate(dataset_truncated):
        if i < num1:
            train_user_ids.append(i)
        elif i < num2:
            valid_user_ids.append(i)
        else:
            test_user_ids.append(i)
        item_data["user_id"] = i

    random.shuffle(dataset_truncated)
    n1 = int(n * 0.64)
    n2 = int(n * 0.8)
    kt_dataset_train = dataset_truncated[:n1]
    kt_dataset_valid = dataset_truncated[n1:n2]
    kt_dataset_test = dataset_truncated[n2:]

    setting_dir = objects["file_manager"].get_setting_dir(params["lab_setting"]["name"])
    write2file(kt_dataset_train, os.path.join(setting_dir, f"{params['dataset_name']}_train.txt"))
    write2file(kt_dataset_valid, os.path.join(setting_dir, f"{params['dataset_name']}_valid.txt"))
    write2file(kt_dataset_test, os.path.join(setting_dir, f"{params['dataset_name']}_test.txt"))

    # 存储训练ER模型时的user ids
    write_json({
        "train": train_user_ids,
        "valid": valid_user_ids,
        "test": test_user_ids
    }, os.path.join(setting_dir, f"{params['dataset_name']}_user_ids.json"))

    # 存储relations
    relations_path = os.path.join(setting_dir, "relations.dict")
    if not os.path.exists(relations_path):
        scores = [round(i * 0.01, 2) for i in range(101)]
        with open(relations_path, "w") as fs:
            for i, s in enumerate(scores):
                fs.write(f"{i}\tmlkc{s}\n")
            for i, s in enumerate(scores):
                fs.write(f"{i+101}\tpkc{s}\n")
            for i, s in enumerate(scores):
                fs.write(f"{i+202}\tefr{s}\n")
            fs.write("303\trec")
