import argparse
import math
import os

import config
from get_mlkc_pkc import save_data

from lib.util.parse import question2concept_from_Q
from lib.util.FileManager import FileManager
from lib.util.data import read_preprocessed_file, load_json


def get_last_frkc(user_data, q2c, num_concept, theta):
    seq_len = user_data["seq_len"]
    time_seq = user_data["time_seq"][:seq_len]
    last_time = time_seq[-1]
    question_seq = user_data["question_seq"][:seq_len]
    # 以小时为时间单位
    delta_t_max = (last_time - time_seq[0]) / 60
    delta_t_from_last = [delta_t_max] * num_concept
    for t, q in zip(time_seq[:-1], question_seq[:-1]):
        cs = q2c[q]
        delta_t = (last_time - t) / 60
        for c in cs:
            delta_t_from_last[c] = delta_t
    frkc = []
    for delta_t in delta_t_from_last:
        frkc.append(1 - math.exp(-theta * delta_t))
    return frkc


def get_efr(user_id, frkc, q2c):
    efr = []
    for q, cs in q2c.items():
        efr.append(round(sum([frkc[c] for c in cs]) / len(cs), 2))
    return user_id, efr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 遗忘模型参数
    parser.add_argument("--theta", type=float, default=0.2)
    # 数据配置
    parser.add_argument("--dataset_name", type=str, default="statics2011")
    parser.add_argument("--data_type", type=str, default="single_concept",
                        choices=("multi_concept", "single_concept", "only_question"))
    parser.add_argument("--user_ids_path", type=str,
                        default="/Users/dream/myProjects/dlkt-release/lab/settings/kg4ex_setting/statics2011_user_ids.json")
    parser.add_argument("--train_file_path", type=str,
                        default="/Users/dream/myProjects/dlkt-release/lab/settings/kg4ex_setting/statics2011_train.txt")
    parser.add_argument("--valid_file_path", type=str,
                        default="/Users/dream/myProjects/dlkt-release/lab/settings/kg4ex_setting/statics2011_valid.txt")
    parser.add_argument("--test_file_path", type=str,
                        default="/Users/dream/myProjects/dlkt-release/lab/settings/kg4ex_setting/statics2011_test.txt")
    parser.add_argument("--target_dir", type=str,
                        default="/Users/dream/myProjects/dlkt-release/lab/settings/kg4ex_setting")

    args = parser.parse_args()
    params = vars(args)

    dataset_train = read_preprocessed_file(params["train_file_path"])
    dataset_valid = read_preprocessed_file(params["valid_file_path"])
    dataset_test = read_preprocessed_file(params["test_file_path"])
    data = dataset_train + dataset_valid + dataset_test
    user_ids = load_json(params["user_ids_path"])
    data_train = []
    data_valid = []
    data_test = []
    for item_data in data:
        user_id = item_data["user_id"]
        if user_id in user_ids["train"]:
            data_train.append(item_data)
        elif user_id in user_ids["valid"]:
            data_valid.append(item_data)
        else:
            data_test.append(item_data)

    file_manager = FileManager(config.FILE_MANAGER_ROOT)
    if params["data_type"] == "only_question":
        Q_table = file_manager.get_q_table(params["dataset_name"], "multi_concept")
    else:
        Q_table = file_manager.get_q_table(params["dataset_name"], params["data_type"])
    question2concept = question2concept_from_Q(Q_table)

    efr_train = []
    efr_valid = []
    efr_test = []
    for item_data in data_train:
        efr_train.append(
            get_efr(item_data["user_id"], get_last_frkc(item_data, question2concept, Q_table.shape[1], params["theta"]),
                    question2concept)
        )
    for item_data in data_valid:
        efr_valid.append(
            get_efr(item_data["user_id"], get_last_frkc(item_data, question2concept, Q_table.shape[1], params["theta"]),
                    question2concept)
        )
    for item_data in data_test:
        efr_test.append(
            get_efr(item_data["user_id"], get_last_frkc(item_data, question2concept, Q_table.shape[1], params["theta"]),
                    question2concept)
        )

    target_dir = params["target_dir"]
    dataset_name = params["dataset_name"]
    efr_train_path = os.path.join(target_dir, f"{dataset_name}_efr_train.txt")
    efr_valid_path = os.path.join(target_dir, f"{dataset_name}_efr_valid.txt")
    efr_test_path = os.path.join(target_dir, f"{dataset_name}_efr_test.txt")

    save_data(efr_train_path, efr_train)
    save_data(efr_valid_path, efr_valid)
    save_data(efr_test_path, efr_test)
