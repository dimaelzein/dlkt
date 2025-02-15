import argparse
import os

import config

from lib.config.dlkt_roster_config import roster_general_config
from lib.util.load_model import load_kt_model
from lib.roster.KTRoster import KTRoster
from lib.util.parse import str2bool
from lib.util.data import read_preprocessed_file, load_json


def data2batches(data_, batch_size):
    batches = []
    batch = []
    for item_data in data_:
        if len(batch) < batch_size:
            batch.append(item_data)
        else:
            batches.append(batch)
            batch = [item_data]
    if len(batch) > 0:
        batches.append(batch)
    return batches


def get_target(roster, data_batches):
    mlkc_all_user = []
    for batch in data_batches:
        users_mlkc = roster.get_last_concept_mastery_level(batch).detach().cpu().numpy().tolist()
        for i, user_mlkc in enumerate(users_mlkc):
            for j, mlkc in enumerate(user_mlkc):
                users_mlkc[i][j] = round(mlkc, 2)
        user_ids = [item_data["user_id"] for item_data in batch]
        for user_id, user_mlkc in zip(user_ids, users_mlkc):
            mlkc_all_user.append((user_id, user_mlkc))
    return mlkc_all_user


def save_data(data_path, data_values):
    with open(data_path, "w") as f:
        for data_value in data_values:
            user_id, target_value = data_value
            f.write(f"{user_id}:" + ','.join(list(map(str, target_value))) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # device配置
    parser.add_argument("--debug_mode", type=str2bool, default=False)
    parser.add_argument("--use_cpu", type=str2bool, default=False)
    # mlkc和pkc获取方法一样，选择获取哪个
    parser.add_argument("--get_mlkc", type=str2bool, default=False)
    parser.add_argument("--target_dir", type=str,
                        default="/Users/dream/myProjects/dlkt-release/lab/settings/kg4ex_setting")
    # 加载KT模型
    parser.add_argument("--save_model_dir", type=str, help="绝对路径",
                        default=r"/Users/dream/myProjects/dlkt-release/lab/saved_models/DKT_KG4EX@@kg4ex_setting@@statics2011_train@@seed_0@@2025-02-15@09-12-17")
    parser.add_argument("--save_model_name", type=str, help="文件名", default="saved.ckt")
    parser.add_argument("--model_name_in_ckt", type=str, help="文件名", default="best_valid")
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
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()
    params = vars(args)

    global_params, global_objects = roster_general_config(params, config.FILE_MANAGER_ROOT)
    file_manager = global_objects["file_manager"]

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

    batches_train = data2batches(data_train, params["batch_size"])
    batches_valid = data2batches(data_valid, params["batch_size"])
    batches_test = data2batches(data_test, params["batch_size"])

    model = load_kt_model(global_params, global_objects, params["save_model_dir"], params["save_model_name"], params["model_name_in_ckt"])
    model.eval()
    global_objects["models"] = {"kt_model": model}
    kt_roster = KTRoster(global_params, global_objects)
    target_train = get_target(kt_roster, batches_train)
    target_valid = get_target(kt_roster, batches_valid)
    target_test = get_target(kt_roster, batches_test)

    target_dir = params["target_dir"]
    dataset_name = params["dataset_name"]
    target_train_path = os.path.join(target_dir, f"{dataset_name}_mlkc_train.txt" if params[
        "get_mlkc"] else f"{dataset_name}_pkc_train.txt")
    target_valid_path = os.path.join(target_dir, f"{dataset_name}_mlkc_valid.txt" if params[
        "get_mlkc"] else f"{dataset_name}_pkc_valid.txt")
    target_test_path = os.path.join(target_dir, f"{dataset_name}_mlkc_test.txt" if params[
        "get_mlkc"] else f"{dataset_name}_pkc_test.txt")

    save_data(target_train_path, target_train)
    save_data(target_valid_path, target_valid)
    save_data(target_test_path, target_test)
