import argparse
import os
import math
import numpy as np

import config

from lib.util.parse import question2concept_from_Q
from lib.util.FileManager import FileManager


def read_data(f_path):
    data = {}
    with open(f_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        line_ = line.split(":")
        user_id, data_value = line_[0], line_[1]
        data[int(user_id)] = list(map(float, data_value.split(",")))
    return data


def cosine_similarity(list1, list2):
    # 将列表转换为NumPy数组
    arr1 = np.array(list1)
    arr2 = np.array(list2)

    # 计算点积
    dot_product = np.dot(arr1, arr2)

    # 计算向量的模
    norm_arr1 = np.linalg.norm(arr1)
    norm_arr2 = np.linalg.norm(arr2)

    # 计算余弦相似度
    if norm_arr1 == 0 or norm_arr2 == 0:
        return 0.0  # 避免除以零的情况
    cosine_sim = dot_product / (norm_arr1 * norm_arr2)

    return cosine_sim


def get_recommended_exercises(q2c, q_table, mlkc_, pkc_, efr_, delta1=0.7, delta2=0.7, top_n=10):
    scores = []
    for q, cs in q2c.items():
        score1 = 1
        for c in cs:
            score1 *= mlkc_[c]
        score1 = math.pow(delta1 - score1, 2)
        score2 = math.pow(cosine_similarity(q_table[q], pkc_), 2)
        score3 = math.pow(delta2 - efr_[q], 2)
        scores.append((q, score1 + score2 + score3))
    return list(map(lambda x: x[0], sorted(scores, key=lambda x: x[1])))[:top_n]


def save_triples(triples_path, mlkc_all, pkc_all, efr_all, rec_exs=None):
    with open(triples_path, "w") as f:
        for user_id, mlkc_ in mlkc_all.items():
            for k, m in enumerate(mlkc_):
                f.write(f"kc{k}\tmlkc{m}\tuid{user_id}\n")
        for user_id, pkc_ in pkc_all.items():
            for k, p in enumerate(pkc_):
                f.write(f"kc{k}\tpkc{p}\tuid{user_id}\n")
        for user_id, efr_ in efr_all.items():
            for q, e in enumerate(efr_):
                f.write(f"ex{q}\tefr{e}\tuid{user_id}\n")
        if rec_exs is not None:
            for user_id, rec_ex_ in rec_exs.items():
                for q in rec_ex_:
                    f.write(f"uid{user_id}\trec\tex{q}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="statics2011")
    parser.add_argument("--data_type", type=str, default="single_concept",
                        choices=("multi_concept", "single_concept", "only_question"))
    parser.add_argument("--target_dir", type=str,
                        default="/Users/dream/myProjects/dlkt-release/lab/settings/kg4ex_setting")
    parser.add_argument("--delta1", type=float, default=0.7)
    parser.add_argument("--delta2", type=float, default=0.7)
    parser.add_argument("--top_n", type=int, default=10)
    args = parser.parse_args()
    params = vars(args)

    file_manager = FileManager(config.FILE_MANAGER_ROOT)
    if params["data_type"] == "only_question":
        Q_table = file_manager.get_q_table(params["dataset_name"], "multi_concept")
    else:
        Q_table = file_manager.get_q_table(params["dataset_name"], params["data_type"])
    question2concept = question2concept_from_Q(Q_table)

    target_dir = params["target_dir"]
    dataset_name = params["dataset_name"]
    mlkc_train_path = os.path.join(target_dir, f"{dataset_name}_mlkc_train.txt")
    mlkc_test_path = os.path.join(target_dir, f"{dataset_name}_mlkc_test.txt")
    pkc_train_path = os.path.join(target_dir, f"{dataset_name}_pkc_train.txt")
    pkc_test_path = os.path.join(target_dir, f"{dataset_name}_pkc_test.txt")

    target_dir = params["target_dir"]
    dataset_name = params["dataset_name"]

    mlkc_train = read_data(os.path.join(target_dir, f"{dataset_name}_mlkc_train.txt"))
    mlkc_valid = read_data(os.path.join(target_dir, f"{dataset_name}_mlkc_valid.txt"))
    mlkc_test = read_data(os.path.join(target_dir, f"{dataset_name}_mlkc_test.txt"))

    pkc_train = read_data(os.path.join(target_dir, f"{dataset_name}_pkc_train.txt"))
    pkc_valid = read_data(os.path.join(target_dir, f"{dataset_name}_pkc_valid.txt"))
    pkc_test = read_data(os.path.join(target_dir, f"{dataset_name}_pkc_test.txt"))

    efr_train = read_data(os.path.join(target_dir, f"{dataset_name}_efr_train.txt"))
    efr_valid = read_data(os.path.join(target_dir, f"{dataset_name}_efr_valid.txt"))
    efr_test = read_data(os.path.join(target_dir, f"{dataset_name}_efr_test.txt"))

    rec_ex_train = {}
    train_user_ids = mlkc_train.keys()
    for train_user_id in train_user_ids:
        mlkc, pkc, efr = mlkc_train[train_user_id], pkc_train[train_user_id], efr_train[train_user_id]
        rec_ex_train[train_user_id] = get_recommended_exercises(
            question2concept, Q_table, mlkc, pkc, efr, params["delta1"], params["delta2"], params["top_n"]
        )

    target_dir = params["target_dir"]
    dataset_name = params["dataset_name"]
    triples_train_path = os.path.join(target_dir, f"{dataset_name}_train_triples.txt")
    triples_valid_path = os.path.join(target_dir, f"{dataset_name}_valid_triples.txt")
    triples_test_path = os.path.join(target_dir, f"{dataset_name}_test_triples.txt")

    save_triples(triples_train_path, mlkc_train, pkc_train, efr_train, rec_ex_train)
    save_triples(triples_valid_path, mlkc_valid, pkc_valid, efr_valid)
    save_triples(triples_test_path, mlkc_test, pkc_test, efr_test)

    # 存储entities
    num_user = len(mlkc_train) + len(mlkc_valid) + len(mlkc_test)
    num_question, num_concept = Q_table.shape[0], Q_table.shape[1]
    with open(os.path.join(target_dir, f"{dataset_name}_entities.dict"), "w") as fs:
        for i in range(num_user):
            fs.write(f"{i}\tuid{i}\n")
        for i in range(num_concept):
            fs.write(f"{i+num_user}\tkc{i}\n")
        for i in range(num_question):
            fs.write(f"{i+num_user+num_concept}\tex{i}\n")
