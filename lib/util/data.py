import os
import json
import torch
import hashlib
import numpy as np
from copy import deepcopy
from collections import defaultdict
from scipy.stats import norm
from scipy.stats import poisson


def get_keys_from_uniform(data_uniformed):
    item_data = data_uniformed[0]
    id_keys = []
    for k in item_data.keys():
        if type(item_data[k]) is not list:
            id_keys.append(k)
    seq_keys = list(set(item_data.keys()) - set(id_keys))
    return id_keys, seq_keys


def write2file(data, data_path):
    # id_keys表示序列级别的特征，如user_id, seq_len
    # seq_keys表示交互级别的特征，如question_id, concept_id
    id_keys = []
    seq_keys = []
    for key in data[0].keys():
        if type(data[0][key]) == list:
            seq_keys.append(key)
        else:
            id_keys.append(key)

    # 不知道为什么，有的数据集到这的时候，数据变成float类型了（比如junyi2015，如果预处理部分数据，就是int，但是如果全量数据，就是float）
    id_keys_ = set(id_keys).intersection({"user_id", "school_id", "premium_pupil", "gender", "seq_len", "campus",
                                          "dataset_type", "order"})
    seq_keys_ = set(seq_keys).intersection({"question_seq", "concept_seq", "correct_seq", "time_seq", "use_time_seq",
                                            "use_time_first_seq", "num_hint_seq", "num_attempt_seq", "age_seq",
                                            "question_mode_seq"})
    for item_data in data:
        for k in id_keys_:
            try:
                item_data[k] = int(item_data[k])
            except ValueError:
                print(f"value of {k} has nan")
        for k in seq_keys_:
            try:
                item_data[k] = list(map(int, item_data[k]))
            except ValueError:
                print(f"value of {k} has nan")

    with open(data_path, "w") as f:
        first_line = ",".join(id_keys) + ";" + ",".join(seq_keys) + "\n"
        f.write(first_line)
        for item_data in data:
            for k in id_keys:
                f.write(f"{item_data[k]}\n")
            for k in seq_keys:
                f.write(",".join(map(str, item_data[k])) + "\n")


def read_preprocessed_file(data_path):
    assert os.path.exists(data_path), f"{data_path} not exist"
    with open(data_path, "r") as f:
        all_lines = f.readlines()
        first_line = all_lines[0].strip()
        seq_interaction_keys_str = first_line.split(";")
        id_keys_str = seq_interaction_keys_str[0].strip()
        seq_keys_str = seq_interaction_keys_str[1].strip()
        id_keys = id_keys_str.split(",")
        seq_keys = seq_keys_str.split(",")
        keys = id_keys + seq_keys
        num_key = len(keys)
        all_lines = all_lines[1:]
        data = []
        for i, line_str in enumerate(all_lines):
            if i % num_key == 0:
                item_data = {}
            current_key = keys[int(i % num_key)]
            if current_key in ["time_factor_seq", "hint_factor_seq", "attempt_factor_seq", "correct_float_seq"]:
                line_content = list(map(float, line_str.strip().split(",")))
            else:
                line_content = list(map(int, line_str.strip().split(",")))
            if len(line_content) == 1:
                # 说明是序列级别的特征，即user id、seq len、segment index等等
                item_data[current_key] = line_content[0]
            else:
                # 说明是interaction级别的特征，即question id等等
                item_data[current_key] = line_content
            if i % num_key == (num_key - 1):
                data.append(item_data)

    return data


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        result = json.load(f)
    return result


def write_json(json_data, json_path):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)


def get_concept_from_question(q_table, question_id):
    return np.argwhere(q_table[question_id] == 1).reshape(-1).tolist()


def dataset_delete_pad(dataset):
    id_keys, seq_keys = get_keys_from_uniform(dataset)
    data_uniformed = []
    for item_data in dataset:
        item_data_new = deepcopy(item_data)
        mask_seq = item_data_new["mask_seq"]
        end_index = mask_seq.index(0) if mask_seq[-1] != 1 else len(mask_seq)
        for k in seq_keys:
            item_data_new[k] = item_data_new[k][0:end_index]
        item_data_new["seq_len"] = len(item_data_new["correct_seq"])
        data_uniformed.append(item_data_new)
    return data_uniformed


def data_pad(data_uniformed, max_seq_len=200, padding_value=0):
    dataset_new = []
    id_keys, seq_keys = get_keys_from_uniform(data_uniformed)
    for item_data in data_uniformed:
        item_new = {k: item_data[k] for k in id_keys}
        seq_len = len(item_data["correct_seq"])
        for k in seq_keys:
            item_new[k] = item_data[k] + [padding_value] * (max_seq_len - seq_len)
        dataset_new.append(item_new)
    return dataset_new


def dataset_agg_concept(data_uniformed):
    """
    用于将数据中question_seq序列为-1的去掉，也就是生成single数据，不做multi
    :param data_uniformed:
    :return:
    """
    data_uniformed = dataset_delete_pad(data_uniformed)
    data_new = []
    id_keys, seq_keys = get_keys_from_uniform(data_uniformed)
    for item_data in data_uniformed:
        item_data_new = {}
        for key in id_keys:
            item_data_new[key] = item_data[key]
        for key in seq_keys:
            item_data_new[key] = []
        for i, q_id in enumerate(item_data["question_seq"]):
            if q_id != -1:
                for key in seq_keys:
                    item_data_new[key].append(item_data[key][i])
        item_data_new["seq_len"] = len(item_data_new["correct_seq"])
        data_new.append(item_data_new)
    return data_new


def data_agg_question(data_uniformed):
    """
    将multi concept的数据中question seq里的-1替换为对应的q id
    :param data_uniformed:
    :return:
    """
    id_keys, seq_keys = get_keys_from_uniform(data_uniformed)
    if "question_seq" not in seq_keys:
        return data_uniformed

    data_converted = []
    for item_data in data_uniformed:
        item_data_new = {}
        for k in id_keys:
            item_data_new[k] = item_data[k]
        for k in seq_keys:
            if k == "question_seq":
                question_seq = item_data["question_seq"]
                question_seq_new = []
                current_q = question_seq[0]
                for q in question_seq:
                    if q != -1:
                        current_q = q
                    question_seq_new.append(current_q)
                item_data_new["question_seq"] = question_seq_new
            else:
                item_data_new[k] = deepcopy(item_data[k])
        data_converted.append(item_data_new)

    return data_converted


def dataset_multi_concept2only_question(dataset_multi_concept, max_seq_len=200):
    dataset_only_question = dataset_agg_concept(dataset_multi_concept)
    for item_data in dataset_only_question:
        del item_data["concept_seq"]
        for k in item_data.keys():
            if type(item_data[k]) is list:
                item_data[k] += [0] * (max_seq_len - item_data["seq_len"])

    return dataset_only_question


def drop_qc(data_uniformed, num2drop=30):
    """
    丢弃练习次数少于指定值的习题，如DIMKT丢弃练习次数少于30次的习题和知识点
    :param data_uniformed:
    :param num2drop:
    :return:
    """
    data_uniformed = deepcopy(data_uniformed)
    id_keys, seq_keys = get_keys_from_uniform(data_uniformed)
    questions_frequency = defaultdict(int)

    for item_data in data_uniformed:
        for question_id in item_data["question_seq"]:
            questions_frequency[question_id] += 1

    questions2drop = set()
    for q_id in questions_frequency.keys():
        if questions_frequency[q_id] < num2drop:
            questions2drop.add(q_id)

    data_dropped = []
    num_drop_interactions = 0
    for item_data in data_uniformed:
        item_data_new = {}
        for k in id_keys:
            item_data_new[k] = item_data[k]
        for k in seq_keys:
            item_data_new[k] = []
        for i in range(item_data["seq_len"]):
            q_id = item_data["question_seq"][i]
            if q_id in questions2drop:
                num_drop_interactions += 1
                continue
            for k in seq_keys:
                item_data_new[k].append(item_data[k][i])
        item_data_new["seq_len"] = len(item_data_new["question_seq"])
        if item_data_new["seq_len"] > 1:
            data_dropped.append(item_data_new)

    return data_dropped


def context2batch(dataset_train, context_list, device):
    """
    将meta数据转换为batch数据 \n
    :param dataset_train: uniformed data
    :param context_list: [{sed_id, seq_len, correct}, ...]
    :param device: cuda or cpu
    :return:
    """
    batch = {
        "question_seq": [],
        "mask_seq": [],
        "correct_seq": []
    }
    if "concept_seq" in dataset_train[0].keys():
        batch["concept_seq"] = []
    if "question_diff_seq" in dataset_train[0].keys():
        batch["question_diff_seq"] = []
    if "concept_diff_seq" in dataset_train[0].keys():
        batch["concept_diff_seq"] = []

    seq_keys = list(batch.keys())
    max_seq_len = 0
    for ctx in context_list:
        item_data = dataset_train[ctx["seq_id"]]
        seq_len = ctx["seq_len"]
        max_seq_len = max(max_seq_len, seq_len)

        batch["question_seq"].append(item_data["question_seq"][:seq_len])
        batch["mask_seq"].append(item_data["mask_seq"][:seq_len])
        batch["correct_seq"].append(item_data["correct_seq"][:seq_len])

        if "concept_seq" in seq_keys:
            batch["concept_seq"].append(item_data["concept_seq"][:seq_len])
        if "question_diff_seq" in seq_keys:
            batch["question_diff_seq"].append(item_data["question_diff_seq"][:seq_len])
        if "concept_diff_seq" in seq_keys:
            batch["concept_diff_seq"].append(item_data["concept_diff_seq"][:seq_len])

    for k in batch.keys():
        for i, seq in enumerate(batch[k]):
            batch[k][i] += [0] * (max_seq_len - len(seq))

    for k in batch.keys():
        batch[k] = torch.tensor(batch[k]).long().to(device)

    return batch


def batch_item_data2batch(batch_item_data, device):
    """
    将一个batch的item data数据转换为batch \n
    :param batch_item_data: [{seq_id, seq_len, question_seq, ...}, ...]
    :param device: cuda or cpu
    :return:
    """
    batch = {
        "question_seq": [],
        "mask_seq": [],
        "correct_seq": []
    }
    if "concept_seq" in batch_item_data[0].keys():
        batch["concept_seq"] = []
    if "question_diff_seq" in batch_item_data[0].keys():
        batch["question_diff_seq"] = []
    if "concept_diff_seq" in batch_item_data[0].keys():
        batch["concept_diff_seq"] = []

    seq_keys = list(batch.keys())
    max_seq_len = 0
    for item_data in batch_item_data:
        seq_len = item_data["seq_len"]
        max_seq_len = max(max_seq_len, seq_len)

        batch["question_seq"].append(item_data["question_seq"][:seq_len])
        batch["mask_seq"].append(item_data["mask_seq"][:seq_len])
        batch["correct_seq"].append(item_data["correct_seq"][:seq_len])

        if "concept_seq" in seq_keys:
            batch["concept_seq"].append(item_data["concept_seq"][:seq_len])
        if "question_diff_seq" in seq_keys:
            batch["question_diff_seq"].append(item_data["question_diff_seq"][:seq_len])
        if "concept_diff_seq" in seq_keys:
            batch["concept_diff_seq"].append(item_data["concept_diff_seq"][:seq_len])

    for k in batch.keys():
        for i, seq in enumerate(batch[k]):
            batch[k][i] += [0] * (max_seq_len - len(seq))

    for k in batch.keys():
        batch[k] = torch.tensor(batch[k]).long().to(device)

    return batch


def kt_data2cd_data(data_uniformed):
    data4cd = []
    for item_data in data_uniformed:
        user_data = {
            "user_id": item_data["user_id"],
            "num_interaction": item_data["seq_len"],
            "all_interaction_data": []
        }
        for i in range(item_data["seq_len"]):
            interaction_data = {
                "question_id": item_data["question_seq"][i],
                "correct": item_data["correct_seq"][i]
            }
            user_data["all_interaction_data"].append(interaction_data)
        data4cd.append(user_data)

    return data4cd


def write_cd_task_dataset(data, data_path):
    id_keys = data[0].keys()
    with open(data_path, "w") as f:
        first_line = ",".join(id_keys) + "\n"
        f.write(first_line)
        for interaction_data in data:
            line_str = ""
            for k in id_keys:
                line_str += str(interaction_data[k]) + ","
            f.write(line_str[:-1] + "\n")


def read_cd_task_dataset(data_path):
    assert os.path.exists(data_path), f"{data_path} not exist"
    with open(data_path, "r") as f:
        all_lines = f.readlines()
        first_line = all_lines[0].strip()
        id_keys_str = first_line.strip()
        id_keys = id_keys_str.split(",")
        all_lines = all_lines[1:]
        data = []
        for i, line_str in enumerate(all_lines):
            interaction_data = {}
            line_content = list(map(int, line_str.strip().split(",")))
            for id_key, v in zip(id_keys, line_content):
                interaction_data[id_key] = v
            data.append(interaction_data)

    return data


def read_mlkc_data(f_path):
    data = {}
    with open(f_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        line_ = line.split(":")
        user_id, data_value = line_[0], line_[1]
        data[int(user_id)] = list(map(float, data_value.split(",")))
    return data


def generate_factor4lbkt(data_uniformed, use_time_mean_dict, use_time_std_dict, num_attempt_mean_dict, num_hint_mean_dict,
                         use_use_time_first=True):
    max_seq_len = len(data_uniformed[0]["mask_seq"])
    # 需要考虑统计信息是从训练集提取的，在测试集和验证集中有些习题没在训练集出现过，对于这些习题，就用训练集的平均代替
    use_time_mean4unseen = sum(use_time_mean_dict.values()) / len(use_time_std_dict)
    use_time_std4unseen = sum(use_time_std_dict.values()) / len(use_time_std_dict)
    num_attempt_mean4unseen = sum(num_attempt_mean_dict.values()) / len(num_attempt_mean_dict)
    num_hint_mean4unseen = sum(num_hint_mean_dict.values()) / len(num_hint_mean_dict)
    for item_data in data_uniformed:
        time_factor_seq = []
        attempt_factor_seq = []
        hint_factor_seq = []
        seq_len = item_data["seq_len"]
        for i in range(seq_len):
            q_id = item_data["question_seq"][i]
            use_time_mean = use_time_mean_dict.get(q_id, use_time_mean4unseen)
            use_time_std = use_time_std_dict.get(q_id, use_time_std4unseen)
            num_attempt_mean = num_attempt_mean_dict.get(q_id, num_attempt_mean4unseen)
            num_hint_mean = num_hint_mean_dict.get(q_id, num_hint_mean4unseen)

            if not use_use_time_first:
                use_time_first = item_data["use_time_seq"][i]
            else:
                use_time_first = item_data["use_time_first_seq"][i]
            # 有些数据集use time first <= 0 （前端均处理为0）
            if use_time_first == 0:
                use_time_first = int(use_time_mean4unseen)
            time_factor = 1 if (use_time_std == 0) else norm(use_time_mean, use_time_std).cdf(np.log(use_time_first))
            time_factor_seq.append(time_factor)

            num_attempt = item_data["num_attempt_seq"][i]
            if num_attempt < 0:
                num_attempt = int(num_attempt_mean4unseen)
            attempt_factor = 1 - poisson(num_attempt_mean).cdf(num_attempt - 1)
            attempt_factor_seq.append(attempt_factor)

            num_hint = item_data["num_hint_seq"][i]
            if num_attempt < 0:
                num_hint = int(num_hint_mean4unseen)
            hint_factor = 1 - poisson(num_hint_mean).cdf(num_hint - 1)
            hint_factor_seq.append(hint_factor)

            if (use_time_first <= 0) or (str(time_factor) == "nan"):
                print(f"time error: {use_time_first}, {time_factor}")
            if str(attempt_factor) == "nan":
                print(f"time error: {num_attempt}, {attempt_factor}")
            if str(hint_factor) == "nan":
                print(f"time error: {num_hint}, {hint_factor}")
        item_data["time_factor_seq"] = time_factor_seq + [0.] * (max_seq_len - seq_len)
        item_data["attempt_factor_seq"] = attempt_factor_seq + [0.] * (max_seq_len - seq_len)
        item_data["hint_factor_seq"] = hint_factor_seq + [0.] * (max_seq_len - seq_len)


def generate_unique_id(input_str):
    hash_object = hashlib.md5(input_str.encode())
    return hash_object.hexdigest()


def kt_data2user_question_matrix(data, num_question, remove_last=1):
    """
    构造user-question矩阵，矩阵元素是用户对习题答对正确率，如果未作答过，则为-1
    """
    num_user = len(data)
    matrix = np.zeros((num_user, num_question))
    sum_matrix = np.zeros((num_user, num_question))
    for item_data in data:
        user_id = item_data["user_id"]
        question_seq = item_data["question_seq"][:item_data["seq_len"]-remove_last]
        correct_seq = item_data["correct_seq"][:item_data["seq_len"] - remove_last]
        for q_id, correctness in zip(question_seq, correct_seq):
            matrix[user_id][q_id] += correctness
            sum_matrix[user_id][q_id] += 1
    matrix[sum_matrix == 0] = -1
    sum_matrix[sum_matrix == 0] = 1
    return matrix / sum_matrix


def kt_data2user_concept_matrix(data, num_concept, q2c, remove_last=1):
    """
    构造user-concept矩阵，矩阵元素是用户对知识点答对正确率，如果未作答过，则为-1
    """
    num_user = len(data)
    matrix = np.zeros((num_user, num_concept))
    sum_matrix = np.zeros((num_user, num_concept))
    for item_data in data:
        user_id = item_data["user_id"]
        question_seq = item_data["question_seq"][:item_data["seq_len"]-remove_last]
        correct_seq = item_data["correct_seq"][:item_data["seq_len"] - remove_last]
        for q_id, correctness in zip(question_seq, correct_seq):
            c_ids = q2c[q_id]
            for c_id in c_ids:
                matrix[user_id][c_id] += correctness
                sum_matrix[user_id][c_id] += 1
    matrix[sum_matrix == 0] = -1
    sum_matrix[sum_matrix == 0] = 1
    return matrix / sum_matrix
