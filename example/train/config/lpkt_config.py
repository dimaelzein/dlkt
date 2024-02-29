import torch

from ._config import *
from ._data_aug_config import *

from lib.template.objects_template import OBJECTS
from lib.template.params_template_v2 import PARAMS
from lib.util.basic import *
from lib.util.parse import cal_diff
from lib.CONSTANT import INTERVAL_TIME4LPKT_PLUS, USE_TIME4LPKT_PLUS


def lpkt_general_config(local_params, global_params, global_objects):
    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    dim_e = local_params["dim_e"]
    dim_k = local_params["dim_k"]
    dim_correct = local_params["dim_correct"]
    dropout = local_params["dropout"]

    global_params["models_config"] = {
        "kt_model": {
            "type": "LPKT",
            "encoder_layer": {
                "LPKT": {}
            }
        }
    }
    encoder_layer_config = global_params["models_config"]["kt_model"]["encoder_layer"]["LPKT"]
    encoder_layer_config["num_concept"] = num_concept
    encoder_layer_config["num_question"] = num_question
    encoder_layer_config["dim_e"] = dim_e
    encoder_layer_config["dim_k"] = dim_k
    encoder_layer_config["dim_correct"] = dim_correct
    encoder_layer_config["dropout"] = dropout
    encoder_layer_config["ablation_set"] = local_params["ablation_set"]

    # q matrix
    global_objects["LPKT"] = {}
    global_objects["LPKT"]["q_matrix"] = torch.from_numpy(
        global_objects["data"]["Q_table"]
    ).float().to(global_params["device"]) + 0.03

    global_objects["logger"].info(
        "model params\n"
        f"    num of concept: {num_concept}, num of question: {num_question}, dim of e: {dim_e}, dim of k: {dim_k}, "
        f"dim of correct emb: {dim_correct}, dropout: {dropout}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"{get_now_time().replace(' ', '@').replace(':', '-')}@@LPKT@@seed_{local_params['seed']}@@{setting_name}@@"
            f"{train_file_name.replace('.txt', '')}")


def lpkt_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    lpkt_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects


def lpkt_max_entropy_adv_aug_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    lpkt_general_config(local_params, global_params, global_objects)
    max_entropy_adv_aug_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        global_params["save_model_dir_name"] = (
            global_params["save_model_dir_name"].replace("@@LPKT@@", "@@LPKT-ME-ADA@@"))
        save_params(global_params, global_objects)

    return global_params, global_objects


def lpkt_plus_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    lpkt_general_config(local_params, global_params, global_objects)

    global_params["datasets_config"]["train"]["type"] = "kt4lpkt_plus"
    global_params["datasets_config"]["train"]["lpkt_plus"] = {}
    global_params["datasets_config"]["test"]["type"] = "kt4lpkt_plus"
    global_params["datasets_config"]["test"]["kt4lpkt_plus"] = {}
    if local_params["train_strategy"] == "valid_test":
        global_params["datasets_config"]["valid"]["type"] = "kt4lpkt_plus"
        global_params["datasets_config"]["valid"]["kt4lpkt_plus"] = {}

    encoder_config = global_params["models_config"]["kt_model"]["encoder_layer"]
    global_params["models_config"]["kt_model"]["type"] = "LPKT_PLUS"
    encoder_config["LPKT_PLUS"] = deepcopy(encoder_config["LPKT"])
    del encoder_config["LPKT"]
    encoder_config["LPKT_PLUS"]["num_interval_time"] = len(INTERVAL_TIME4LPKT_PLUS)
    encoder_config["LPKT_PLUS"]["num_use_time"] = len(USE_TIME4LPKT_PLUS)
    encoder_config["LPKT_PLUS"]["ablation_set"] = local_params["ablation_set"]

    global_objects["LPKT_PLUS"] = {}
    global_objects["LPKT_PLUS"]["q_matrix"] = torch.from_numpy(
        global_objects["data"]["Q_table"]
    ).float().to(global_params["device"]) + 0.05
    q_matrix = global_objects["LPKT_PLUS"]["q_matrix"]
    q_matrix[q_matrix > 1] = 1
    del global_objects["LPKT"]

    # 统计习题难度和区分度
    dataset_train = read_preprocessed_file(os.path.join(
        global_objects["file_manager"].get_setting_dir(global_params["datasets_config"]["train"]["setting_name"]),
        global_params["datasets_config"]["train"]["file_name"]
    ))
    que_accuracy = cal_diff(dataset_train, "question_seq", local_params["min_fre4diff"])
    que_difficulty = {k: round(1 - v, 4) for k, v in que_accuracy.items()}
    global_objects["LPKT_PLUS"]["Q_table_mask"] = torch.from_numpy(
        global_objects["data"]["Q_table"]
    ).long().to(global_params["device"])
    global_objects["LPKT_PLUS"]["que_diff_ground_truth"] = torch.from_numpy(
        global_objects["data"]["Q_table"]
    ).float().to(global_params["device"])
    que_diff_ground_truth = global_objects["LPKT_PLUS"]["que_diff_ground_truth"]
    for q_id, que_diff in que_difficulty.items():
        que_diff_ground_truth[q_id] = que_diff_ground_truth[q_id] * que_diff
    global_objects["LPKT_PLUS"]["que_has_diff_ground_truth"] = torch.tensor(
        list(que_difficulty.keys())
    ).long().to(global_params["device"])

    que_discrimination = cal_que_discrimination(dataset_train, {
        "num2drop4question": local_params["min_fre4disc"],
        "min_seq_len": local_params["min_seq_len4disc"],
        "percent_threshold": local_params["percent_threshold"]
    })

    # 损失权重配置
    global_params["loss_config"]["que diff pred loss"] = local_params["w_que_diff_pred"]
    global_params["loss_config"]["que disc pred loss"] = local_params["w_que_disc_pred"]

    if local_params["save_model"]:
        global_params["save_model_dir_name"] = (
            global_params["save_model_dir_name"].replace("@@LPKT@@", "@@LPKT_PLUS@@"))
        save_params(global_params, global_objects)

    return global_params, global_objects
