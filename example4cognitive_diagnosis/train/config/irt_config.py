from ._config import *

from lib.util.basic import *


def irt_general_config(local_params, global_params, global_objects):
    global_params["models_config"] = {
        "cd_model": {
            "IRT": {}
        }
    }

    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    num_user = local_params["num_user"]
    value_range = local_params["value_range"]
    a_range = local_params["a_range"]
    D = local_params["D"]

    # backbone
    model_config = global_params["models_config"]["cd_model"]["IRT"]
    model_config["num_concept"] = num_concept
    model_config["num_question"] = num_question
    model_config["num_user"] = num_user
    model_config["value_range"] = value_range
    model_config["a_range"] = a_range
    model_config["D"] = D

    global_objects["logger"].info(
        "model params\n    "
        f"num_user: {num_user}, num_question: {num_question}, num_concept: {num_concept}\n    "
        f"value_range: {value_range}, a_range: {a_range}, D: {D}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]
        global_params["save_model_dir_name"] = (
            f"IRT@@{setting_name}@@{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@"
            f"{get_now_time().replace(' ', '@').replace(':', '-')}"
        )


def irt_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    irt_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
