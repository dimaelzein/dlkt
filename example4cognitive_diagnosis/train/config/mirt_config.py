from ._config import *

from lib.util.basic import *


def mirt_general_config(local_params, global_params, global_objects):
    global_params["models_config"] = {
        "cd_model": {
            "MIRT": {}
        }
    }

    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    num_user = local_params["num_user"]
    a_range = local_params["a_range"]

    # backbone
    model_config = global_params["models_config"]["cd_model"]["MIRT"]
    model_config["num_concept"] = num_concept
    model_config["num_question"] = num_question
    model_config["num_user"] = num_user
    model_config["a_range"] = a_range

    global_objects["logger"].info(
        "model params\n    "
        f"num_user: {num_user}, num_question: {num_question}, num_concept: {num_concept}\n    "
        f"a_range: {a_range}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]
        global_params["save_model_dir_name"] = (
            f"MIRT@@{setting_name}@@{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@"
            f"{get_now_time().replace(' ', '@').replace(':', '-')}"
        )


def mirt_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    mirt_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
