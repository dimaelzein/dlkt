from ._config import *

from lib.util.basic import *


def dina_general_config(local_params, global_params, global_objects):
    global_params["models_config"] = {
        "cd_model": {
            "DINA": {}
        }
    }

    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    num_user = local_params["num_user"]
    max_slip = local_params["max_slip"]
    max_guess = local_params["max_guess"]
    max_step = local_params["max_step"]
    use_ste = local_params["use_ste"]

    # backbone
    model_config = global_params["models_config"]["cd_model"]["DINA"]
    model_config["num_concept"] = num_concept
    model_config["num_question"] = num_question
    model_config["num_user"] = num_user
    model_config["max_slip"] = max_slip
    model_config["max_guess"] = max_guess
    model_config["max_step"] = max_step
    model_config["use_ste"] = use_ste

    global_objects["logger"].info(
        "model params\n    "
        f"num_user: {num_user}, num_question: {num_question}, num_concept: {num_concept}\n    "
        f"max_slip: {max_slip}, max_guess: {max_guess}, max_step: {max_step}, use_ste: {use_ste}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]
        global_params["save_model_dir_name"] = (
            f"DINA@@{setting_name}@@{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@"
            f"{get_now_time().replace(' ', '@').replace(':', '-')}"
        )


def dina_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    dina_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
