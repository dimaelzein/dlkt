from ._config import *

from lib.dataset.KG4EXDataset import read_id_map


def kg4ex_general_config(local_params, global_params, global_objects):
    global_params["models_config"] = {
        "er_model": {
            "type": "KG4EX",
            "KG4EX": {

            }
        }
    }

    # 配置数据参数
    negative_sample_size = local_params["negative_sample_size"]
    datasets_config = global_params["datasets_config"]
    datasets_config["train"]["negative_sample_size"] = negative_sample_size

    # 配置模型参数
    model_selection = local_params["model_selection"]
    dim_hidden = local_params["dim_hidden"]
    gamma = local_params["gamma"]
    double_entity_embedding = local_params["double_entity_embedding"]
    double_relation_embedding = local_params["double_relation_embedding"]
    negative_adversarial_sampling = local_params["negative_adversarial_sampling"]
    uni_weight = local_params["uni_weight"]
    adversarial_temperature = local_params["adversarial_temperature"]
    epsilon = local_params["epsilon"]

    model_config = global_params["models_config"]["er_model"]["KG4EX"]
    model_config["model_selection"] = model_selection
    model_config["dim_hidden"] = dim_hidden
    model_config["gamma"] = gamma
    model_config["double_entity_embedding"] = double_entity_embedding
    model_config["double_relation_embedding"] = double_relation_embedding
    model_config["negative_adversarial_sampling"] = negative_adversarial_sampling
    model_config["uni_weight"] = uni_weight
    model_config["adversarial_temperature"] = adversarial_temperature
    model_config["epsilon"] = epsilon

    # 损失权重
    global_params["loss_config"]["regularization loss"] = local_params["regularization_loss"]

    # 读取id map
    setting_name = local_params["setting_name"]
    dataset_name = local_params["dataset_name"]
    setting_dir = global_objects["file_manager"].get_setting_dir(setting_name)
    entity2id = read_id_map(os.path.join(setting_dir, f'{dataset_name}_entities.dict'))
    relation2id = read_id_map(os.path.join(setting_dir, 'relations.dict'))

    global_objects["data"]["entity2id"] = entity2id
    global_objects["data"]["relation2id"] = relation2id

    # global_objects["logger"].info(
    #     "model params\n"
    #     f"    num of user: {num_user}, num of concept: {num_concept}, num of question: {num_question}, "
    #     f"dim of predict layer 1: {dim_predict1}, dim of predict layer 2: {dim_predict2}, dropout: {dropout}"
    # )
    #
    # if local_params["save_model"]:
    #     setting_name = local_params["setting_name"]
    #     train_file_name = local_params["train_file_name"]
    #     global_params["save_model_dir_name"] = (
    #         f"KG4EX@@{setting_name}@@{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@"
    #         f"{get_now_time().replace(' ', '@').replace(':', '-')}"
    #     )


def kg4ex_config(local_params):
    global_params = {}
    global_objects = {}
    local_params["train_based_epoch"] = False
    general_config(local_params, global_params, global_objects)
    kg4ex_general_config(local_params, global_params, global_objects)
    # if local_params["save_model"]:
    #     save_params(global_params, global_objects)

    return global_params, global_objects
