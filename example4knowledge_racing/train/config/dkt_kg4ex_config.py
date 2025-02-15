from ._config import *

from lib.util.basic import *


def dkt_kg4ex_general_config(local_params, global_params, global_objects):
    global_params["models_config"] = {
        "kt_model": {
            "kt_embed_layer": {},
            "encoder_layer": {
                "type": "DKT_KG4EX",
                "DKT_KG4EX": {}
            },
            "predict_layer": {
                "type": "direct",
                "direct": {}
            }
        }
    }
    data_type = global_params["datasets_config"]["data_type"]

    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    dim_emb = local_params["dim_emb"]
    dim_latent = local_params["dim_latent"]
    rnn_type = local_params["rnn_type"]
    num_rnn_layer = local_params["num_rnn_layer"]
    dropout = local_params["dropout"]
    num_predict_layer = local_params["num_predict_layer"]
    dim_predict_mid = local_params["dim_predict_mid"]
    activate_type = local_params["activate_type"]

    # embed layer
    kt_embed_layer_config = global_params["models_config"]["kt_model"]["kt_embed_layer"]
    if data_type != "only_question":
        kt_embed_layer_config["concept"] = [num_concept, dim_emb]
    else:
        kt_embed_layer_config["question"] = [num_question, dim_emb]
        kt_embed_layer_config["concept"] = [num_concept, dim_emb]

    # encoder layer
    encoder_config = global_params["models_config"]["kt_model"]["encoder_layer"]["DKT_KG4EX"]
    encoder_config["num_concept"] = num_concept
    encoder_config["num_question"] = num_question
    encoder_config["dim_emb"] = dim_emb
    encoder_config["dim_latent"] = dim_latent
    encoder_config["rnn_type"] = rnn_type
    encoder_config["num_rnn_layer"] = num_rnn_layer

    # predict layer
    predict_layer_config = global_params["models_config"]["kt_model"]["predict_layer"]
    predict_layer_config["type"] = "direct"
    predict_layer_config["direct"]["dropout"] = dropout
    predict_layer_config["direct"]["num_predict_layer"] = num_predict_layer
    predict_layer_config["direct"]["dim_predict_mid"] = dim_predict_mid
    predict_layer_config["direct"]["activate_type"] = activate_type
    # 当使用知识点输入，并且data type为multi 或者 single concept时，预测层输出为num concept，即每个知识点上的状态
    # 其它情况都是RNN输出的latent拼接上对应知识点（1、多个知识点embedding平均；2、对于像assist2015数据集，只有习题，则是习题emb）
    # 然后送入预测层，直接输出score
    if data_type != "only_question":
        predict_layer_config["direct"]["dim_predict_in"] = dim_latent
        predict_layer_config["direct"]["dim_predict_out"] = num_concept
    else:
        predict_layer_config["direct"]["dim_predict_in"] = dim_latent + dim_emb
        predict_layer_config["direct"]["dim_predict_out"] = 1

    # 损失权重
    weight_aux_loss = local_params["weight_aux_loss"]
    global_params["loss_config"]["aux loss"] = weight_aux_loss

    global_objects["logger"].info(
        "model params\n    "
        f"num_concept: {num_concept}, num_question: {num_question}, \n    "
        f"dim_emb: {dim_emb}, dim_latent: {dim_latent}, rnn_type: {rnn_type}, num_rnn_layer: {num_rnn_layer}\n    "
        f"dropout: {dropout}, num_predict_layer: {num_predict_layer}, dim_predict_mid: {dim_predict_mid}, "
        f"activate_type: {activate_type}, weight_aux_loss: {weight_aux_loss}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"DKT_KG4EX@@{setting_name}@@{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@"
            f"{get_now_time().replace(' ', '@').replace(':', '-')}")


def dkt_kg4ex_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    dkt_kg4ex_general_config(local_params, global_params, global_objects)

    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
