import argparse

from copy import deepcopy
from torch.utils.data import DataLoader

from config.kg4ex_config import kg4ex_config

from lib.util.parse import str2bool
from lib.util.data import read_preprocessed_file, load_json
from lib.util.set_up import set_seed
from lib.dataset.KG4EXDataset import *
from lib.model.KG4EX import KG4EX
from lib.trainer.ExerciseRecommendationTrainer import ExerciseRecommendationTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据集相关
    parser.add_argument("--setting_name", type=str, default="kg4ex_setting")
    parser.add_argument("--dataset_name", type=str, default="statics2011")
    parser.add_argument("--train_file_name", type=str, default="statics2011_train_triples.txt")
    parser.add_argument("--valid_file_name", type=str, default="statics2011_valid_triples.txt")
    parser.add_argument("--test_file_name", type=str, default="statics2011_test_triples.txt")
    # 优化器相关参数选择
    parser.add_argument("--optimizer_type", type=str, default="adam", choices=("adam", "sgd"))
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--momentum", type=float, default=0.9)
    # 基于step训练
    parser.add_argument("--train_strategy", type=str, default="valid_test", choices=("valid_test", "no_test"))
    parser.add_argument("--num_step", type=int, default=30000)
    parser.add_argument("--num_step2evaluate", type=int, default=1000)
    parser.add_argument("--use_early_stop", type=str2bool, default=True)
    parser.add_argument("--num_early_stop", type=int, default=5, help="num_early_stop * num_step2evaluate")
    # 评价指标选择
    parser.add_argument("--main_metric", type=str, default="KG4EX_ACC", choices=("KG4EX_ACC", "KG4EX_VOL"),
                        help="average performance of top_ns")
    parser.add_argument("--use_multi_metrics", type=str2bool, default=False)
    parser.add_argument("--top_ns", type=str, default="[10,20]")
    parser.add_argument("--multi_metrics", type=str, default="[('KG4EX_ACC', 1), ('KG4EX_VOL', 1)]")
    # 学习率
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--enable_lr_schedule", type=str2bool, default=True)
    parser.add_argument("--lr_schedule_type", type=str, default="MultiStepLR",
                        choices=("StepLR", "MultiStepLR"))
    parser.add_argument("--lr_schedule_step", type=int, default=2000,
                        help="unit: step")
    parser.add_argument("--lr_schedule_milestones", type=str, default="[1000, 3000, 9000]",
                        help="unit: step")
    parser.add_argument("--lr_schedule_gamma", type=float, default=0.5)
    # batch size
    parser.add_argument("--train_batch_size", type=int, default=1024)
    parser.add_argument("--evaluate_batch_size", type=int, default=2048)
    # 梯度裁剪
    parser.add_argument("--enable_clip_grad", type=str2bool, default=False)
    parser.add_argument("--grad_clipped", type=float, default=10.0)
    # 负样本
    parser.add_argument("--negative_sample_size", type=int, default=256)
    # 模型参数
    parser.add_argument("--model_selection", type=str, default="TransE", choices=('TransE', 'RotatE'))
    parser.add_argument("--dim_hidden", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=12)
    parser.add_argument("--double_entity_embedding", type=str2bool, default=True)
    parser.add_argument("--double_relation_embedding", type=str2bool, default=True)
    parser.add_argument("--negative_adversarial_sampling", type=str2bool, default=True)
    parser.add_argument("--uni_weight", type=str2bool, default=True)
    parser.add_argument("--adversarial_temperature", type=float, default=1)
    parser.add_argument("--epsilon", type=float, default=2)
    # 辅助损失
    parser.add_argument("--regularization_loss", type=float, default=0)
    # 其它
    parser.add_argument("--save_model", type=str2bool, default=False)
    parser.add_argument("--use_cpu", type=str2bool, default=False)
    parser.add_argument("--debug_mode", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    params = vars(args)
    set_seed(params["seed"])
    global_params, global_objects = kg4ex_config(params)

    setting_name = params["setting_name"]
    file_manager = global_objects["file_manager"]
    setting_dir = global_objects["file_manager"].get_setting_dir(setting_name)
    entity2id = global_objects["data"]["entity2id"]
    relation2id = global_objects["data"]["relation2id"]

    train_head_params = deepcopy(global_params)
    train_head_params["datasets_config"]["dataset_this"] = "train"
    train_head_params["datasets_config"]["train"]["mode"] = "head-batch"
    train_tail_params = deepcopy(global_params)
    train_tail_params["datasets_config"]["dataset_this"] = "train"
    train_tail_params["datasets_config"]["train"]["mode"] = "tail-batch"
    dataset_head_train = KG4EXDataset(train_head_params, global_objects)
    dataset_tail_train = KG4EXDataset(train_tail_params, global_objects)
    dataloader_head_train = DataLoader(dataset_head_train, batch_size=params["train_batch_size"], shuffle=True)
    dataloader_tail_train = DataLoader(dataset_tail_train, batch_size=params["train_batch_size"], shuffle=True)
    train_iterator = BidirectionalOneShotIterator(dataloader_head_train, dataloader_tail_train)

    user_ids = load_json(os.path.join(setting_dir, f"{params['dataset_name']}_user_ids.json"))
    global_objects["data"]["user_ids"] = user_ids

    dataset_all = read_preprocessed_file(
        os.path.join(file_manager.get_setting_dir(setting_name), f"{params['dataset_name']}_train.txt")
    ) + read_preprocessed_file(
        os.path.join(file_manager.get_setting_dir(setting_name), f"{params['dataset_name']}_test.txt")
    ) + read_preprocessed_file(
        os.path.join(file_manager.get_setting_dir(setting_name), f"{params['dataset_name']}_valid.txt")
    )
    users_history_answer_all = {}
    for item_data in dataset_all:
        users_history_answer_all[item_data["user_id"]] = item_data

    valid_history_answer = {}
    for user_id in user_ids["valid"]:
        valid_history_answer[user_id] = users_history_answer_all[user_id]
    valid_triples_path = os.path.join(file_manager.get_setting_dir(setting_name), params["valid_file_name"])
    valid_data = (read_triple(valid_triples_path, entity2id, relation2id, original=True), valid_history_answer)

    if params["train_strategy"] == "valid_test":
        test_history_answer = {}
        for user_id in user_ids["test"]:
            test_history_answer[user_id] = users_history_answer_all[user_id]
        test_triples_path = os.path.join(file_manager.get_setting_dir(setting_name), params["test_file_name"])
        test_data = (read_triple(test_triples_path, entity2id, relation2id, original=True), test_history_answer)
    else:
        test_data = None

    global_objects["data_loaders"] = {}
    global_objects["data_loaders"]["train_loader"] = train_iterator
    global_objects["data_loaders"]["valid_loader"] = valid_data
    global_objects["data_loaders"]["test_loader"] = test_data

    global_params["models_config"]["er_model"]["KG4EX"]["num_entity"] = len(entity2id)
    global_params["models_config"]["er_model"]["KG4EX"]["num_relation"] = len(relation2id)
    model = KG4EX(global_params, global_objects).to(global_params["device"])
    global_objects["models"] = {}
    global_objects["models"]["er_model"] = model
    trainer = ExerciseRecommendationTrainer(global_params, global_objects)
    trainer.train()
