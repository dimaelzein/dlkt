import argparse
from copy import deepcopy
from torch.utils.data import DataLoader

from config.simple_kt_config import simple_kt_config

from lib.util.parse import str2bool
from lib.util.set_up import set_seed
from lib.dataset.KTDataset import KTDataset
from lib.model.SimpleKT import SimpleKT
from lib.trainer.KnowledgeTracingTrainer import KnowledgeTracingTrainer


# 必须同时有question和concept id
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据集相关
    parser.add_argument("--setting_name", type=str, default="our_setting")
    parser.add_argument("--dataset_name", type=str, default="statics2011")
    parser.add_argument("--data_type", type=str, default="single_concept",
                        choices=("multi_concept", "single_concept", "only_question"))
    parser.add_argument("--train_file_name", type=str, default="statics2011_train_fold_0.txt")
    parser.add_argument("--valid_file_name", type=str, default="statics2011_valid_fold_0.txt")
    parser.add_argument("--test_file_name", type=str, default="statics2011_test_fold_0.txt")
    # 优化器相关参数选择
    parser.add_argument("--optimizer_type", type=str, default="adam", choices=("adam", "sgd"))
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--momentum", type=float, default=0.9)
    # 训练策略
    parser.add_argument("--train_strategy", type=str, default="valid_test",
                        choices=("valid_test", "no_test"))
    parser.add_argument("--num_epoch", type=int, default=200)
    parser.add_argument("--use_early_stop", type=str2bool, default=True)
    parser.add_argument("--epoch_early_stop", type=int, default=10)
    parser.add_argument("--use_last_average", type=str2bool, default=False)
    parser.add_argument("--epoch_last_average", type=int, default=5)
    # 评价指标选择
    parser.add_argument("--main_metric", type=str, default="AUC")
    parser.add_argument("--use_multi_metrics", type=str2bool, default=False)
    parser.add_argument("--multi_metrics", type=str, default="[('AUC', 1), ('ACC', 1)]")
    # 学习率
    parser.add_argument("--learning_rate", type=float, default=0.002)
    parser.add_argument("--enable_lr_schedule", type=str2bool, default=True)
    parser.add_argument("--lr_schedule_type", type=str, default="MultiStepLR",
                        choices=("StepLR", "MultiStepLR"))
    parser.add_argument("--lr_schedule_step", type=int, default=10)
    parser.add_argument("--lr_schedule_milestones", type=str, default="[5]")
    parser.add_argument("--lr_schedule_gamma", type=float, default=0.5)
    # batch size
    parser.add_argument("--train_batch_size", type=int, default=24)
    parser.add_argument("--evaluate_batch_size", type=int, default=256)
    # 梯度裁剪
    parser.add_argument("--enable_clip_grad", type=str2bool, default=False)
    parser.add_argument("--grad_clipped", type=float, default=10.0)
    # 模型参数
    parser.add_argument("--dim_model", type=int, default=64)
    parser.add_argument("--num_block", type=int, default=2)
    parser.add_argument("--num_head", type=int, default=4)
    parser.add_argument("--dim_ff", type=int, default=64)
    parser.add_argument("--dim_final_fc", type=int, default=64)
    parser.add_argument("--dim_final_fc2", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=200)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--key_query_same", type=str2bool, default=True)
    parser.add_argument("--separate_qa", type=str2bool, default=False)
    parser.add_argument("--difficulty_scalar", type=str2bool, default=False)
    # 其它
    parser.add_argument("--save_model", type=str2bool, default=False)
    parser.add_argument("--debug_mode", type=str2bool, default=False)
    parser.add_argument("--trace_epoch", type=str2bool, default=False)
    parser.add_argument("--use_cpu", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    params = vars(args)
    set_seed(params["seed"])
    global_params, global_objects = simple_kt_config(params)

    valid_params = deepcopy(global_params)
    valid_params["datasets_config"]["dataset_this"] = "valid"
    dataset_valid = KTDataset(valid_params, global_objects)
    dataloader_valid = DataLoader(dataset_valid, batch_size=params["evaluate_batch_size"], shuffle=False)

    train_params = deepcopy(global_params)
    train_params["datasets_config"]["dataset_this"] = "train"
    dataset_train = KTDataset(train_params, global_objects)
    dataloader_train = DataLoader(dataset_train, batch_size=params["train_batch_size"], shuffle=True)

    if params["train_strategy"] == "valid_test":
        test_params = deepcopy(global_params)
        test_params["datasets_config"]["dataset_this"] = "test"
        dataset_test = KTDataset(test_params, global_objects)
        dataloader_test = DataLoader(dataset_test, batch_size=params["evaluate_batch_size"], shuffle=False)
    else:
        dataloader_test = None

    global_objects["data_loaders"] = {}
    global_objects["data_loaders"]["train_loader"] = dataloader_train
    global_objects["data_loaders"]["valid_loader"] = dataloader_valid
    global_objects["data_loaders"]["test_loader"] = dataloader_test

    global_objects["models"] = {}
    model = SimpleKT(global_params, global_objects).to(global_params["device"])
    global_objects["models"]["kt_model"] = model
    trainer = KnowledgeTracingTrainer(global_params, global_objects)
    trainer.train()
