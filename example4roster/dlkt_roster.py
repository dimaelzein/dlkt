import argparse

import config

from lib.config.dlkt_roster_config import roster_general_config
from lib.util.load_model import load_kt_model
from lib.roster.KTRoster import KTRoster
from lib.util.parse import str2bool
from lib.util.data import read_preprocessed_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # device配置
    parser.add_argument("--debug_mode", type=str2bool, default=False)
    parser.add_argument("--use_cpu", type=str2bool, default=False)

    # 加载模型参数配置
    parser.add_argument("--save_model_dir", type=str, help="绝对路径",
                        default=r"/Users/dream/myProjects/dlkt-release/lab/saved_models/DKT@@kg4ex_setting@@assist2009_train@@seed_0@@2025-02-08@16-36-05")
    parser.add_argument("--save_model_name", type=str, help="文件名", default="saved.ckt")
    parser.add_argument("--model_name_in_ckt", type=str, help="文件名", default="best_valid")
    # 测试配置
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--data_type", type=str, default="only_question",
                        choices=("multi_concept", "single_concept", "only_question"))

    # -----------------------特殊配置：如DIMKT需要统计习题难度信息-------------------------------------
    # 如果是DIMKT，需要训练集数据的difficulty信息
    parser.add_argument("--is_dimkt", type=str2bool, default=False)
    parser.add_argument("--train_diff_file_path", type=str,
                        default=r"dir/dataset_file_dimkt_diff.json")
    parser.add_argument("--num_question_diff", type=int, default=100)
    parser.add_argument("--num_concept_diff", type=int, default=100)
    # --------------------------------------------------------------------------------------------

    args = parser.parse_args()
    params = vars(args)

    global_params, global_objects = roster_general_config(params, config.FILE_MANAGER_ROOT)

    model = load_kt_model(global_params, global_objects,
                          params["save_model_dir"], params["save_model_name"], params["model_name_in_ckt"])

    global_objects["models"] = {}
    global_objects["data_loaders"] = {}
    global_objects["models"]["kt_model"] = model

    roster = KTRoster(global_params, global_objects)
    data = read_preprocessed_file("/Users/dream/myProjects/dlkt-release/lab/settings/kg4ex_setting/assist2009_test.txt")
    batch_data = data[:4]
    last_concept_mastery_level = roster.get_last_concept_mastery_level(batch_data)
