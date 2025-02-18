import argparse
import json

import config
from load_data import user_ids, users_history

from rec_strategy import *

from lib.util.data import read_mlkc_data
from lib.util.parse import question2concept_from_Q, cal_diff
from lib.util.set_up import set_seed
from lib.dataset.KG4EXDataset import *
from lib.util.FileManager import FileManager
from lib.metric.exercise_recommendation import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据集相关
    parser.add_argument("--setting_name", type=str, default="kg4ex_setting")
    parser.add_argument("--dataset_name", type=str, default="statics2011")
    parser.add_argument("--que_sim_mat_file_name", type=str, default="user_smi_mat_statics2011_pearson_corr_1_0.25_0.5.npy")
    parser.add_argument("--target", type=str, default="test", help="must in user_ids")
    # 评价指标选择
    parser.add_argument("--used_metrics", type=str, default="['KG4EX_ACC', 'KG4EX_NOV']",
                        help='KG4EX_ACC, KG4EX_VOL')
    parser.add_argument("--top_ns", type=str, default="[10,20]")
    # KG4EX_ACC指标需要的数据
    parser.add_argument("--mlkc_file_name", type=str, default="statics2011_mlkc_test.txt")
    parser.add_argument("--delta", type=float, default=0.7)
    # 推荐策略
    parser.add_argument("--rec_strategy", type=int, default=0,
                        help="0: 从相似用户的练习习题中找出用户未练习过的习题，并根据用户的难度偏好（如历史平均难度 ± 阈值）筛选习题")
    parser.add_argument("--method0_diff_threshold", type=float, default=0.1,
                        help="method 0过滤习题时的难度阈值")
    # 其它
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    params = vars(args)
    set_seed(params["seed"])

    setting_name = params["setting_name"]
    file_manager = FileManager(config.FILE_MANAGER_ROOT)
    setting_dir = file_manager.get_setting_dir(setting_name)
    Q_table = file_manager.get_q_table(params["dataset_name"], "only_question")
    if Q_table is None:
        Q_table = file_manager.get_q_table(params["dataset_name"], "single_concept")
    question2concept = question2concept_from_Q(Q_table)
    num_question, num_concept = Q_table.shape[0], Q_table.shape[1]

    question_acc = cal_diff(users_history, "question_seq", 0)
    average_que_acc = sum(question_acc.values()) / len(question_acc)
    question_diff = {}
    for q_id in range(num_question):
        if q_id not in question_acc:
            question_diff[q_id] = average_que_acc
        else:
            question_diff[q_id] = 1 - question_acc[q_id]

    users_data = list(filter(lambda x: x["user_id"] in user_ids[params["target"]], users_history))
    que_sim_mat = np.load(os.path.join(setting_dir, params["que_sim_mat_file_name"]))
    # 理论上只能使用训练集的user
    train_users_data = list(filter(lambda x: x["user_id"] in user_ids["train"], users_history))
    que_sim_mat[:, user_ids["train"]] += 100
    for i in range(len(users_data)):
        que_sim_mat[i, i] += 1000
    similar_users = np.argsort(-que_sim_mat, axis=1)[:, 1:len(train_users_data)]

    rec_strategy = params["rec_strategy"]
    top_ns = eval(params["top_ns"])
    rec_result = {x: {} for x in top_ns}
    if rec_strategy == 0:
        top_ns = sorted(top_ns, reverse=True)
        last_top_n = top_ns[0]
        for i, top_n in enumerate(top_ns):
            if i == 0:
                rec_result[top_n] = rec_method_0(
                    train_users_data, users_data, similar_users, question_diff, params["method0_diff_threshold"], top_n)
            else:
                rec_result[top_n] = {
                    user_id: rec_ques[:top_n] for user_id, rec_ques in rec_result[last_top_n].items()}
                last_top_n = top_n
    else:
        raise ValueError(f"{rec_strategy} is not implemented")

    used_metrics = eval(params["used_metrics"])
    performance = {x: {} for x in top_ns}
    for metric in used_metrics:
        for top_n in top_ns:
            rec_ques = rec_result[top_n]
            if metric == "KG4EX_ACC":
                mlkc = read_mlkc_data(os.path.join(setting_dir, params["mlkc_file_name"]))
                rec_ques_ = []
                mlkc_ = []
                for user_id in rec_ques.keys():
                    rec_ques_.append(rec_ques[user_id])
                    mlkc_.append(mlkc[user_id])

                performance[top_n][metric] = kg4ex_acc(mlkc_, rec_ques_, question2concept, params["delta"])
            if metric == "KG4EX_NOV":
                correct_cs = {}
                for item_data in users_data:
                    user_id = item_data["user_id"]
                    seq_len = item_data["seq_len"]
                    question_seq = item_data["question_seq"][:seq_len]
                    correct_seq = item_data["correct_seq"][:seq_len]
                    correct_cs[user_id] = get_user_answer_correctly_concepts(question_seq, correct_seq, question2concept)

                rec_ques_ = []
                correct_cs_ = []
                for user_id in rec_ques.keys():
                    rec_ques_.append(rec_ques[user_id])
                    correct_cs_.append(correct_cs[user_id])

                performance[top_n][metric] = kg4ex_novelty(correct_cs_, rec_ques_, question2concept)

    top_ns = sorted(top_ns)
    for top_n in top_ns:
        print(f"top {top_n}, {json.dumps(performance[top_n])}")
