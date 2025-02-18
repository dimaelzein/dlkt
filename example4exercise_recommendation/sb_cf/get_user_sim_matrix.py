import argparse

import config
from load_data import users_history

from lib.util.parse import cal_diff, cosine_similarity_matrix, pearson_similarity, question2concept_from_Q
from lib.util.data import kt_data2user_question_matrix, kt_data2user_concept_matrix
from lib.util.set_up import set_seed
from lib.dataset.KG4EXDataset import *
from lib.util.FileManager import FileManager


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据集相关
    parser.add_argument("--setting_name", type=str, default="kg4ex_setting")
    parser.add_argument("--dataset_name", type=str, default="statics2011")
    # 构建习题相似度矩阵的方法
    parser.add_argument("--similarity", type=str, default="pearson_corr", choices=("cossim", "pearson_corr"))
    parser.add_argument("--alpha", type=float, default=1,
                        help="相似度（余弦相似度或者皮尔逊相关系数）的权重")
    parser.add_argument("--beta", type=float, default=0.25,
                        help="习题知识点相似度（tf-idf）的权重")
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="难度相似度的权重")
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
        data_type = "single_concept"
    else:
        data_type = "only_question"
    num_question, num_concept = Q_table.shape[0], Q_table.shape[1]
    q2c = question2concept_from_Q(Q_table)
    num_user = len(users_history)

    question_acc = cal_diff(users_history, "question_seq", 0)
    average_que_acc = sum(question_acc.values()) / len(question_acc)
    question_diff = {}
    for q_id in range(num_question):
        if q_id not in question_acc:
            question_diff[q_id] = average_que_acc
        else:
            question_diff[q_id] = 1 - question_acc[q_id]

    user_average_diff = {}
    for item_data in users_history:
        user_id = item_data["user_id"]
        question_seq = item_data["question_seq"][:item_data["seq_len"]]
        diff_sum = 0
        for q_id in question_seq:
            diff_sum += question_diff[q_id]
        user_average_diff[user_id] = diff_sum / item_data["seq_len"]

    user_diff_similarity = np.zeros((num_user, num_user))
    for i in range(num_user):
        for j in range(num_user):
            user_diff_similarity[i][j] = 1 - abs(user_average_diff[i] - user_average_diff[j])

    user_question_matrix = kt_data2user_question_matrix(users_history, num_question, 0)
    user_concept_matrix = kt_data2user_concept_matrix(users_history, num_concept, q2c, 0)
    if params["similarity"] == "cossim":
        user_que_similarity = cosine_similarity_matrix(user_question_matrix, axis=1)
        user_concept_similarity = cosine_similarity_matrix(user_concept_matrix, axis=1)
    elif params["similarity"] == "pearson_corr":
        user_que_similarity = np.zeros((num_user, num_user))
        user_concept_similarity = np.zeros((num_user, num_user))
        for i in range(num_user):
            for j in range(num_user):
                si = user_question_matrix[i, :]
                sj = user_question_matrix[j, :]
                user_que_similarity[i][j] = pearson_similarity(si, sj)
        for i in range(num_user):
            for j in range(num_user):
                si = user_concept_matrix[i, :]
                sj = user_concept_matrix[j, :]
                user_concept_similarity[i][j] = pearson_similarity(si, sj)
    else:
        raise NotImplementedError(f'{params["similarity"]} is not implemented')

    alpha = params["alpha"]
    beta = params["beta"]
    gamma = params["gamma"]
    que_sim_matrix = alpha * user_que_similarity + beta * user_concept_similarity + gamma * user_diff_similarity

    save_path = os.path.join(setting_dir,
                             f"user_smi_mat_{params['dataset_name']}_{params['similarity']}_{alpha}_{beta}_{gamma}.npy")
    np.save(save_path, que_sim_matrix)
