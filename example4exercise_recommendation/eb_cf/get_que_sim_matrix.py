import argparse

import config
from load_data import users_history

from lib.util.parse import cal_diff, tf_idf_from_Q, cosine_similarity_matrix, pearson_similarity
from lib.util.data import kt_data2user_question_matrix
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
    num_question, num_concept = Q_table.shape[0], Q_table.shape[1]

    question_acc = cal_diff(users_history, "question_seq", 0)
    average_acc = sum(question_acc.values()) / len(question_acc)
    question_diff = {}
    for q_id in range(num_question):
        if q_id not in question_acc:
            question_diff[q_id] = average_acc
        else:
            question_diff[q_id] = 1 - question_acc[q_id]
    difficulty_dissimilarity = np.zeros((num_question, num_question))
    for i in range(num_question):
        for j in range(num_question):
            difficulty_dissimilarity[i][j] = abs(question_diff[i] - question_diff[j])

    tf_idf = tf_idf_from_Q(Q_table)
    concept_similarity = cosine_similarity_matrix(Q_table, axis=1)

    user_question_matrix = kt_data2user_question_matrix(users_history, num_question, 0)
    if params["similarity"] == "cossim":
        similarity = cosine_similarity_matrix(user_question_matrix, axis=0)
    elif params["similarity"] == "pearson_corr":
        similarity = np.zeros((num_question, num_question))
        for i in range(num_question):
            for j in range(num_question):
                si = user_question_matrix[:, i]
                sj = user_question_matrix[:, j]
                similarity[i][j] = pearson_similarity(si, sj)
    else:
        raise NotImplementedError(f'{params["similarity"]} is not implemented')

    alpha = params["alpha"]
    beta = params["beta"]
    gamma = params["gamma"]
    que_sim_matrix = alpha * similarity + beta * concept_similarity - gamma * difficulty_dissimilarity

    save_path = os.path.join(setting_dir,
                             f"que_smi_mta_{params['dataset_name']}_{params['similarity']}_{alpha}_{beta}_{gamma}.npy")
    np.save(save_path, que_sim_matrix)
