import platform
import numpy as np
from multiprocessing import Pool

# 多进程共享全局变量
_global_users_sets = []


def kg4ex_acc(users_mlkc, users_recommended_questions, q2c, delta1):
    """
    users_mlkc: [[mlkc_{user_0, kc_0}, ..., mlkc_{user_0, kc_j}, ...], ...]
    users_recommended_questions: [[user_0_rec_q_0, ..., user_0_rec_q_n], ...]
    """
    users_acc = []
    for user_mlkc, user_recommended_questions in zip(users_mlkc, users_recommended_questions):
        acc = 0
        for q_id in user_recommended_questions:
            c_ids = q2c[q_id]
            diff = 1.0
            for c_id in c_ids:
                diff = diff * user_mlkc[c_id]
            acc += 1 - abs(delta1 - diff)
        users_acc.append(acc / len(user_recommended_questions))
    return np.mean(users_acc)


def kg4ex_novelty(users_history_concepts, users_recommended_questions, q2c):
    """
        users_history_concepts: [{user_0_c_0, ..., user_0_c_T}, ...]
        users_recommended_questions: [[user_0_rec_q_0, ..., user_0_rec_q_n], ...]
    """
    user_novelty = []
    for user_history_concepts, user_recommended_questions in zip(users_history_concepts, users_recommended_questions):
        novelty = 0
        for q_id in user_recommended_questions:
            recommended_concepts = set(q2c[q_id])
            intersection = len(user_history_concepts.intersection(recommended_concepts))
            union = len(user_history_concepts.union(recommended_concepts))
            novelty += 1 - intersection / union
        user_novelty.append(novelty / len(user_recommended_questions))
    return np.mean(user_novelty)


def get_user_answer_correctly_concepts(question_seq, correct_seq, q2c):
    answer_correctly_concepts = []
    for q_id, correctness in zip(question_seq, correct_seq):
        if correctness == 1:
            answer_correctly_concepts.extend(q2c[q_id])
    return set(answer_correctly_concepts)


def init_pool4per_ind(users_recommended):
    """初始化进程池，预处理推荐列表为集合"""
    global _global_users_sets
    _global_users_sets = [set(rec) for rec in users_recommended]


def compute_i_similarity(i):
    """计算单个用户i与其他用户的相似度"""
    global _global_users_sets
    users_sets = _global_users_sets
    n = len(users_sets)

    if i >= n - 1:
        return 0.0, 0

    i_set = users_sets[i]
    total_sim = 0.0
    count = 0

    for j in range(i + 1, n):
        j_set = users_sets[j]
        intersection = len(i_set & j_set)
        union = len(i_set | j_set)
        jac = intersection / union if union else 0.0
        total_sim += jac
        count += 1

    return total_sim, count


def personalization_index(users_recommended_questions):
    """
    衡量推荐系统对不同用户的推荐差异程度。通过计算不同用户之间推荐内容的Jaccard相似度来评估，越低的相似度意味着系统个性化程度越高
    users_recommended_questions: [[user_0_rec_q_0, ..., user_0_rec_q_n], ...]
    """
    n = len(users_recommended_questions)
    total_sim = 0.0
    total_num = 0
    # windows多进程主程序需要放在__main__下面，否则报错
    use_multiprocessing = (n >= 5000) and (platform.system().lower() != "windows")
    if use_multiprocessing:
        with Pool(initializer=init_pool4per_ind, initargs=(users_recommended_questions,)) as pool:
            results = pool.map(compute_i_similarity, range(n - 1))
            for sim, cnt in results:
                total_sim += sim
                total_num += cnt
    else:
        users_sets = [set(rec) for rec in users_recommended_questions]
        for i in range(n - 1):
            i_set = users_sets[i]
            current_sim = 0.0
            for j in range(i + 1, n):
                j_set = users_sets[j]
                intersection = len(i_set & j_set)
                union = len(i_set | j_set)
                jac = intersection / union if union else 0.0
                current_sim += jac
            total_sim += current_sim
            total_num += n - i - 1

    return 1 - total_sim / total_num
