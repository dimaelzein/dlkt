import numpy as np


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
            acc += 1 - np.abs(delta1 - diff)
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

