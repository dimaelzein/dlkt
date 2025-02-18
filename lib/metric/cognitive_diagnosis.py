def ncd_doa(users_mlkc, users_history_qa, c2q):
    """
        users_history_qa: [[(user_0_q_0, correctness_0_0), (user_0_q_1, correctness_0_1), ...], ...]
    """
    num_user = len(users_mlkc)
    num_concept = len(c2q)

    users_score4question = []
    for user_history_qa in users_history_qa:
        user_score4question = {}
        for q_id, correctness in user_history_qa:
            if q_id in users_score4question:
                user_score4question[q_id].append(correctness)
            else:
                user_score4question[q_id] = [correctness]
        for q_id, q_correctness in user_score4question.items():
            user_score4question[q_id] = sum(q_correctness) / len(q_correctness)
        users_score4question.append(user_score4question)

    qs_both_answered_all = [[list() for _ in range(num_user)] for _ in range(num_user)]
    for i, ui_history_qa in enumerate(users_history_qa):
        for j, uj_history_qa in enumerate(users_history_qa):
            if j <= i:
                continue
            ui_history_qs = set(map(lambda x: x[0], ui_history_qa))
            uj_history_qs = set(map(lambda x: x[0], uj_history_qa))
            uij_both_answered_qs = ui_history_qs.intersection(uj_history_qs)
            uij_both_answered_qa = []
            for q_id in uij_both_answered_qs:
                uij_both_answered_qa.append((q_id, users_score4question[i][q_id], users_score4question[j][q_id]))
            qs_both_answered_all[i][j] = uij_both_answered_qa

    doa = 0
    for c_id in range(num_concept):
        Z = 0
        J = 0
        for i, user1_mlkc in enumerate(users_mlkc):
            for j, user2_mlkc in enumerate(users_mlkc):
                if i == j:
                    continue

                if user1_mlkc[c_id] > user2_mlkc[c_id]:
                    Z += 1
                    JS = []
                    q_ids = c2q[c_id]
                    qa_both_answered = qs_both_answered_all[min(i, j)][max(i, j)]
                    target_qa = []
                    for q_id, si, sj in qa_both_answered:
                        if q_id in q_ids:
                            target_qa.append((si, sj))

                    for si, sj in target_qa:
                        JS.append(int(si > sj))
                    J += (sum(JS) / len(JS)) if (len(JS) > 0) else 0
        doa += J / Z

    return doa / num_concept

