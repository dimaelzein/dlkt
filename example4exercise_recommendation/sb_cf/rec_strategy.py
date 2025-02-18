def rec_method_0(train_users_data, users_history, similar_users, question_diff, th, top_n):
    users_answered_ques = {}
    for item_data in train_users_data:
        users_answered_ques[item_data["user_id"]] = set(item_data["question_seq"][:item_data["seq_len"]])

    rec_ques = {x["user_id"]: [] for x in users_history}
    for item_data in users_history:
        user_id = item_data["user_id"]
        seq_len = item_data["seq_len"]
        question_seq = item_data["question_seq"][:seq_len]
        correct_seq = item_data["correct_seq"][:seq_len]
        answered_ques = set(question_seq)
        average_diff = 1 - sum(correct_seq) / seq_len
        while len(rec_ques[user_id]) < top_n:
            # 如果阈值过小，可能不能满足推荐top n个习题，加大阈值
            th += 0.05
            for sim_user_id in similar_users[user_id]:
                for q_id in (users_answered_ques[sim_user_id] - answered_ques):
                    q_diff = question_diff[q_id]
                    if abs(average_diff - q_diff) < 0.1:
                        rec_ques[user_id].append(q_id)
                    if len(rec_ques[user_id]) >= top_n:
                        break
                if len(rec_ques[user_id]) >= top_n:
                    break

    return rec_ques
