from ..util.parse import question2concept_from_Q


def data_process(dataset, q_table, state_target="concept", data_type="single_concept"):
    # todo: only question data, transfer to multi concept
    num_question = q_table.shape[0]
    num_concept = q_table.shape[1]
    num_item = num_concept if state_target == "concept" else num_question
    data = {i: {
        "data": [],
        "seq_starts": [],
        "seq_ends": [],
        "seq_lens": []
    } for i in range(num_item)}
    q2c = question2concept_from_Q(q_table)
    for item_data in dataset:
        item_in_this = set()
        seq_len = item_data["seq_len"]
        correct_seq = item_data["correct_seq"]

        if state_target == "concept" and data_type == "only_question":
            question_seq = item_data["question_seq"]
            for i, q_id in enumerate(question_seq):
                c_ids = q2c[q_id]
                correctness = correct_seq[i]
                for c_id in c_ids:
                    if c_id in item_in_this:
                        data[c_id]["seq_lens"][-1] = data[c_id]["seq_lens"][-1] + 1
                    else:
                        data[c_id]["seq_starts"].append(len(data[c_id]["data"]))
                        data[c_id]["seq_lens"].append(1)
                        item_in_this.add(c_id)
                    data[c_id]["data"].append(correctness)
        else:
            item_seq = item_data["concept_seq"] if state_target == "concept" else item_data["question_seq"]
            for i in range(seq_len):
                item_idx = item_seq[i]
                if item_idx in item_in_this:
                    data[item_idx]["seq_lens"][-1] = data[item_idx]["seq_lens"][-1] + 1
                else:
                    data[item_idx]["seq_starts"].append(len(data[item_idx]["data"]))
                    data[item_idx]["seq_lens"].append(1)
                    item_in_this.add(item_idx)
                correctness = correct_seq[i]
                data[item_idx]["data"].append(correctness)

        for item_idx in data:
            if len(data[item_idx]["seq_starts"]) != len(data[item_idx]["seq_ends"]):
                data[item_idx]["seq_ends"].append(data[item_idx]["seq_starts"][-1] + data[item_idx]["seq_lens"][-1])

    return data
