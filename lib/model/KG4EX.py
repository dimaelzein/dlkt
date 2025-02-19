import torch
import torch.nn as nn
import torch.nn.functional as F

from ..metric.exercise_recommendation import *


def TransE(head, relation, tail, gamma):
    return gamma - torch.norm((head + relation) - tail, p=2)


class KG4EX(nn.Module):
    model_name = "KG4EX"

    def __init__(self, params, objects):
        super(KG4EX, self).__init__()
        self.params = params
        self.objects = objects

        model_config = params["models_config"]["er_model"]["KG4EX"]
        model_selection = model_config["model_selection"]
        num_entity = model_config["num_entity"]
        num_relation = model_config["num_relation"]
        dim_hidden = model_config["dim_hidden"]
        gamma = model_config["gamma"]
        double_entity_embedding = model_config["double_entity_embedding"]
        double_relation_embedding = model_config["double_relation_embedding"]
        epsilon = model_config["epsilon"]

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + epsilon) / dim_hidden]),
            requires_grad=False
        )
        self.entity_dim = dim_hidden * 2 if double_entity_embedding else dim_hidden
        self.relation_dim = dim_hidden * 2 if double_relation_embedding else dim_hidden
        self.entity_embedding = nn.Parameter(torch.zeros(num_entity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(num_relation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        # Do not forget to modify this line when you add a new model in the "forward" function
        if model_selection not in ['TransE', 'RotatE']:
            raise ValueError(f'model {model_selection} not supported')

        if model_selection == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

    def forward(self, sample, mode='single'):
        if mode == 'single':
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        model_selection = self.params["models_config"]["er_model"]["KG4EX"]["model_selection"]
        if model_selection == "TransE":
            score = self.TransE(head, relation, tail)
        elif model_selection == "RotatE":
            score = self.RotatE(head, relation, tail, mode)
        else:
            raise ValueError(f'{model_selection} not supported')

        return score

    def TransE(self, head, relation, tail):
        return self.gamma.item() - torch.norm(head + relation - tail, p=1, dim=2)

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score

    def train_one_step(self, one_step_data):
        model_config = self.params["models_config"]["er_model"]["KG4EX"]
        negative_adversarial_sampling = model_config["negative_adversarial_sampling"]
        uni_weight = model_config["uni_weight"]
        adversarial_temperature = model_config["adversarial_temperature"]
        w_regularization_loss = self.params["loss_config"]["regularization loss"]

        self.train()
        positive_sample, negative_sample, subsampling_weight, mode = one_step_data
        negative_score = self.forward((positive_sample, negative_sample), mode=mode[0])

        if negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        positive_score = self.forward(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        if uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2
        losses_value = {
            "positive sample loss": {
                "value": positive_sample_loss.detach().cpu().item(),
                "num_sample": 1
            },
            "negative sample loss": {
                "value": negative_sample_loss.detach().cpu().item(),
                "num_sample": 1
            },
        }
        if w_regularization_loss > 0:
            # Use L3 regularization for ComplEx and DistMult
            re_loss = w_regularization_loss * (
                    self.entity_embedding.norm(p=3) ** 3 +
                    self.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + re_loss
            losses_value["regularization loss"] = {
                "value": negative_sample_loss.detach().cpu().item(),
                "num_sample": 1
            }
        return {
            "total_loss": loss,
            "losses_value": losses_value
        }

    def evaluate(self, data):
        entity2id = self.objects["data"]["entity2id"]
        relation2id = self.objects["data"]["relation2id"]
        q_table = self.objects["data"]["Q_table"]
        q2c = self.objects["data"]["question2concept"]
        num_question, num_concept = q_table.shape[0], q_table.shape[1]
        top_ns = self.params["top_ns"]
        gamma = self.params["models_config"]["er_model"]["KG4EX"]["gamma"]

        rec_embedding = self.relation_embedding[303]
        users_data = {}

        triple_data, users_history_answer = data
        for triple in triple_data:
            h, r, t = triple
            user_id = int(t[3:])
            if user_id not in users_data:
                users_data[user_id] = {
                    "mlkc": [0] * num_concept,
                    "pkc": [0] * num_concept,
                    "exfr": [0] * num_question
                }
            qc_id = int(h[2:])
            if "mlkc" in r:
                users_data[user_id]["mlkc"][qc_id] = relation2id[r]
            elif "pkc" in r:
                users_data[user_id]["pkc"][qc_id] = relation2id[r]
            else:
                users_data[user_id]["exfr"][qc_id] = relation2id[r]

        max_top_n = max(top_ns)
        users_rec_questions = {}
        for user_id, user_data in users_data.items():
            rec_questions = {top_n: [] for top_n in top_ns}
            scores = []
            s_mlkc_list = []
            s_pkc_list = []

            for i in range(num_concept):
                c_id = entity2id[f"kc{i}"]
                mlkc_id = user_data["mlkc"][i]
                pkc_id = user_data["pkc"][i]

                kc_embedding = self.entity_embedding[c_id]
                mlkc_embedding = self.relation_embedding[mlkc_id]
                pkc_embedding = self.relation_embedding[pkc_id]

                s_mlkc_list.append(kc_embedding + mlkc_embedding)
                s_pkc_list.append(kc_embedding + pkc_embedding)

            for j in range(num_question):
                q_id = entity2id[f"ex{j}"]
                exfr_id = user_data["exfr"][j]

                e = self.entity_embedding[q_id]
                fr1 = 0.0

                for s_mlkc in s_mlkc_list:
                    fr1 += TransE(s_mlkc, rec_embedding, e, gamma)

                for s_pkc in s_pkc_list:
                    fr1 += TransE(s_pkc, rec_embedding, e, gamma)

                ej_embedding = self.entity_embedding[q_id]
                efr_embedding = self.relation_embedding[exfr_id]
                s_efr = ej_embedding + efr_embedding
                fr2 = TransE(s_efr, rec_embedding, e, gamma)

                scores.append((j, (fr1 / num_concept + fr2).detach().cpu().item()))

            question_sorted = list(map(lambda x: x[0], sorted(scores, key=lambda x: x[1], reverse=True)))
            for i, rec_q_id in enumerate(question_sorted):
                if i >= max_top_n:
                    break
                for top_n in top_ns:
                    if i < top_n:
                        rec_questions[top_n].append(rec_q_id)
            users_rec_questions[user_id] = rec_questions

        performance = {top_n: {} for top_n in top_ns}
        users_mlkc = []
        users_concepts = []
        users_recommended_questions = {top_n: [] for top_n in top_ns}
        for user_id in users_rec_questions:
            users_mlkc.append(list(map(lambda x: x / 100, users_data[user_id]["mlkc"])))
            user_history_answer = users_history_answer[user_id]
            users_concepts.append(
                get_user_answer_correctly_concepts(
                    user_history_answer["question_seq"], user_history_answer["correct_seq"], q2c
                )
            )
            for top_n in top_ns:
                users_recommended_questions[top_n].append(users_rec_questions[user_id][top_n])

        for top_n in top_ns:
            performance[top_n]["KG4EX_ACC"] = kg4ex_acc(users_mlkc, users_recommended_questions[top_n], q2c, 0.7)
            performance[top_n]["KG4EX_NOV"] = kg4ex_novelty(users_concepts, users_recommended_questions[top_n], q2c)
            performance[top_n]["PER_IND"] = personalization_index(users_recommended_questions[top_n])

        return performance

