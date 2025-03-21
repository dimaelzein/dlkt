import torch
import torch.nn as nn

from .Module.KTEmbedLayer import KTEmbedLayer
from .Module.PredictorLayer import PredictorLayer
from .util import get_mask4last_or_penultimate


class DKT(nn.Module):
    model_name = "DKT"
    use_question = False

    def __init__(self, params, objects):
        super(DKT, self).__init__()
        self.params = params
        self.objects = objects

        self.embed_layer = KTEmbedLayer(self.params, self.objects)
        data_type = self.params["datasets_config"]["data_type"]
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DKT"]
        use_concept = encoder_config["use_concept"]
        dim_emb = encoder_config["dim_emb"]

        if not use_concept:
            # 可以做细粒度评估
            self.use_question = True

        if not (use_concept and data_type != "only_question"):
            # 送入RNN的是知识点|习题emb拼接上correct emb
            dim_emb *= 2

        dim_latent = encoder_config["dim_latent"]
        rnn_type = encoder_config["rnn_type"]
        num_rnn_layer = encoder_config["num_rnn_layer"]

        if rnn_type == "rnn":
            self.encoder_layer = nn.RNN(dim_emb, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        elif rnn_type == "lstm":
            self.encoder_layer = nn.LSTM(dim_emb, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        else:
            self.encoder_layer = nn.GRU(dim_emb, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        self.predict_layer = PredictorLayer(self.params, self.objects)

    def get_concept_emb4single_concept(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DKT"]
        use_concept = encoder_config["use_concept"]
        if use_concept:
            return self.embed_layer.get_emb("concept", batch["concept_seq"])
        else:
            return self.embed_layer.get_emb("question", batch["question_seq"])

    def get_concept_emb4only_question(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DKT"]
        use_concept = encoder_config["use_concept"]
        if use_concept:
            return self.embed_layer.get_concept_fused_emb(batch["question_seq"], fusion_type="mean")
        else:
            return self.embed_layer.get_emb("question", batch["question_seq"])

    def forward(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DKT"]
        num_concept = encoder_config["num_concept"]
        use_concept = encoder_config["use_concept"]
        dim_emb = encoder_config["dim_emb"]
        data_type = self.params["datasets_config"]["data_type"]

        self.encoder_layer.flatten_parameters()
        if use_concept and data_type != "only_question":
            interaction_seq = batch["concept_seq"] + num_concept * batch["correct_seq"]
            interaction_emb = self.embed_layer.get_emb("interaction", interaction_seq)
            latent, _ = self.encoder_layer(interaction_emb)
            predict_score = self.predict_layer(latent)
        else:
            batch_size = batch["correct_seq"].shape[0]
            correct_emb = batch["correct_seq"].reshape(-1, 1).repeat(1, dim_emb).reshape(batch_size, -1, dim_emb)
            if data_type == "only_question":
                concept_emb = self.get_concept_emb4only_question(batch)
            else:
                concept_emb = self.get_concept_emb4single_concept(batch)
            interaction_emb = torch.cat((concept_emb[:, :-1], correct_emb[:, :-1]), dim=2)
            latent, _ = self.encoder_layer(interaction_emb)
            predict_layer_input = torch.cat((latent, concept_emb[:, 1:]), dim=2)
            predict_score = self.predict_layer(predict_layer_input).squeeze(dim=-1)

        return predict_score

    def get_predict_loss(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score_result = self.get_predict_score(batch)
        predict_score = predict_score_result["predict_score"]
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])
        predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())
        num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
        return {
            "total_loss": predict_loss,
            "losses_value": {
                "predict loss": {
                    "value": predict_loss.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                },
            },
            "predict_score": predict_score,
            "predict_score_batch": predict_score_result["predict_score_batch"]
        }

    def get_predict_score(self, batch):
        data_type = self.params["datasets_config"]["data_type"]
        use_concept = self.params["models_config"]["kt_model"]["encoder_layer"]["DKT"]["use_concept"]
        num_concept = self.params["models_config"]["kt_model"]["encoder_layer"]["DKT"]["num_concept"]

        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        if use_concept and data_type != "only_question":
            one_hot4predict_score = nn.functional.one_hot(batch["concept_seq"][:, 1:], num_concept)
            predict_score_batch = self.forward(batch)[:, :-1]
            predict_score_batch = (predict_score_batch * one_hot4predict_score).sum(-1)
        else:
            predict_score_batch = self.forward(batch)
        predict_score = torch.masked_select(predict_score_batch, mask_bool_seq[:, 1:])

        return {
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }

    def forward4question_evaluate(self, batch):
        # 直接输出的是每个序列最后一个时刻的预测分数
        predict_score = self.forward(batch)[:, :-1]
        # 只保留mask每行的最后一个1
        mask4last = get_mask4last_or_penultimate(batch["mask_seq"], penultimate=False)[:, 1:]
        predict_score = predict_score * mask4last
        predict_score = torch.sum(predict_score, dim=1)

        return predict_score

    def get_last_concept_mastery_level(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DKT"]
        num_concept = encoder_config["num_concept"]
        use_concept = encoder_config["use_concept"]
        dim_emb = encoder_config["dim_emb"]
        data_type = self.params["datasets_config"]["data_type"]

        self.encoder_layer.flatten_parameters()
        batch_size = batch["correct_seq"].shape[0]
        first_index = torch.arange(batch_size).long().to(self.params["device"])
        if use_concept:
            if data_type != "only_question":
                interaction_seq = batch["concept_seq"] + num_concept * batch["correct_seq"]
                interaction_emb = self.embed_layer.get_emb("interaction", interaction_seq)
                latent, _ = self.encoder_layer(interaction_emb)
                last_latent = latent[first_index, batch["seq_len"] - 2]
                last_mlkc = self.predict_layer(last_latent)
            else:
                all_concept_id = torch.arange(num_concept).long().to(self.params["device"])
                all_concept_emb = self.embed_layer.get_emb("concept", all_concept_id)
                correct_emb = batch["correct_seq"].reshape(-1, 1).repeat(1, dim_emb).reshape(batch_size, -1, dim_emb)
                concept_emb = self.get_concept_emb4only_question(batch)
                interaction_emb = torch.cat((concept_emb[:, :-1], correct_emb[:, :-1]), dim=2)
                latent, _ = self.encoder_layer(interaction_emb)
                last_latent = latent[first_index, batch["seq_len"] - 2]
                last_latent_expanded = last_latent.repeat_interleave(num_concept, dim=0).view(batch_size, num_concept, -1)
                all_concept_emb_expanded = all_concept_emb.expand(batch_size, -1, -1)
                predict_layer_input = torch.cat([last_latent_expanded, all_concept_emb_expanded], dim=-1)
                last_mlkc = self.predict_layer(predict_layer_input).squeeze(dim=-1)
        else:
            raise NotImplementedError("must use concept")

        return last_mlkc
