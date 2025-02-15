import torch
import torch.nn as nn

from .Module.KTEmbedLayer import KTEmbedLayer
from .Module.PredictorLayer import PredictorLayer


class DKT_KG4EX(nn.Module):
    model_name = "DKT_KG4EX"
    use_question = False

    def __init__(self, params, objects):
        super(DKT_KG4EX, self).__init__()
        self.params = params
        self.objects = objects

        self.embed_layer = KTEmbedLayer(self.params, self.objects)
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DKT_KG4EX"]
        dim_emb = encoder_config["dim_emb"]
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

    def forward(self, batch):
        data_type = self.params["datasets_config"]["data_type"]
        self.encoder_layer.flatten_parameters()
        if data_type == "only_question":
            concept_emb = self.embed_layer.get_concept_fused_emb(batch["question_seq"], fusion_type="mean")
            latent, _ = self.encoder_layer(concept_emb[:, :-1])
            predict_layer_input = torch.cat((latent, concept_emb[:, 1:]), dim=2)
            predict_score = self.predict_layer(predict_layer_input).squeeze(dim=-1)
        else:
            concept_emb = self.embed_layer.get_emb("concept", batch["concept_seq"])
            latent, _ = self.encoder_layer(concept_emb)
            predict_score = self.predict_layer(latent)

        return predict_score

    def get_predict_score(self, batch):
        data_type = self.params["datasets_config"]["data_type"]
        num_concept = self.params["models_config"]["kt_model"]["encoder_layer"]["DKT_KG4EX"]["num_concept"]
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        if data_type != "only_question":
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

    def get_aux_loss(self, batch):
        """防止模型全输出1（原论文代码提供的案例中单个学生最后时刻的pkc之和不为1，所以不是概率分布，不采用多分类损失，并且论文中也不是用的多分类损失）"""
        data_type = self.params["datasets_config"]["data_type"]
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DKT_KG4EX"]
        num_concept = encoder_config["num_concept"]
        dim_latent = encoder_config["dim_latent"]
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        if data_type != "only_question":
            predict_score_batch = self.forward(batch)[:, :-1]
        else:
            batch_size = batch["correct_seq"].shape[0]
            seq_len = batch["correct_seq"].shape[1]
            all_concept_id = torch.arange(num_concept).long().to(self.params["device"])
            all_concept_emb = self.embed_layer.get_emb("concept", all_concept_id)
            concept_emb = self.embed_layer.get_concept_fused_emb(batch["question_seq"], fusion_type="mean")
            latent, _ = self.encoder_layer(concept_emb[:, :-1])
            latent_expanded = latent.repeat_interleave(num_concept, dim=1).view(batch_size, -1, num_concept, dim_latent)
            all_concept_emb_expanded = all_concept_emb.expand(batch_size, seq_len-1, -1, -1)
            predict_layer_input = torch.cat([latent_expanded, all_concept_emb_expanded], dim=-1)
            predict_score_batch = self.predict_layer(predict_layer_input).squeeze(dim=-1)
        mask_expanded = mask_bool_seq[:, 1:].unsqueeze(-1).repeat_interleave(num_concept, dim=-1)
        predict_score = predict_score_batch[mask_expanded.bool()]
        return torch.mean(predict_score)

    def get_predict_loss(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score_result = self.get_predict_score(batch)
        predict_score = predict_score_result["predict_score"]
        batch_size, seq_len = batch["correct_seq"].shape[0], batch["correct_seq"].shape[1]
        kg4ex_pkc_ground_truth = torch.ones((batch_size, seq_len - 1)).to(self.params["device"])
        ground_truth = torch.masked_select(kg4ex_pkc_ground_truth, mask_bool_seq[:, 1:])
        predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())
        aux_loss = self.get_aux_loss(batch)
        loss = predict_loss + self.params["loss_config"]["aux loss"] * aux_loss

        num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
        return {
            "total_loss": loss,
            "losses_value": {
                "predict loss": {
                    "value": predict_loss.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                },
                "aux loss": {
                    "value": aux_loss.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                }
            },
            "predict_score": predict_score,
            "predict_score_batch": predict_score_result["predict_score_batch"]
        }

    def get_last_concept_mastery_level(self, batch):
        """等价于求pkc"""
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DKT_KG4EX"]
        num_concept = encoder_config["num_concept"]
        data_type = self.params["datasets_config"]["data_type"]

        self.encoder_layer.flatten_parameters()
        batch_size = batch["correct_seq"].shape[0]
        first_index = torch.arange(batch_size).long().to(self.params["device"])
        if data_type != "only_question":
            concept_emb = self.embed_layer.get_concept_fused_emb(batch["question_seq"], fusion_type="mean")
            latent, _ = self.encoder_layer(concept_emb)
            last_latent = latent[first_index, batch["seq_len"] - 2]
            last_pkc = self.predict_layer(last_latent)
        else:
            all_concept_id = torch.arange(num_concept).long().to(self.params["device"])
            all_concept_emb = self.embed_layer.get_emb("concept", all_concept_id)
            concept_emb = self.embed_layer.get_concept_fused_emb(batch["question_seq"], fusion_type="mean")
            latent, _ = self.encoder_layer(concept_emb)
            last_latent = latent[first_index, batch["seq_len"] - 2]
            last_latent_expanded = last_latent.repeat_interleave(num_concept, dim=0).view(batch_size, num_concept, -1)
            all_concept_emb_expanded = all_concept_emb.expand(batch_size, -1, -1)
            predict_layer_input = torch.cat([last_latent_expanded, all_concept_emb_expanded], dim=-1)
            last_pkc = self.predict_layer(predict_layer_input).squeeze(dim=-1)

        return last_pkc
