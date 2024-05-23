import torch
import torch.nn as nn
from torch.nn import Module, Parameter, Embedding, Linear, Dropout
from torch.nn.init import kaiming_normal_


class DeepIRT(Module):
    model_name = "DeepIRT"

    def __init__(self, params, objects):
        super().__init__()
        self.params = params
        self.objects = objects

        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DeepIRT"]
        num_concept = encoder_config["num_concept"]
        dim_emb = encoder_config["dim_emb"]
        size_memory = encoder_config["size_memory"]
        dropout = encoder_config["dropout"]

        self.embed_key = Embedding(num_concept, dim_emb)
        self.Mk = Parameter(torch.Tensor(size_memory, dim_emb))
        self.Mv0 = Parameter(torch.Tensor(size_memory, dim_emb))
        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)

        self.embed_value = Embedding(num_concept * 2, dim_emb)
        self.f_layer = Linear(dim_emb * 2, dim_emb)
        self.dropout_layer = Dropout(dropout)
        self.p_layer = Linear(dim_emb, 1)

        self.diff_layer = nn.Sequential(Linear(dim_emb, 1), nn.Tanh())
        self.ability_layer = nn.Sequential(Linear(dim_emb, 1), nn.Tanh())

        self.e_layer = Linear(dim_emb, dim_emb)
        self.a_layer = Linear(dim_emb, dim_emb)

    def forward(self, batch):
        num_concept = self.params["models_config"]["kt_model"]["encoder_layer"]["DeepIRT"]["num_concept"]
        concept_seq = batch["concept_seq"]
        correct_seq = batch["correct_seq"]
        batch_size = concept_seq.shape[0]

        x = concept_seq + num_concept * correct_seq
        k = self.embed_key(concept_seq)
        v = self.embed_value(x)
        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1)
        Mv = [Mvt]
        w = torch.softmax(torch.matmul(k, self.Mk.T), dim=-1)

        # Write Process
        e = torch.sigmoid(self.e_layer(v))
        a = torch.tanh(self.a_layer(v))
        for et, at, wt in zip(e.permute(1, 0, 2), a.permute(1, 0, 2), w.permute(1, 0, 2)):
            Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + (wt.unsqueeze(-1) * at.unsqueeze(1))
            Mv.append(Mvt)
        Mv = torch.stack(Mv, dim=1)

        # Read Process
        f = torch.tanh(
            self.f_layer(
                torch.cat(
                    [
                        (w.unsqueeze(-1) * Mv[:, :-1]).sum(-2),
                        k
                    ],
                    dim=-1
                )
            )
        )

        stu_ability = self.ability_layer(self.dropout_layer(f))  # equ 12
        que_diff = self.diff_layer(self.dropout_layer(k))  # equ 13

        predict_score = torch.sigmoid(3.0 * stu_ability - que_diff)  # equ 14
        predict_score = predict_score.squeeze(-1)

        return predict_score

    def get_predict_score(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = self.forward(batch)
        predict_score = torch.masked_select(predict_score[:, 1:], mask_bool_seq[:, 1:])

        return predict_score

    def get_predict_loss(self, batch, loss_record=None):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)

        predict_score = self.get_predict_score(batch)
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])
        predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())

        if loss_record is not None:
            num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
            loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)

        return predict_loss
