import torch
import torch.nn as nn


class DCT(nn.Module):
    model_name = "DCT"

    def __init__(self, params, objects):
        super().__init__()
        self.params = params
        self.objects = objects

        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DCT"]
        num_question = encoder_config["num_question"]
        num_concept = encoder_config["num_concept"]
        dim_question = encoder_config["dim_question"]
        dim_correct = encoder_config["dim_correct"]
        dim_latent = encoder_config["dim_latent"]
        rnn_type = encoder_config["rnn_type"]
        num_rnn_layer = encoder_config["num_rnn_layer"]
        dropout = encoder_config["dropout"]

        self.embed_question = nn.Embedding(num_question, dim_question)
        if rnn_type == "rnn":
            self.encoder_layer = nn.RNN(dim_question + dim_correct, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        elif rnn_type == "lstm":
            self.encoder_layer = nn.LSTM(dim_question + dim_correct, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        else:
            self.encoder_layer = nn.GRU(dim_question + dim_correct, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        self.proj_latent2ability = nn.Linear(dim_latent, num_concept)
        self.proj_que2difficulty = nn.Linear(dim_question, num_concept)
        self.proj_que2discrimination = nn.Linear(dim_question, 1)
        self.dropout = nn.Dropout(dropout)

        self.init_weight()

    def init_weight(self):
        torch.nn.init.xavier_uniform_(self.proj_latent2ability.weight)
        torch.nn.init.xavier_uniform_(self.proj_que2difficulty.weight)
        torch.nn.init.xavier_uniform_(self.proj_que2discrimination.weight)

    def predict_score(self, latent, question_emb, target_question):
        user_ability = torch.sigmoid(self.proj_latent2ability(latent))
        que_difficulty = torch.sigmoid(self.proj_que2difficulty(question_emb))
        que_discrimination = torch.sigmoid(self.proj_que2discrimination(question_emb)) * 10
        # 要将target_que_concept变成可学习的一个参数
        target_que_concept = self.objects["cognition_tracing"]["q_matrix"][target_question]
        y = (que_discrimination * (user_ability - que_difficulty))
        predict_score = torch.sigmoid(torch.sum(y * target_que_concept, dim=-1))

        return predict_score

    def forward(self, batch):
        dim_correct = self.params["models_config"]["kt_model"]["encoder_layer"]["DCT"]["dim_correct"]
        correct_seq = batch["correct_seq"]
        question_seq = batch["question_seq"]
        batch_size, seq_len = correct_seq.shape[0], correct_seq.shape[1]

        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_correct).reshape(batch_size, -1, dim_correct)
        question_emb = self.embed_question(question_seq)
        interaction_emb = torch.cat((question_emb[:, :-1], correct_emb[:, :-1]), dim=2)

        self.encoder_layer.flatten_parameters()
        latent, _ = self.encoder_layer(interaction_emb)
        predict_score = self.predict_score(latent, question_emb[:, 1:], question_seq[:, 1:])

        return predict_score

    def get_que_diff_pred_loss(self, target_question):
        Q_table_mask = self.objects["cognition_tracing"]["Q_table_mask"]
        que_diff_label = self.objects["cognition_tracing"]["que_diff_ground_truth"]
        mask = Q_table_mask[target_question].bool()

        question_emb = self.embed_question(target_question)
        pred_diff_all = torch.sigmoid(self.proj_que2difficulty(self.dropout(question_emb)))
        pred_diff = torch.masked_select(pred_diff_all, mask)
        ground_truth = torch.masked_select(que_diff_label[target_question], mask)
        predict_loss = nn.functional.mse_loss(pred_diff, ground_truth)

        return predict_loss

    def get_que_disc_pred_loss(self, target_question):
        question_emb = self.embed_question(target_question)
        pred_disc = torch.sigmoid(self.proj_que2discrimination(self.dropout(question_emb))).squeeze(dim=-1) * 10
        ground_truth = self.objects["cognition_tracing"]["que_disc_ground_truth"]
        predict_loss = nn.functional.mse_loss(pred_disc, ground_truth)

        return predict_loss

    def get_predict_score(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = self.forward(batch)
        predict_score = torch.masked_select(predict_score, mask_bool_seq[:, 1:])

        return predict_score

    def get_predict_loss(self, batch, loss_record=None):
        dim_correct = self.params["models_config"]["kt_model"]["encoder_layer"]["DCT"]["dim_correct"]
        num_concept = self.params["models_config"]["kt_model"]["encoder_layer"]["DCT"]["num_concept"]
        w_penalty_neg = self.params["loss_config"]["penalty neg loss"]
        w_learning = self.params["loss_config"]["learning loss"]

        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        correct_seq = batch["correct_seq"]
        question_seq = batch["question_seq"]
        batch_size, seq_len = correct_seq.shape[0], correct_seq.shape[1]

        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_correct).reshape(batch_size, -1, dim_correct)
        question_emb = self.embed_question(question_seq)
        interaction_emb = torch.cat((question_emb[:, :-1], correct_emb[:, :-1]), dim=2)

        self.encoder_layer.flatten_parameters()
        latent, _ = self.encoder_layer(interaction_emb)

        user_ability = torch.sigmoid(self.proj_latent2ability(self.dropout(latent)))
        que_difficulty = torch.sigmoid(self.proj_que2difficulty(self.dropout(question_emb[:, 1:])))
        que_discrimination = torch.sigmoid(self.proj_que2discrimination(self.dropout(question_emb[:, 1:]))) * 10
        # 要将target_que_concept变成可学习的一个参数
        target_que_concept = self.objects["cognition_tracing"]["q_matrix"][question_seq[:, 1:]]
        inter_func_in = user_ability - que_difficulty
        predict_score = torch.sigmoid(torch.sum(que_discrimination * inter_func_in * target_que_concept, dim=-1))
        predict_score = torch.masked_select(predict_score, mask_bool_seq[:, 1:])

        loss = 0.
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])
        predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())
        if loss_record is not None:
            num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
            loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)
        loss = loss + predict_loss

        # 对于做对的题，惩罚user_ability - que_difficulty小于0的值（只惩罚考察的知识点）
        q2c_table = self.objects["data"]["q2c_table"][batch["question_seq"][:, 1:]]
        q2c_mask_table = self.objects["data"]["q2c_mask_table"][batch["question_seq"][:, 1:]]
        if w_penalty_neg != 0:
            target_inter_func_in = torch.gather(inter_func_in, 2, q2c_table)
            mask4inter_func_in = mask_bool_seq[:, 1:].unsqueeze(-1) & \
                                 batch["correct_seq"][:, 1:].bool().unsqueeze(-1) & \
                                 q2c_mask_table.bool()
            target_inter_func_in = torch.masked_select(target_inter_func_in, mask4inter_func_in)
            neg_inter_func_in = target_inter_func_in[target_inter_func_in <= 0]
            if neg_inter_func_in.numel() > 0:
                penalty_neg_loss = -neg_inter_func_in.mean()
                if loss_record is not None:
                    num_sample = neg_inter_func_in.shape[0]
                    loss_record.add_loss("penalty neg loss", penalty_neg_loss.detach().cpu().item() * num_sample,
                                         num_sample)
                loss = loss + penalty_neg_loss * w_penalty_neg

        return loss
