import torch
from torch import nn
import torch.nn.functional as F


class IRT(nn.Module):
    model_name = "IRT"

    def __init__(self, params, objects):
        super(IRT, self).__init__()
        self.params = params
        self.objects = objects

        model_config = self.params["models_config"]["cd_model"]["IRT"]
        num_user = model_config["num_user"]
        num_question = model_config["num_question"]
        value_range = model_config["value_range"]
        a_range = model_config["a_range"]

        self.theta = nn.Embedding(num_user, 1)
        self.a = nn.Embedding(num_question, 1)
        self.b = nn.Embedding(num_question, 1)
        self.c = nn.Embedding(num_question, 1)
        self.value_range = value_range
        self.a_range = a_range

    def forward(self, batch):
        user_id = batch["user_id"]
        question_id = batch["question_id"]

        model_config = self.params["models_config"]["cd_model"]["IRT"]
        value_range = model_config["value_range"]
        a_range = model_config["a_range"]
        D = model_config["D"]

        theta = torch.squeeze(self.theta(user_id), dim=-1)
        a = torch.squeeze(self.a(question_id), dim=-1)
        b = torch.squeeze(self.b(question_id), dim=-1)
        c = torch.squeeze(self.c(question_id), dim=-1)
        c = torch.sigmoid(c)
        if value_range > 0:
            theta = value_range * (torch.sigmoid(theta) - 0.5)
            b = value_range * (torch.sigmoid(b) - 0.5)
        if a_range > 0:
            a = a_range * torch.sigmoid(a)
        else:
            a = F.softplus(a)

        if torch.max(theta != theta) or torch.max(a != a) or torch.max(b != b):
            raise ValueError('ValueError:theta,a,b may contains nan!  The value_range or a_range is too large.')

        return c + (1 - c) / (1 + torch.exp(-D * a * (theta - b)))

    def get_predict_loss(self, batch):
        predict_score = self.forward(batch)
        ground_truth = batch["correct"]
        loss = torch.nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())
        return loss

    def get_predict_score(self, batch):
        return self.forward(batch)
