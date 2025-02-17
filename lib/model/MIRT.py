import torch
from torch import nn
import torch.nn.functional as F


class MIRT(nn.Module):
    model_name = "MIRT"

    def __init__(self, params, objects):
        super(MIRT, self).__init__()
        self.params = params
        self.objects = objects

        model_config = self.params["models_config"]["cd_model"]["MIRT"]
        num_user = model_config["num_user"]
        num_question = model_config["num_question"]
        num_concept = model_config["num_concept"]

        self.theta = nn.Embedding(num_user, num_concept)
        self.a = nn.Embedding(num_question, num_concept)
        self.b = nn.Embedding(num_question, 1)

    def forward(self, batch):
        user_id = batch["user_id"]
        question_id = batch["question_id"]

        model_config = self.params["models_config"]["cd_model"]["MIRT"]
        a_range = model_config["a_range"]

        theta = torch.squeeze(self.theta(user_id), dim=-1)
        a = torch.squeeze(self.a(question_id), dim=-1)
        if a_range > 0:
            a = a_range * torch.sigmoid(a)
        else:
            a = F.softplus(a)
        b = torch.squeeze(self.b(question_id), dim=-1)
        if torch.max(theta != theta) or torch.max(a != a) or torch.max(b != b):  # pragma: no cover
            raise ValueError('ValueError:theta,a,b may contains nan!  The a_range is too large.')

        return 1 / (1 + torch.exp(- torch.sum(torch.multiply(a, theta), dim=-1) + b))

    def get_predict_loss(self, batch):
        predict_score = self.forward(batch)
        ground_truth = batch["correct"]
        loss = torch.nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())
        return loss

    def get_predict_score(self, batch):
        return self.forward(batch)
