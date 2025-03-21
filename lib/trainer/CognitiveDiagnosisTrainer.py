import torch
import os
import torch.nn as nn

from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error, mean_squared_error

from .util import *
from ..util.basic import *
from .LossRecord import *
from .TrainRecord import *


class CognitiveDiagnosisTrainer:
    def __init__(self, params, objects):
        self.params = params
        self.objects = objects
        self.best_model = None
        self.objects["optimizers"] = {}
        self.objects["schedulers"] = {}

        self.loss_record = self.init_loss_record()
        self.train_record = TrainRecord(params, objects)
        self.init_trainer()

    def init_loss_record(self):
        used_losses = ["predict loss"]
        loss_config = self.params["loss_config"]
        for loss_name in loss_config:
            used_losses.append(loss_name)
        return LossRecord(used_losses)

    def init_trainer(self):
        # 初始化optimizer和scheduler
        models = self.objects["models"]
        optimizers = self.objects["optimizers"]
        schedulers = self.objects["schedulers"]
        optimizers_config = self.params["optimizers_config"]
        schedulers_config = self.params["schedulers_config"]

        for model_name, optimizer_config in optimizers_config.items():
            scheduler_config = schedulers_config[model_name]
            optimizers[model_name] = create_optimizer(models[model_name].parameters(), optimizer_config)

            if scheduler_config["use_scheduler"]:
                schedulers[model_name] = create_scheduler(optimizers[model_name], scheduler_config)
            else:
                schedulers[model_name] = None

    def train(self):
        train_strategy = self.params["train_strategy"]
        grad_clip_config = self.params["grad_clip_config"]["cd_model"]
        schedulers_config = self.params["schedulers_config"]["cd_model"]
        num_epoch = train_strategy["num_epoch"]
        train_loader = self.objects["data_loaders"]["train_loader"]
        optimizer = self.objects["optimizers"]["cd_model"]
        scheduler = self.objects["schedulers"]["cd_model"]
        model = self.objects["models"]["cd_model"]

        self.print_data_statics()

        for epoch in range(1, num_epoch + 1):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                predict_loss = model.get_predict_loss(batch)
                num_sample = batch["user_id"].shape[0]
                self.loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)
                predict_loss.backward()
                if grad_clip_config["use_clip"]:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_config["grad_clipped"])
                optimizer.step()
                if model.model_name == "NCD":
                    model.apply_clipper()
            if schedulers_config["use_scheduler"]:
                scheduler.step()
            self.evaluate()
            if self.stop_train():
                break

    def stop_train(self):
        train_strategy = self.params["train_strategy"]
        stop_flag = self.train_record.stop_training()
        if stop_flag:
            if train_strategy["type"] == "no_valid":
                pass
            else:
                best_train_performance_by_valid = self.train_record.get_evaluate_result_str("train", "valid")
                best_valid_performance_by_valid = self.train_record.get_evaluate_result_str("valid", "valid")
                best_test_performance_by_valid = self.train_record.get_evaluate_result_str("test", "valid")

                self.objects["logger"].info(
                    f"best valid epoch: {self.train_record.get_best_epoch('valid'):<3} , "
                    f"best test epoch: {self.train_record.get_best_epoch('test')}\n"
                    f"train performance by best valid epoch is {best_train_performance_by_valid}\n"
                    f"valid performance by best valid epoch is {best_valid_performance_by_valid}\n"
                    f"test performance by best valid epoch is {best_test_performance_by_valid}\n"
                    f"{'-'*100}\n"
                    f"train performance by best train epoch is "
                    f"{self.train_record.get_evaluate_result_str('train', 'train')}\n"
                    f"test performance by best test epoch is "
                    f"{self.train_record.get_evaluate_result_str('test', 'test')}\n"
                )

        return stop_flag

    def evaluate(self):
        train_strategy = self.params["train_strategy"]
        save_model = self.params["save_model"]
        data_loaders = self.objects["data_loaders"]
        train_loader = data_loaders["train_loader"]
        model = self.objects["models"]["cd_model"]
        train_performance = self.evaluate_cd_dataset(model, train_loader)
        if train_strategy["type"] == "valid_test":
            # 有验证集，同时在验证集和测试集上测试
            data_loader = data_loaders["valid_loader"]
            valid_performance = self.evaluate_cd_dataset(model, data_loader)
            data_loader = data_loaders["test_loader"]
            test_performance = self.evaluate_cd_dataset(model, data_loader)
            self.train_record.next_epoch(train_performance, valid_performance, test_performance)
            valid_performance_str = self.train_record.get_performance_str("valid")
            test_performance_str = self.train_record.get_performance_str("test")
            best_epoch = self.train_record.get_best_epoch("valid")
            self.objects["logger"].info(
                f"{get_now_time()} epoch {self.train_record.get_current_epoch():<3} , valid performance is "
                f"{valid_performance_str}train loss is {self.loss_record.get_str()}, test performance is "
                f"{test_performance_str}current best epoch is {best_epoch}")
        else:
            # 无测试集
            data_loader = data_loaders["valid_loader"]
            valid_performance = self.evaluate_cd_dataset(model, data_loader)
            self.train_record.next_epoch(train_performance, valid_performance)
            valid_performance_str = self.train_record.get_performance_str("valid")
            best_epoch = self.train_record.get_best_epoch("valid")
            self.objects["logger"].info(
                f"{get_now_time()} epoch {self.train_record.get_current_epoch():<3} , valid performance is "
                f"{valid_performance_str}train loss is {self.loss_record.get_str()}current best epoch is {best_epoch}")

        self.loss_record.clear_loss()
        current_epoch = self.train_record.get_current_epoch()
        if best_epoch == current_epoch:
            if save_model:
                save_model_dir = self.params["save_model_dir"]
                model_weight_path = os.path.join(save_model_dir, "saved.ckt")
                torch.save({"best_valid": model.state_dict()}, model_weight_path)

    def print_data_statics(self):
        train_strategy = self.params["train_strategy"]
        train_loader = self.objects["data_loaders"]["train_loader"]
        valid_loader = self.objects["data_loaders"]["valid_loader"]
        self.objects["logger"].info("")
        if train_strategy["type"] == "valid_test":
            test_loader = self.objects["data_loaders"]["test_loader"]
            self.objects["logger"].info(f"train, sample: {len(train_loader.dataset)}\n"
                                        f"valid, sample: {len(valid_loader.dataset)}\n"
                                        f"test, sample: {len(test_loader.dataset)}")
        else:
            self.objects["logger"].info(f"train, sample: {len(train_loader.dataset)}\n"
                                        f"valid, sample: {len(valid_loader.dataset)}")
        self.objects["logger"].info("")

    @staticmethod
    def evaluate_cd_dataset(model, data_loader):
        model.eval()
        with torch.no_grad():
            predict_score_all = []
            ground_truth_all = []
            for batch in data_loader:
                predict_score = model.get_predict_score(batch).detach().cpu().numpy()
                ground_truth = batch["correct"].detach().cpu().numpy()
                predict_score_all.append(predict_score)
                ground_truth_all.append(ground_truth)
            predict_score_all = np.concatenate(predict_score_all, axis=0)
            ground_truth_all = np.concatenate(ground_truth_all, axis=0)
            predict_label_all = [1 if p >= 0.5 else 0 for p in predict_score_all]
            AUC = roc_auc_score(y_true=ground_truth_all, y_score=predict_score_all)
            ACC = accuracy_score(y_true=ground_truth_all, y_pred=predict_label_all)
            MAE = mean_absolute_error(y_true=ground_truth_all, y_pred=predict_score_all)
            RMSE = mean_squared_error(y_true=ground_truth_all, y_pred=predict_score_all) ** 0.5
        return {"AUC": AUC, "ACC": ACC, "MAE": MAE, "RMSE": RMSE}
