import json

import torch
import os
import torch.nn as nn

from .util import *
from ..util.basic import *
from .LossRecord import *
from .TrainRecord import *


def get_average_performance_top_ns(performance_top_ns):
    metric_names = list(list(performance_top_ns.values())[0].keys())
    average_performance_top_ns = {metric_name: [] for metric_name in metric_names}
    for top_n, top_n_performance in performance_top_ns.items():
        for metric_name, metric_value in top_n_performance.items():
            average_performance_top_ns[metric_name].append(metric_value)
    for metric_name, metric_values in average_performance_top_ns.items():
        average_performance_top_ns[metric_name] = sum(metric_values) / len(metric_values)
    return average_performance_top_ns


class ExerciseRecommendationTrainer:
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
        train_based_epoch = self.params["train_strategy"]["train_based_epoch"]
        if train_based_epoch:
            self.train_based_on_epoch()
        else:
            self.train_based_on_step()

    def train_based_on_epoch(self):
        train_strategy = self.params["train_strategy"]
        grad_clip_config = self.params["grad_clip_config"]["er_model"]
        schedulers_config = self.params["schedulers_config"]["er_model"]
        num_epoch = train_strategy["num_epoch"]
        train_loader = self.objects["data_loaders"]["train_loader"]
        optimizer = self.objects["optimizers"]["er_model"]
        scheduler = self.objects["schedulers"]["er_model"]
        model = self.objects["models"]["er_model"]

        for epoch in range(1, num_epoch + 1):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                predict_loss = model.get_predict_loss(batch)
                num_sample = batch["user_id"].shape[0]
                self.loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample,
                                          num_sample)
                predict_loss.backward()
                if grad_clip_config["use_clip"]:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_config["grad_clipped"])
                optimizer.step()
            if schedulers_config["use_scheduler"]:
                scheduler.step()
            self.evaluate_based_on_epoch()
            if self.stop_train_based_on_epoch():
                break

    def train_based_on_step(self):
        train_strategy = self.params["train_strategy"]
        train_strategy_type = train_strategy["type"]
        use_early_stop = train_strategy[train_strategy_type]["use_early_stop"]
        grad_clip_config = self.params["grad_clip_config"]["er_model"]
        schedulers_config = self.params["schedulers_config"]["er_model"]
        num_step = train_strategy["num_step"]
        num_step2evaluate = train_strategy["num_step2evaluate"]
        num_early_stop = train_strategy[train_strategy_type]["num_early_stop"]
        use_multi_metrics = train_strategy["use_multi_metrics"]
        main_metric_key = train_strategy["main_metric"]
        multi_metrics = train_strategy["multi_metrics"]
        save_model = self.params["save_model"]

        model = self.objects["models"]["er_model"]
        optimizer = self.objects["optimizers"]["er_model"]
        scheduler = self.objects["schedulers"]["er_model"]
        train_loader = self.objects["data_loaders"]["train_loader"]
        logs_every_step = []
        logs_performance = []
        best_valid_main_metric = -100
        best_index = 0
        self.objects["logger"].info("\nstart training:")
        for step in range(num_step):
            optimizer.zero_grad()
            loss_result = model.train_one_step(next(train_loader))
            loss_result["total_loss"].backward()
            if grad_clip_config["use_clip"]:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_config["grad_clipped"])
            optimizer.step()

            if schedulers_config["use_scheduler"]:
                scheduler.step()

            loss_record = {}
            for loss_name, loss_info in loss_result["losses_value"].items():
                if loss_name != "total_loss":
                    loss_record[loss_name] = {
                        "num_sample": loss_info["num_sample"],
                        "value": loss_info["value"]
                    }
            log_step = {"loss_record": loss_record}
            logs_every_step.append(log_step)

            if (step > 0) and (step % 100 == 0):
                logs_step = logs_every_step[step - 100:]
                loss_str = f""
                for loss_name in logs_step[0]["loss_record"].keys():
                    loss_value = 0
                    num_sample = 0
                    for log_one_step in logs_step:
                        loss_value += log_one_step["loss_record"][loss_name]["value"]
                        num_sample += log_one_step["loss_record"][loss_name]["num_sample"]
                    loss_value = loss_value / num_sample
                    loss_str += f"{loss_name}: {loss_value:<12.6}, "
                self.objects["logger"].info(f"{get_now_time()} step {step:<9}: train loss is {loss_str}")

            if (step > 0) and (step % num_step2evaluate == 0):
                log_performance = {}

                valid_data_loader = self.objects["data_loaders"]["valid_loader"]
                valid_performance = model.evaluate(valid_data_loader)
                average_valid_performance = get_average_performance_top_ns(valid_performance)
                if use_multi_metrics:
                    valid_main_metric = TrainRecord.cal_main_metric(average_valid_performance, multi_metrics)
                else:
                    valid_main_metric = ((-1 if main_metric_key in ["RMSE", "MAE"] else 1) *
                                         average_valid_performance[main_metric_key])
                log_performance["valid_performance"] = valid_performance
                log_performance["valid_main_metric"] = valid_main_metric

                performance_str = ""
                for top_n, top_n_performance in valid_performance.items():
                    performance_str += f"top{top_n}, "
                    for metric_name, metric_value in top_n_performance.items():
                        performance_str += f"{metric_name}: {metric_value:<9.5}, "
                    performance_str += "\n"
                self.objects["logger"].info(
                    f"{get_now_time()} step {step:<9}, valid performance, main metric: "
                    f"{valid_main_metric}\n{performance_str}")

                if train_strategy_type == "valid_test":
                    test_data_loader = self.objects["data_loaders"]["test_loader"]
                    test_performance = model.evaluate(test_data_loader)
                    average_test_performance = get_average_performance_top_ns(test_performance)
                    if use_multi_metrics:
                        test_main_metric = TrainRecord.cal_main_metric(average_test_performance, multi_metrics)
                    else:
                        test_main_metric = ((-1 if main_metric_key in ["RMSE", "MAE"] else 1) *
                                            average_test_performance[main_metric_key])
                    log_performance["test_performance"] = test_performance
                    log_performance["test_main_metric"] = test_main_metric

                    performance_str = ""
                    for top_n, top_n_performance in test_performance.items():
                        performance_str += f"top{top_n}, "
                        for metric_name, metric_value in top_n_performance.items():
                            performance_str += f"{metric_name}: {metric_value:<9.5}, "
                        performance_str += "\n"
                    self.objects["logger"].info(
                        f"{get_now_time()} step {step:<9}, test performance, main metric: "
                        f"{test_main_metric}\n{performance_str}")

                logs_performance.append(log_performance)

                if use_early_stop:
                    current_index = int(step / num_step2evaluate) - 1
                    if (valid_main_metric - best_valid_main_metric) > 0.001:
                        best_valid_main_metric = valid_main_metric
                        best_index = current_index
                        if save_model:
                            save_model_dir = self.params["save_model_dir"]
                            model_weight_path = os.path.join(save_model_dir, "saved.ckt")
                            torch.save({"best_valid": model.state_dict()}, model_weight_path)

                    if ((current_index - best_index) >= num_early_stop) or ((num_step - step) < num_step2evaluate):
                        best_log_performance = logs_performance[best_index]
                        if train_strategy["type"] == "valid_test":
                            self.objects["logger"].info(
                                f"best valid step: {num_step2evaluate * (best_index + 1):<9}\n"
                                f"valid performance by best valid is "
                                f"{json.dumps(best_log_performance['valid_performance'])}\n"
                                f"test performance by best valid is "
                                f"{json.dumps(best_log_performance['test_performance'])}\n"
                            )
                        elif train_strategy["type"] == "no_test":
                            self.objects["logger"].info(
                                f"best valid step: {num_step2evaluate * (best_index + 1):<9}\n"
                                f"valid performance by best valid epoch is "
                                f"{json.dumps(best_log_performance['valid_performance'])}\n"
                            )
                        else:
                            raise NotImplementedError()
                        break

    def stop_train_based_on_epoch(self):
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
                    f"{'-' * 100}\n"
                    f"train performance by best train epoch is "
                    f"{self.train_record.get_evaluate_result_str('train', 'train')}\n"
                    f"test performance by best test epoch is "
                    f"{self.train_record.get_evaluate_result_str('test', 'test')}\n"
                )

        return stop_flag

    def evaluate_based_on_epoch(self):
        train_strategy = self.params["train_strategy"]
        save_model = self.params["save_model"]
        data_loaders = self.objects["data_loaders"]
        train_loader = data_loaders["train_loader"]
        model = self.objects["models"]["cd_model"]
        train_performance = model.evaluate(model, train_loader)
        if train_strategy["type"] == "valid_test":
            # 有验证集，同时在验证集和测试集上测试
            data_loader = data_loaders["valid_loader"]
            valid_performance = model.evaluate(model, data_loader)
            data_loader = data_loaders["test_loader"]
            test_performance = model.evaluate(model, data_loader)
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
            valid_performance = model.evaluate(model, data_loader)
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
