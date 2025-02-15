import numpy as np
from tqdm import tqdm
from copy import deepcopy

from .TrainRecord import *
from ..evaluator.util import get_performance_no_error


def copy_fit_model(from_model, to_model, item_idx):
    to_model["Pi"][item_idx] = deepcopy(from_model["Pi"][item_idx])
    to_model["T"][item_idx] = deepcopy(from_model["T"][item_idx])
    to_model["E"][item_idx] = deepcopy(from_model["E"][item_idx])


class BKT:
    def __init__(self, params, objects):
        self.params = params
        self.objects = objects

        self.fit_model = {}
        self.train_record = TrainRecord(params, objects)
        self.init_model()
        self.best_fit_model = deepcopy(self.fit_model)

    def init_model(self):
        model_params = self.params["model_params"]

        num_item = model_params["num_item"]
        num_state = model_params["num_state"]
        num_observation = model_params["num_observation"]

        # init the transmission probability matrix T, the emission probability matrix E and the initial state probability Pi
        self.fit_model["T"] = []
        self.fit_model["E"] = []
        self.fit_model["Pi"] = np.array([[0.5, 0.5] for _ in range(num_item)])
        for i in range(num_item):
            T = np.random.uniform(0, 1, num_state * num_state).reshape(num_state, num_state)
            self.fit_model["T"].append(T.astype(float) / (T.sum(axis=1, keepdims=True) + 1e-8))
            E = np.random.uniform(0, 1, num_state * num_observation).reshape(num_state, num_observation)
            self.fit_model["E"].append(E.astype(float) / (E.sum(axis=1, keepdims=True) + 1e-8))
        self.fit_model["T"] = np.stack(self.fit_model["T"])
        self.fit_model["E"] = np.stack(self.fit_model["E"])

    def forward(self, data, item_idx):
        """
        Calculate the probability of an observation sequence given model params
        """
        model_params = self.params["model_params"]
        num_state = model_params["num_state"]
        use_scaling = model_params["use_scaling"]

        seq_data = data["data"]
        seq_starts = data["seq_starts"]
        seq_ends = data["seq_ends"]
        seq_lens = data["seq_lens"]

        all_scaling_factor = []
        all_alpha = []
        for t_start, t_end, T in zip(seq_starts, seq_ends, seq_lens):
            seq = seq_data[t_start:t_end]
            alpha = np.zeros([num_state, T], float)
            alpha[:, 0] = self.fit_model["Pi"][item_idx]
            scaling_factors = []
            for t in range(T):
                if t != 0:
                    alpha[:, t] = np.dot(alpha[:, t - 1], self.fit_model["T"][item_idx])
                alpha[:, t] *= self.fit_model["E"][item_idx][:, seq[t]]
                scaling_factor = (1 / (alpha[:, t].sum() + 1e-8)) if use_scaling else 1
                alpha[:, t] *= scaling_factor
                scaling_factors.append(scaling_factor)
            all_alpha.append(alpha)
            all_scaling_factor.append(scaling_factors)

        return {
            "all_alpha": all_alpha,
            "all_scaling_factor": all_scaling_factor
        }

    def backward(self, data, item_idx):
        """
        Calculate the probability of a partial observation sequence from t+1 to T given the model params.
        """
        model_params = self.params["model_params"]
        num_state = model_params["num_state"]
        use_scaling = model_params["use_scaling"]

        seq_data = data["data"]
        seq_starts = data["seq_starts"]
        seq_ends = data["seq_ends"]
        seq_lens = data["seq_lens"]

        all_scaling_factor = []
        all_beta = []
        for t_start, t_end, T in zip(seq_starts, seq_ends, seq_lens):
            seq = seq_data[t_start:t_end]
            beta = np.zeros([num_state, T], float)
            beta[:, T - 1] = 1
            scaling_factors = []
            for t in list(range(T))[::-1]:
                if t != T - 1:
                    beta[:, t] = np.dot(
                        self.fit_model["T"][item_idx] * self.fit_model["E"][item_idx][:, seq[t + 1]],
                        beta[:, t + 1]
                    )
                scaling_factor = (1 / (beta[:, t].sum() + 1e-8)) if use_scaling else 1
                beta[:, t] *= scaling_factor
                scaling_factors.append(scaling_factor)
            all_beta.append(beta)
            all_scaling_factor.append(scaling_factors)

        return {
            "all_beta": all_beta,
            "all_scaling_factor": all_scaling_factor
        }

    def e_step(self, data, item_idx):
        """
        Calculate the posterior probability and the transition probability
        """
        f_result = self.forward(data, item_idx)
        b_result = self.backward(data, item_idx)

        # gamma: the posterior probability
        all_alpha = f_result["all_alpha"]
        all_beta = b_result["all_beta"]
        all_gamma = []
        for alpha, beta in zip(all_alpha, all_beta):
            raw_gamma = alpha * beta
            all_gamma.append(raw_gamma / (raw_gamma.sum(0) + 1e-8))

        # sigma: the transition probability
        model_params = self.params["model_params"]
        num_state = model_params["num_state"]

        seq_data = data["data"]
        seq_starts = data["seq_starts"]
        seq_ends = data["seq_ends"]
        seq_lens = data["seq_lens"]

        all_sigma = []
        for i, (t_start, t_end, T) in enumerate(zip(seq_starts, seq_ends, seq_lens)):
            seq = seq_data[t_start:t_end]
            sigma = np.zeros([T - 1, num_state, num_state], float)
            alpha = all_alpha[i]
            beta = all_beta[i]
            T_matrix = self.fit_model["T"][item_idx]
            E_matrix = self.fit_model["E"][item_idx]
            for t in range(T - 1):
                for s_i in range(num_state):
                    for s_j in range(num_state):
                        sigma[t, s_i, s_j] = alpha[s_i, t] * T_matrix[s_i, s_j] * E_matrix[s_j, seq[t + 1]] * beta[
                            s_j, t + 1]
                sigma[t, :, :] /= (sigma[t, :, :].sum() + 1e-8)
            all_sigma.append(sigma)

        return all_gamma, all_sigma

    def m_step(self, data, item_idx, all_gamma, all_sigma):
        model_params = self.params["model_params"]
        num_state = model_params["num_state"]
        num_observation = model_params["num_observation"]
        fixed_params = model_params["fixed_params"]

        if "Pi" not in fixed_params:
            pi = np.zeros((num_state, ))
            for gamma in all_gamma:
                pi += gamma[:, 0].flatten()
            self.fit_model["Pi"][item_idx] = pi / (pi.sum() + 1e-8)

        if "T" not in fixed_params:
            sigma_sum = [[0 for _ in range(num_state)] for _ in range(num_state)]
            gamma_sum = [0 for _ in range(num_state)]
            for gamma, sigma in zip(all_gamma, all_sigma):
                for s_i in range(num_state):
                    gamma_sum[s_i] += gamma[s_i, :-1].sum()
                    for s_j in range(num_state):
                        sigma_sum[s_i][s_j] += sigma[:, s_i, s_j].sum()
            for s_i in range(num_state):
                for s_j in range(num_state):
                    self.fit_model["T"][item_idx][s_i, s_j] = sigma_sum[s_i][s_j] / (gamma_sum[s_i] + 1e-8)

        if "E" not in fixed_params:
            seq_data = data["data"]
            seq_starts = data["seq_starts"]
            seq_ends = data["seq_ends"]
            denominator_sum = [[0 for _ in range(num_observation)] for _ in range(num_state)]
            numerator_sum = [0 for _ in range(num_state)]
            for i, gamma in enumerate(all_gamma):
                for s in range(num_state):
                    numerator_sum[s] += gamma[s, :].sum()
                    for o in range(num_observation):
                        o_seq = np.array(seq_data[seq_starts[i]: seq_ends[i]])
                        denominator_sum[s][o] += gamma[s, :][o_seq == o].sum()
            for s in range(num_state):
                for o in range(num_observation):
                    self.fit_model["E"][item_idx][s, o] = denominator_sum[s][o] / (numerator_sum[s] + 1e-8)

    def train(self):
        """
        Using EM algorithm to fit data
        使用的是baum_welch算法（1972），即前-后向算法，该算法是EM算法（1977）的一个特例
        """
        data_train = self.objects["data_train"]
        model_params = self.params["model_params"]
        train_strategy = self.params["train_strategy"]
        num_item = model_params["num_item"]
        num_epoch = train_strategy["num_epoch"]

        self.objects["logger"].info(f"Start training:")
        for item_idx in tqdm(range(num_item)):
            for _ in range(num_epoch):
                all_gamma, all_sigma = self.e_step(data_train[item_idx], item_idx)
                self.m_step(data_train[item_idx], item_idx, all_gamma, all_sigma)
                self.evaluate(eval_one_item=True, item_idx=item_idx)
                if self.train_record.stop_training():
                    self.train_record = TrainRecord(self.params, self.objects)
                    break
            copy_fit_model(self.best_fit_model, self.fit_model, item_idx)
        self.objects["logger"].info(f"Evaluation on all item:")
        self.evaluate(eval_one_item=False)

    def evaluate_one_item(self, data, item_idx):
        labels = []
        predict_scores = []
        all_alpha = self.forward(data[item_idx], item_idx)["all_alpha"]
        seq_data = data[item_idx]["data"]
        seq_starts = data[item_idx]["seq_starts"]
        seq_ends = data[item_idx]["seq_ends"]
        for alpha, seq_start, seq_end in zip(all_alpha, seq_starts, seq_ends):
            seq = seq_data[seq_start:seq_end]
            labels += seq[1:]
            predict_scores += (alpha[:, :-1] * self.fit_model["E"][item_idx][:, 1:2]).sum(0).tolist()

        predict_labels = [1 if p > 0.5 else 0 for p in predict_scores]
        performance = get_performance_no_error(predict_scores, predict_labels, labels)
        return performance

    def evaluate_all_item(self, data):
        model_params = self.params["model_params"]
        num_item = model_params["num_item"]

        all_label = [[] for _ in range(num_item)]
        all_predict_score = [[] for _ in range(num_item)]
        for item_idx in range(num_item):
            all_alpha = self.forward(data[item_idx], item_idx)["all_alpha"]
            seq_data = data[item_idx]["data"]
            seq_starts = data[item_idx]["seq_starts"]
            seq_ends = data[item_idx]["seq_ends"]
            for alpha, seq_start, seq_end in zip(all_alpha, seq_starts, seq_ends):
                seq = seq_data[seq_start:seq_end]
                all_label[item_idx] += seq[1:]
                predict_score = (alpha[:, :-1] * self.fit_model["E"][item_idx][:, 1:2]).sum(0).tolist()
                all_predict_score[item_idx] += predict_score

        ls_all_item = [label for ls in all_label for label in ls]
        ps_all_item = [p for ps in all_predict_score for p in ps]
        pl_all_item = [1 if p > 0.5 else 0 for p in ps_all_item]
        performance = get_performance_no_error(ps_all_item, pl_all_item, ls_all_item)
        return performance

    def evaluate(self, eval_one_item=True, item_idx=0):
        data_train = self.objects["data_train"]
        data_valid = self.objects["data_valid"]
        data_test = self.objects["data_test"]

        train_strategy = self.params["train_strategy"]
        save_model = self.params["save_model"]

        if eval_one_item:
            train_performance = self.evaluate_one_item(data_train, item_idx)
        else:
            train_performance = self.evaluate_all_item(data_train)
        if train_strategy["type"] == "no_test":
            if eval_one_item:
                valid_performance = self.evaluate_one_item(data_valid, item_idx)
            else:
                valid_performance = self.evaluate_all_item(data_valid)
            self.train_record.next_epoch(train_performance, valid_performance)
            best_epoch = self.train_record.get_best_epoch("valid")
            if not eval_one_item:
                self.objects["logger"].info(
                    f"train performance, AUC: {train_performance['AUC']:<9.5}, "
                    f"ACC: {train_performance['ACC']:<9.5}, "
                    f"RMSE: {train_performance['RMSE']:<9.5}, "
                    f"MAE: {train_performance['MAE']:<9.5}\n"
                    f"valid performance, AUC: {valid_performance['AUC']:<9.5}, "
                    f"ACC: {valid_performance['ACC']:<9.5}, "
                    f"RMSE: {valid_performance['RMSE']:<9.5}, "
                    f"MAE: {valid_performance['MAE']:<9.5}\n"
                )
            current_epoch = self.train_record.get_current_epoch()
            if best_epoch == current_epoch:
                copy_fit_model(self.fit_model, self.best_fit_model, item_idx)
                if save_model:
                    save_model_dir = self.params["save_model_dir"]
                    # todo: save
                    pass
        else:
            if eval_one_item:
                valid_performance = self.evaluate_one_item(data_valid, item_idx)
                test_performance = self.evaluate_one_item(data_test, item_idx)
            else:
                valid_performance = self.evaluate_all_item(data_valid)
                test_performance = self.evaluate_all_item(data_test)
            self.train_record.next_epoch(train_performance, valid_performance, test_performance)
            best_epoch = self.train_record.get_best_epoch("valid")
            if not eval_one_item:
                self.objects["logger"].info(
                    f"train performance, AUC: {train_performance['AUC']:<9.5}, "
                    f"ACC: {train_performance['ACC']:<9.5}, "
                    f"RMSE: {train_performance['RMSE']:<9.5}, "
                    f"MAE: {train_performance['MAE']:<9.5}\n"
                    f"valid performance, AUC: {valid_performance['AUC']:<9.5}, "
                    f"ACC: {valid_performance['ACC']:<9.5}, "
                    f"RMSE: {valid_performance['RMSE']:<9.5}, "
                    f"MAE: {valid_performance['MAE']:<9.5}\n"
                    f"test performance, AUC: {test_performance['AUC']:<9.5}, "
                    f"ACC: {test_performance['ACC']:<9.5}, "
                    f"RMSE: {test_performance['RMSE']:<9.5}, "
                    f"MAE: {test_performance['MAE']:<9.5}\n"
                )
            current_epoch = self.train_record.get_current_epoch()
            if best_epoch == current_epoch:
                copy_fit_model(self.fit_model, self.best_fit_model, item_idx)
                if save_model:
                    save_model_dir = self.params["save_model_dir"]
                    # todo: save
