import json
import os
import argparse
import time
import pickle
import numpy as np
import pandas as pd
import scipy.io as scio
import scipy.sparse
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn import metrics
from functools import partial


class Dataset():
    def __init__(self, dataset_name="HMDAD", seed=666, mode="CVS1", n_splits=5, equal_train_num=False, equal_test_num=False, **kwargs):
        assert mode in ["CVS1", "CVS2", "CVS3"]
        self.dataset_name = dataset_name
        self.mode = mode
        self.n_splits = n_splits
        self.equal_train = equal_train_num
        self.equal_test = equal_test_num
        self.seed = seed
        load_fn = self.get_load_fn(dataset_name)
        self.interaction, self.d_feature, self.m_feature, self.d_info, self.m_info = load_fn()
        if equal_train_num and equal_test_num:
            equal_flag = "num"
        elif not equal_train_num and not equal_test_num:
            equal_flag = "rate"
        elif equal_train_num:
            equal_flag = "train_num"
        else:
            equal_flag = "test_num"
        self.sub_dir = os.path.join(self.dataset_name,
                                    f"{self.mode}-split_{n_splits}-seed-{self.seed}-equal_{equal_flag}")
        self.save_dir = os.path.join("experiment_split", self.sub_dir)
        print(f"load {dataset_name}, shape={self.interaction.shape}, disease_num:{len(self.d_feature)}, microbe_num:{len(self.m_feature)}")

    def get_load_fn(self, dataset_name):
        if dataset_name == "HMDAD":
            load_fn = self.load_HMDAD
        elif dataset_name == "Disbiome":
            load_fn = self.load_Disbiome
        elif dataset_name == "Combined":
            load_fn = self.load_Peryton_MicroPhenoDB
        return load_fn

    def load_Peryton_MicroPhenoDB(self, path="./dataset/Combined"):
        with open(os.path.join(path, "allData.p"), "rb") as f:
            data = pickle.load(f)
            u2e, i2e, u_train, i_train, r_train, u_test, i_test, r_test, u_adj, i_adj = data
            disease_idx = np.array(u_train+u_test)
            microbe_idx = np.array(i_train+i_test)
            interaction_idx = np.array(r_train+r_test)
            interaction = scipy.sparse.coo_matrix((interaction_idx, (disease_idx, microbe_idx)), shape=(u2e.shape[0], i2e.shape[0])).todense()
        interaction = np.array(interaction)
        disease_feature = u2e
        microbe_feature = i2e
        disease_info = pd.read_csv(os.path.join(path, "disease_index.txt"), sep="\t", header=None, names=["name", "idx"])
        microbe_info = pd.read_csv(os.path.join(path, "micro_index.txt"), sep="\t", header=None, names=["name", "idx"])
        return interaction, disease_feature, microbe_feature, disease_info, microbe_info

    def get_m_name(self, idx):
        return self.m_info.iloc[idx]["name"].values

    def get_d_name(self, idx):
        return self.d_info.iloc[idx]["name"].values

    def split(self):
        if self.mode=="CVS1":
            split_fn = partial(self.local_split, n_splits=self.n_splits, seed=self.seed,
                               equal_train=self.equal_train, equal_test=self.equal_test)
            total_folds = self.n_splits
        elif self.mode=="CVS2":
            split_fn = partial(self.leave_one_split, by_row=True, seed=self.seed,
                               equal_train=self.equal_train, equal_test=self.equal_test)
            total_folds = self.interaction.shape[0]
        elif self.mode=="CVS3":
            split_fn = partial(self.leave_one_split, by_row=False, seed=self.seed,
                               equal_train=self.equal_train, equal_test=self.equal_test)
            total_folds = self.interaction.shape[1]

        save_dir = self.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if len(os.listdir(save_dir))!=total_folds*2:
            print(f"create dataset split in {save_dir}")
            for i, (train_data, test_data) in enumerate(split_fn()):
                np.savetxt(os.path.join(save_dir, f"{i:05d}-train.txt"), train_data)
                np.savetxt(os.path.join(save_dir, f"{i:05d}-test.txt"), test_data)
        print(f"load dataset split from {save_dir}")
        files = sorted(os.listdir(save_dir))
        for test_file, train_file in zip(files[::2], files[1::2]):
            assert test_file.split("-")[0]==test_file.split("-")[0]
            train_data = np.loadtxt(os.path.join(save_dir, train_file)).astype(int)
            test_data = np.loadtxt(os.path.join(save_dir, test_file)).astype(int)
            yield (train_data, test_data, self.d_feature, self.m_feature, (len(self.d_feature), len(self.m_feature)))


    def load_HMDAD(self, path="./dataset/HMDAD"):
        interaction = scio.loadmat(os.path.join(path, "interaction.mat"))
        interaction = interaction['interaction']
        disease_info = pd.read_excel(os.path.join(path, "diseases.xlsx"), names=["idx", "name"], header=None)
        disease_feature = pd.read_csv(os.path.join(path, "disease_features.txt"), sep='\t', header=None)
        microbe_info = pd.read_excel(os.path.join(path, "microbes.xlsx"), names=["idx", "name"], header=None)
        microbe_feature = pd.read_csv(os.path.join(path, "microbe_features.txt"), sep='\t', header=None)
        return interaction, disease_feature.values, microbe_feature.values, disease_info, microbe_info

    def load_Disbiome(self, path="./dataset/Disbiome"):
        interaction = scio.loadmat(os.path.join(path, "interaction.mat"))
        interaction = interaction['interaction1']
        disease_info = pd.read_excel(os.path.join(path, "diseases.xlsx"), names=["idx", "name"], header=None)
        disease_feature = pd.read_csv(os.path.join(path, "disease_features.txt"), sep='\t', header=None)
        microbe_info = pd.read_excel(os.path.join(path, "microbes.xlsx"), names=["idx", "name"], header=None)
        microbe_feature = pd.read_csv(os.path.join(path, "microbe_features.txt"), sep='\t', header=None)
        return interaction, disease_feature.values, microbe_feature.values, disease_info, microbe_info

    def local_split(self, n_splits=5, seed=666, equal_train=False, equal_test=False):
        pos_row, pos_col = np.where(self.interaction>0.5)
        neg_row, neg_col = np.where(self.interaction<0.5)
        np.random.seed(seed)
        neg_index = np.random.permutation(np.arange(len(neg_row)))
        pos_kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        neg_kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for (train_index, test_index), (train_neg_index, test_neg_index) in zip(pos_kf.split(pos_row), neg_kf.split(neg_row)):
            train_pos_interaction = np.vstack([pos_row[train_index], pos_col[train_index], np.ones_like(train_index)])
            if equal_train:
                train_neg_interaction = np.vstack([neg_row[neg_index[train_index]], neg_col[neg_index[train_index]], np.zeros_like(train_index)])
            else:
                train_neg_interaction = np.vstack([neg_row[train_neg_index], neg_col[train_neg_index], np.zeros_like(train_neg_index)])
            train_interaction_indices = np.hstack([train_pos_interaction, train_neg_interaction]).T

            test_pos_interaction = np.vstack([pos_row[test_index], pos_col[test_index], np.ones_like(test_index)])
            if equal_test:
                test_neg_interaction = np.vstack([neg_row[neg_index[test_index]], neg_col[neg_index[test_index]], np.zeros_like(test_index)])
            else:
                test_neg_interaction = np.vstack([neg_row[test_neg_index], neg_col[test_neg_index], np.zeros_like(test_neg_index)])
            test_interaction_indices = np.hstack([test_pos_interaction, test_neg_interaction]).T
            yield train_interaction_indices, test_interaction_indices

    def leave_one_split(self, by_row=True, seed=666, equal_train=False, equal_test=False):
        interaction = self.interaction if by_row else self.interaction.T
        np.random.seed(seed)
        pos_row, pos_col = np.where(interaction>0.5)
        neg_row, neg_col = np.where(interaction<0.5)
        for i, row in enumerate(interaction):
            test_pos_mask = pos_row==i
            test_neg_mask = neg_row==i
            neg_index = np.random.permutation(np.arange(test_neg_mask.sum()))[:test_pos_mask.sum()]
            test_pos_interaction = np.vstack([pos_row[test_pos_mask], pos_col[test_pos_mask], np.ones(test_pos_mask.sum(), dtype=int)])
            if equal_test:
                test_neg_interaction = np.vstack([neg_row[test_neg_mask][neg_index], neg_col[test_neg_mask][neg_index], np.zeros(test_pos_mask.sum(), dtype=int)])
            else:
                test_neg_interaction = np.vstack([neg_row[test_neg_mask], neg_col[test_neg_mask], np.zeros(test_neg_mask.sum(), dtype=int)])
            test_interaction_indices = np.hstack([test_pos_interaction, test_neg_interaction]).T

            train_pos_mask = pos_row!=i
            train_neg_mask = neg_row!=i
            neg_index = np.random.permutation(np.arange(train_neg_mask.sum()))[:train_pos_mask.sum()]
            train_pos_interaction = np.vstack([pos_row[train_pos_mask], pos_col[train_pos_mask], np.ones(train_pos_mask.sum(), dtype=int)])
            if equal_train:
                train_neg_interaction = np.vstack([neg_row[train_neg_mask][neg_index], neg_col[train_neg_mask][neg_index], np.zeros(train_pos_mask.sum(), dtype=int)])
            else:
                train_neg_interaction = np.vstack([neg_row[train_neg_mask], neg_col[train_neg_mask], np.zeros(train_neg_mask.sum(), dtype=int)])
            train_interaction_indices = np.hstack([train_pos_interaction, train_neg_interaction]).T

            if not by_row:
                train_interaction_indices = train_interaction_indices[:, (1, 0, 2)]
                test_interaction_indices = test_interaction_indices[:, (1, 0, 2)]
            yield (train_interaction_indices, test_interaction_indices)

    @staticmethod
    def add_argparse_args(parent_parser):
        parent_parser.add_argument("--dataset_name", default="HMDAD", type=str, choices=["HMDAD", "Disbiome", "Combined"])
        parent_parser.add_argument("--seed", default=666, type=int)
        parent_parser.add_argument("--mode", default="CVS1", type=str, choices=["CVS1", "CVS2", "CVS3"])
        parent_parser.add_argument("--n_splits", default=5, type=int)
        parent_parser.add_argument("--equal_train_num", default=False, action="store_true")
        parent_parser.add_argument("--equal_test_num", default=False, action="store_true")
        return parent_parser


class BaseModel():
    def __init__(self, **config):
        self.config = config

    def fit_transform(self, train_indices, test_indices, d_feature, m_feature, shape):
        raise NotImplemented

    def extract_edge_from(self, matrix, indices):
        value = matrix[(indices[:, 0], indices[:, 1])]
        edges = np.vstack([indices[:, 0], indices[:, 1], value]).T
        return edges


class Experiment():
    DEFAULT_DIR = "experiment_result"
    def __init__(self, **kwargs):
        self.config = kwargs
        self.dataset = Dataset(**kwargs)
        self.sub_dir = self.dataset.sub_dir
        self.save_dir = os.path.join(self.DEFAULT_DIR, self.sub_dir)

    @staticmethod
    def add_argparse_args(parent_parser):
        parent_parser = Dataset.add_argparse_args(parent_parser)
        parent_parser.add_argument("--noise_rate", default=None, type=float)
        return parent_parser

    @classmethod
    def extract_value_from(cls, matrix, indices):
        return indices[:, :-1], indices[:, -1], matrix[(indices[:, 0], indices[:, 1])]

    @classmethod
    def GATMDA_extract_value_from(cls, matrix, train_indices, test_indices):
        # 负样本从完整的关联里采样
        indices = np.concatenate([train_indices, test_indices])
        interaction = scipy.sparse.coo_matrix((indices[:,-1], (indices[:,0], indices[:, 1])), shape=matrix.shape).todense()
        row, col = np.where(interaction<0.5)
        neg_index = np.random.permutation(np.arange(len(row)))[:test_indices[:, 2].sum()]
        neg_indices = np.vstack([row[neg_index], col[neg_index], np.zeros_like(neg_index)]).T
        neg_score = matrix[(neg_indices[:, 0], neg_indices[:, 1])]
        pos_indices = test_indices[test_indices[:, -1]>0.5]
        pos_score = matrix[(pos_indices[:,0], pos_indices[:, 1])]
        score = np.concatenate([pos_score, neg_score])
        label_indices = np.concatenate([pos_indices, neg_indices])
        return label_indices[:, :-1], label_indices[:, -1], score

    @classmethod
    def evaluate(cls, score_matrix, train_indices, test_indices):
        indices, label, score = cls.GATMDA_extract_value_from(score_matrix, train_indices, test_indices)
        sample_aupr = metrics.average_precision_score(y_true=label, y_score=score)
        sample_auroc = metrics.roc_auc_score(y_true=label, y_score=score)
        indices, label, score = cls.extract_value_from(score_matrix, test_indices)
        aupr = metrics.average_precision_score(y_true=label, y_score=score)
        auroc = metrics.roc_auc_score(y_true=label, y_score=score)
        return {"auroc": auroc,
                "aupr": aupr,
                "sample_auroc": sample_auroc,
                "sample_aupr": sample_aupr}


    def run(self, model_cls, comment=None, debug=False):
        model_name = model_cls.__name__
        comment = model_name if comment is None else comment
        noise_rate = self.config["noise_rate"]
        if noise_rate is not None:
            comment = f"{comment}-noisy_{noise_rate}"
        root_dir = os.path.join("experiment_result", comment, model_name)
        save_dir = os.path.join(root_dir, self.sub_dir)
        print(f"begin {model_name} experiment:{comment} in {save_dir}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        start_time_stamp = time.time()
        count = 0
        metrics_total = []
        for i, (old_train_indices, test_indices, d_feature, m_feature, shape) in enumerate(self.dataset.split()):
            start_time = time.time()
            model = model_cls(**self.config)
            if noise_rate is not None:
                train_indices = self.add_noise(old_train_indices, noise_rate)
            else:
                train_indices = old_train_indices
            score_matrix = model.fit_transform(train_indices, test_indices, d_feature, m_feature, shape)
            metrics_info = self.evaluate(score_matrix, old_train_indices, test_indices)
            end_time = time.time()

            result_name = f'roc_{metrics_info["auroc"]:.5f}-pr_{metrics_info["aupr"]:.5f}-split_{i:05d}'
            message = ", ".join([f"{key}={value:.5f}" for key, value in metrics_info.items()])
            print(f"split {i:05d}: {message}, time cost:{end_time - start_time}s")
            print("=" * 80)
            np.savetxt(os.path.join(save_dir, f"{result_name}_score.txt"), score_matrix)
            metrics_total.append(metrics_info)
            count += 1
            if debug:
                break
        end_time_stamp = time.time()

        metrics_info = {key: [item[key] for item in metrics_total] for key in metrics_total[0]}
        message = ", ".join([f"{key}={np.mean(value):.5f}" for key, value in metrics_info.items()])
        print(f"average: {message} time cost:{end_time_stamp-start_time_stamp}s")
        print("********************************************************************************")
        result_name = f'roc_{np.mean(metrics_info["auroc"]):.5f}-pr_{np.mean(metrics_info["aupr"]):.5f}'

        metrics_data = pd.DataFrame(metrics_info)
        metrics_data["model"] = model_name
        metrics_data["seed"] = self.config["seed"]
        metrics_data["mode"] = self.config["mode"]
        metrics_data["fold_idx"] = np.arange(count)
        if noise_rate is not None:
            metrics_data["noise_rate"] = noise_rate
        metrics_data.to_excel(os.path.join(save_dir, f"{result_name}.xlsx"), index=False)
        with open(os.path.join(save_dir, f"{result_name}_params.json"), "w") as f:
            json.dump(self.config, f)


    def add_noise(self, indices, noise_rate=0.1, seed=777):
        new_indices = np.copy(indices)
        np.random.seed(seed)
        row, col = np.where(indices<0.5)
        index = np.random.permutation(np.arange(len(row)))[:int(len(row)*noise_rate)]
        new_indices[(row[index], col[index])] = 1
        # row, col = np.where(indices>0.5)
        # index = np.random.permutation(np.arange(len(row)))[:int(len(row)*noise_rate)]
        # new_indices[(row[index], col[index])] = 0
        return new_indices

    def noise_run(self, model_cls, comment=None, noise_rate=0.1):
        model_name = model_cls.__name__
        comment = model_name if comment is None else comment
        root_dir = os.path.join("experiment_result", comment, model_name)
        save_dir = os.path.join(root_dir, self.sub_dir)
        print(f"begin {model_name} noise {noise_rate} experiment:{comment} in {save_dir}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        start_time_stamp = time.time()
        count = 0
        metrics_total = []
        for i, (old_train_indices, test_indices, d_feature, m_feature, shape) in enumerate(self.dataset.split()):
            train_indices = self.add_noise(old_train_indices, noise_rate)
            start_time = time.time()
            model = model_cls(**self.config)
            score_matrix = model.fit_transform(train_indices, test_indices, d_feature, m_feature, shape)
            metrics_info = self.evaluate(score_matrix, old_train_indices, test_indices)
            end_time = time.time()

            result_name = f'roc_{metrics_info["auroc"]:.5f}-pr_{metrics_info["aupr"]:.5f}-split_{i:05d}'
            message = ", ".join([f"{key}={value:.5f}" for key, value in metrics_info.items()])
            print(f"split {i:05d}: {message}, time cost:{end_time - start_time}s")
            print("=" * 80)
            np.savetxt(os.path.join(save_dir, f"{result_name}_score.txt"), score_matrix)
            metrics_total.append(metrics_info)
            count += 1
        end_time_stamp = time.time()

        metrics_info = {key: [item[key] for item in metrics_total] for key in metrics_total[0]}
        message = ", ".join([f"{key}={np.mean(value):.5f}" for key, value in metrics_info.items()])
        print(f"average: {message} time cost:{end_time_stamp-start_time_stamp}s")
        print("********************************************************************************")
        result_name = f'roc_{np.mean(metrics_info["auroc"]):.5f}-pr_{np.mean(metrics_info["aupr"]):.5f}'

        metrics_data = pd.DataFrame(metrics_info)
        metrics_data["model"] = model_name
        metrics_data["seed"] = self.config["seed"]
        metrics_data["mode"] = self.config["mode"]
        metrics_data["fold_idx"] = np.arange(count)
        metrics_data["noise_rate"] = noise_rate
        metrics_data.to_excel(os.path.join(save_dir, f"{result_name}.xlsx"), index=False)
        with open(os.path.join(save_dir, f"{result_name}_params.json"), "w") as f:
            json.dump(self.config, f)

    @classmethod
    def collect_result(cls, save_dir=None, tag=""):
        value_cols = ["auroc", "aupr", "sample_auroc", "sample_aupr"]
        save_dir = save_dir if save_dir is not None else cls.DEFAULT_DIR
        metric_files = defaultdict(dict)
        for comment in os.listdir(save_dir):
            if not os.path.isdir(os.path.join(save_dir, comment)):
                continue
            for model in os.listdir(os.path.join(save_dir, comment)):
                for dataset in os.listdir(os.path.join(save_dir, comment, model)):
                    for exp in os.listdir(os.path.join(save_dir, comment, model, dataset)):
                        CV = exp.split("-")[0]
                        metric_files[dataset][CV] = metric_files[dataset].get(CV, [])
                        result_dir = os.path.join(os.path.join(save_dir, comment, model, dataset, exp))
                        files = os.listdir(result_dir)
                        for file in files:
                            if file.startswith("roc_") and file.endswith(".xlsx"):
                                metric_file = os.path.join(result_dir, file)
                                metric_files[dataset][CV].append((comment, model, metric_file))


        for dataset in metric_files:
            with pd.ExcelWriter(os.path.join(save_dir, f"merged_{dataset}_{tag}.xlsx")) as f:
                ans = []
                for CV in metric_files[dataset]:
                    merged_data = []
                    CV_info = []
                    for comment, model, file in metric_files[dataset][CV]:
                        data = pd.read_excel(file)
                        data["dataset"] = dataset
                        data["comment"] = comment
                        data["model"] = model
                        metric_value = data[value_cols].values.mean(axis=0)
                        other_cols = data.columns.difference(value_cols)
                        info = {key:value for key, value in zip(value_cols, metric_value)}
                        for key in other_cols:
                            info[key] = data[key].iloc[0]
                        info.pop("fold_idx")
                        CV_info.append(info)
                        merged_data.append(data)
                    merged_data = pd.concat(merged_data).sort_values(["dataset", "mode", "model", "seed"])
                    merged_data.to_excel(f, sheet_name=CV, index=False)
                    CV_info = pd.DataFrame(CV_info).sort_values(["dataset", "mode", "model", "seed"])
                    CV_info.to_excel(f, sheet_name=f"summary_{CV}", index=False)
                    ans.append(CV_info)

                res = []
                ans = pd.concat(ans)
                for model in ans["model"].unique():
                    for mode in ans['mode'].unique():
                        data = ans[(ans["model"]==model) & (ans['mode']==mode)]
                        if len(data)==0:
                            continue
                        metric_value = data[value_cols].values.mean(axis=0)
                        other_cols = data.columns.difference(value_cols)
                        info = {key:value for key, value in zip(value_cols, metric_value)}
                        for key in other_cols:
                            info[key] = data[key].iloc[0]
                        info.pop("seed")
                        res.append(info)
                res = pd.DataFrame(res).sort_values(["dataset", "mode", "model"])
                res.to_excel(f, sheet_name="summary", index=False)