import json
import os
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Final, List, Optional, Union

import openml
import pandas as pd
import ConfigSpace as CS

import numpy as np
import pyarrow.parquet as pq  # type: ignore
import time

from utils.utils import get_project_root

"""数据下载链接
xgb="https://ndownloader.figshare.com/files/30469920",

rf="https://ndownloader.figshare.com/files/30469089",

nn="https://ndownloader.figshare.com/files/30379005"

svm="https://ndownloader.figshare.com/files/30379359",

lr="https://ndownloader.figshare.com/files/30379038",
"""

DATA_DIR_NAME = os.path.join(get_project_root(), r"data/benchmarks/hpo_bench/data")
SEEDS: Final = [665, 1319, 7222, 7541, 8916]
VALUE_RANGES = json.load(open(os.path.join(get_project_root(), r"data/benchmarks/hpo_bench/hpo_bench.json")))

# [dataset id, openml task id]

DATASET_MAP = {
    "credit_g": [0, 31],
    "vehicle": [1, 53],
    "kc1": [2, 3917],
    "phoneme": [3, 9952],
    "blood_transfusion": [4, 10101],
    "australian": [5, 146818],
    "car": [6, 146821],
    "segment": [7, 146822],
}

MODEL_MAP = {
    'rf': 'Random Forest',
    'nn': 'Multilayer Perceptron',
    'xgb': 'XGBoost',
    "svm": 'SVM',
    "lr": "Logistic Regression"
}

class AbstractBench(metaclass=ABCMeta):
    _rng: np.random.RandomState
    _value_range: Dict[str, List[Union[int, float, str]]]
    dataset_name: str

    def __init__(self):
        self.all_results = []

    def reset_results(self):
        self.all_results = []

    def add_config(self, config):
        self.all_results += [config]

    def reseed(self, seed: int) -> None:
        self._rng = np.random.RandomState(seed)

    def _fetch_discrete_config_space(self) -> CS.ConfigurationSpace:
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameters(
            [
                CS.UniformIntegerHyperparameter(name=name, lower=0, upper=len(choices) - 1)
                if not isinstance(choices[0], (str, bool))
                else CS.CategoricalHyperparameter(name=name, choices=[str(i) for i in range(len(choices))])
                for name, choices in self._value_range.items()
            ]
        )
        return config_space

    @property
    @abstractmethod
    def config_space(self) -> CS.ConfigurationSpace:
        raise NotImplementedError

class HPOBench(AbstractBench):
    def __init__(
        self,
        model_name: str,
        dataset_id: int
    ):
        super() .__init__()
        # https://ndownloader.figshare.com/files/30379005
        dataset_info = [
            ("credit_g", 31),
            ("vehicle", 53),
            ("kc1", 3917),
            ("phoneme", 9952),
            ("blood_transfusion", 10101),
            ("australian", 146818),
            ("car", 146821),
            ("segment", 146822),
        ]
        order_list_info = {
            "nn": ["alpha", "batch_size", "depth", "learning_rate_init", "width"],
            "rf": ["max_depth", "max_features", "min_samples_leaf", "min_samples_split"],
            "xgb": ["colsample_bytree", "eta", "max_depth", "reg_lambda"],
            "svm": ["C", "gamma"],
            "lr": ["alpha", "eta0"]
        }
        budget_tuple_info = {
            "nn": ("iter", 243),
            "rf": ("n_estimators", 512),
            "xgb": ("n_estimators", 2000),
            "lr": ("iter", 1000)
        }

        # self.num_0 = 0
        self.model_name = model_name
        dataset_name, dataset_id = dataset_info[dataset_id]
        self.dataset_name = '%s_%s' % (dataset_name, model_name)
        self.order_list  = order_list_info[model_name]
        if model_name=="svm":
            data_path = os.path.join(DATA_DIR_NAME, "%s" % model_name, str(dataset_id),
                                     f"{model_name}_{dataset_id}_data.parquet.gzip")
            db = pd.read_parquet(data_path, filters=[('subsample', "==", 1.0)])
            self._db = db.drop(["subsample"], axis=1)
        else:
            budget_name, budget_value = budget_tuple_info[model_name]
            data_path = os.path.join(DATA_DIR_NAME, "%s" % model_name, str(dataset_id),
                                     f"{model_name}_{dataset_id}_data.parquet.gzip")
            db = pd.read_parquet(data_path, filters=[(budget_name, "==", budget_value), ('subsample', "==", 1.0)])
            self._db = db.drop([budget_name, "subsample"], axis=1)
        self._value_range = VALUE_RANGES[f"{model_name}"]

    def ordinal_to_real(self, config):
        if self.model_name in ["lr", "svm"]:
            return config
        return {key: self._value_range[key][config[key]] for key in config.keys()}

    def _search_dataframe(self, row_dict, df):
        # https://stackoverflow.com/a/46165056/8363967
        mask = np.array([True] * df.shape[0])

        for i, param in enumerate(df.drop(columns=["result"], axis=1).columns):
            mask *= df[param].values == row_dict[param]
        idx = np.where(mask)
        assert len(idx) == 1, 'The query has resulted into mulitple matches. This should not happen. ' \
                              f'The Query was {row_dict}'
        idx = idx[0][0]
        result = df.iloc[idx]["result"]
        return result
    
    def complete_call(self, config):
        #_config = config.copy()
        key_path = config.copy()
        loss = []
        test_info = {}
        time_init = time.time()
        idx = 0
        for seed in SEEDS:
            key_path["seed"] = seed
            res = self._search_dataframe(key_path, self._db)
            # loss.append(1 - res["info"]['val_scores']['acc'])
            loss.append(res["info"]['val_scores']['acc'])
            for key in res["info"]['test_scores'].keys():
                if idx == 0:
                    test_info[key]  = res["info"]['test_scores'][key]* 1.0 / len(SEEDS)
                else:
                    test_info[key] += res["info"]['test_scores'][key] * 1.0 / len(SEEDS)
            key_path.pop("seed")
            idx += 1
        loss = np.mean(loss)
        time_final = time.time()
        test_info['generalization_score'] = test_info['acc']
        test_info['time_init']  = time_init
        test_info['time_final'] = time_final 
        test_info['score'] = loss
        # return loss, test_info
        return test_info

    def call_and_add_ordinal(self, config):
        loss, _ = self.complete_call(config)
        return loss

    def __call__(self, config):
        new_config = self.ordinal_to_real(config)
        loss, test_info = self.complete_call(new_config)
        test_info.update(new_config)
        self.add_config(test_info)
        return loss

    @property
    def config_space(self) -> CS.ConfigurationSpace:
        return self._fetch_discrete_config_space()

class HPOExpRunner:

    def __init__(self, model, dataset, seed):
        self.model = model
        self.dataset = dataset
        self.hpo_bench = HPOBench(model, dataset)
        self.seed = seed
        self.config_path = os.path.join(get_project_root(), f'data/benchmarks/hpo_bench/configs/{model}/config{seed}.json')

    def generate_initialization(self, n_samples=5):
        '''
        Generate initialization points for BO search
        Args: n_samples (int)
        Returns: list of dictionaries, each dictionary is a point to be evaluated
        '''
        # load initial configs
        with open(self.config_path, 'r') as f:
            configs = json.load(f)

        assert isinstance(configs, list)
        init_configs = []
        for i, config in enumerate(configs):
            assert isinstance(config, dict)

            if i < n_samples:
                init_configs.append(self.hpo_bench.ordinal_to_real(config))

        assert len(init_configs) == n_samples

        return init_configs

    def _find_nearest_neighbor(self, config):
        discrete_grid = self.hpo_bench._value_range
        nearest_config = {}
        for key in config:
            if key in discrete_grid:
                # Find the nearest value in the grid for the current key
                nearest_value = min(discrete_grid[key], key=lambda x: abs(x - config[key]))
                nearest_config[key] = nearest_value
            else:
                raise ValueError(f"Key '{key}' not found in the discrete grid.")
        return nearest_config

    def evaluate_point(self, candidate_config):
        '''
        Evaluate a single point on bbox
        Args: candidate_config (dict), dictionary containing point to be evaluated
        Returns: (dict, dict), first dictionary is candidate_config (the evaluated point), second dictionary is fvals (the evaluation results)
        Example fval:
        fvals = {
            'score': float,
            'generalization_score': float
        }
        '''
        # dataset_info = [
        #     ("credit_g", 31),
        #     ("vehicle", 53),
        #     ("kc1", 3917),
        #     ("phoneme", 9952),
        #     ("blood_transfusion", 10101),
        #     ("australian", 146818),
        #     ("car", 146821),
        #     ("segment", 146822),
        # ]
        # config = [candidate_config[key] for key in candidate_config]
        # import xgboost as xgb
        # bst_surrogate = xgb.Booster()
        # bst_surrogate.load_model(os.path.join(surrogates_dir, f'surrogate-{self.model}-{dataset_info[self.dataset][1]}.json'))
        # new_x_arry = np.array(config)
        # x_q = xgb.DMatrix(new_x_arry.reshape(-1, len(config)))
        # new_y_arry = bst_surrogate.predict(x_q)
        # new_y = float(new_y_arry[0])
        # return new_y

        # find nearest neighbor
        nearest_config = self._find_nearest_neighbor(candidate_config)
        # evaluate nearest neighbor
        fvals = self.hpo_bench.complete_call(nearest_config)
        return float(fvals["score"])

surrogates_dir = os.path.join(get_project_root(), f"data/benchmarks/hpo_bench/saved-surrogates")

def evaluate(config, model, dataset):
    dataset = DATASET_MAP[dataset][1]
    import xgboost as xgb
    bst_surrogate = xgb.Booster()
    bst_surrogate.load_model(os.path.join(surrogates_dir, f'surrogate-{model}-{dataset}.json'))
    new_x_arry = np.array(config)
    x_q = xgb.DMatrix(new_x_arry.reshape(-1, len(config)))
    new_y_arry = bst_surrogate.predict(x_q)
    new_y = float(new_y_arry[0])
    return new_y

if __name__ == '__main__':
    dataset_name = "phoneme"
    dataset = DATASET_MAP[dataset_name][0]
    model = "lr"
    seed = 0

    # Describe task context
    task_context = {}
    task_context['model'] = MODEL_MAP[model]

    # hpo_bech datasets are all classification
    task_context['task'] = 'classification'

    task = openml.tasks.get_task(DATASET_MAP[dataset_name][1])
    dataset_ = task.get_dataset()
    X, y, categorical_mask, _ = dataset_.get_data(target=dataset_.default_target_attribute)

    task_context['tot_feats'] = X.shape[1]
    task_context['cat_feats'] = len(categorical_mask)
    task_context['num_feats'] = X.shape[1] - len(categorical_mask)
    task_context['n_classes'] = len(np.unique(y))
    task_context['metric'] = "accuracy"
    task_context['num_samples'] = X.shape[0]
    with open(os.path.join(get_project_root(), r"data/benchmarks/hpo_bench/hpo_bench.json"), 'r') as f:
        task_context['hyperparameter_constraints'] = json.load(f)[model]
    print(json.dumps(task_context, indent=4, ensure_ascii=False))
    benchmark = HPOExpRunner(model, dataset, seed)
    init_configs = benchmark.generate_initialization()
    print(json.dumps(init_configs[0], indent=4, ensure_ascii=False))
    score = benchmark.evaluate_point(init_configs[0])
    print(score)
    print(evaluate([init_configs[0][key] for key in init_configs[0]],model,DATASET_MAP[dataset_name][1]))