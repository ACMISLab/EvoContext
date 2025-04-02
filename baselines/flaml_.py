import random
random.seed(42)
import numpy as np
import os
import json
from utils.utils import get_project_root
from baselines.baseline import Baseline
from flaml import tune
import xgboost as xgb

class FLAML(Baseline):

    def __init__(self):
        super().__init__()
        # Bayesian：TPE，Heuristic：Evolution
        self.name = f"FLAML"

    def run_method(self, seed_id, search_space_id, dataset_id):
        # 加载评估代理模型
        surrogate_name = 'surrogate-' + search_space_id + '-' + dataset_id
        y_min = self.surrogates_stats[surrogate_name]["y_min"]
        y_max = self.surrogates_stats[surrogate_name]["y_max"]
        x_observed = []
        y_observed = []
        max_accuracy_history = []
        self.exe_params = {
            "model_name": search_space_id,
            "dataset": dataset_id,
            "seed_id": seed_id,
            "benchmark_name": "hpob"
        }
        x, y = self.run_experiment()
        for i in range(self.n_trials):
            new_x = x[i]
            new_y = y[i]
            y_observed.append(new_y)
            x_observed.append(new_x)
            # 记录当前最优的分数
            max_accuracy_history.append(max(y_observed))
        record = {
            "search_space": search_space_id,
            "dataset": dataset_id,
            "seed": seed_id,
            "x_observed": x_observed,
            "y_observed": y_observed,
            "max_accuracy_history": max_accuracy_history,
            "y_min": y_min,
            "y_max": y_max
        }
        return record

    def run_hpo_bench_method(self, seed_id, model, dataset_name):
        x_observed = []
        y_observed = []
        max_accuracy_history = []
        self.exe_params = {
            "model_name": model,
            "dataset": dataset_name,
            "seed_id": seed_id,
            "benchmark_name": "hpo_bench"
        }
        x, y = self.run_experiment()
        for i in range(self.n_trials):
            new_x = x[i]
            new_y = y[i]
            y_observed.append(new_y)
            x_observed.append(new_x)
            # 记录当前最优的分数
            max_accuracy_history.append(max(y_observed))

        record = {
            "search_space": model,
            "dataset": dataset_name,
            "seed": seed_id,
            "x_observed": x_observed,
            "y_observed": y_observed,
            "max_accuracy_history": max_accuracy_history
        }

        return record

    def run_nl2workflow_method(self, seed_id, model_name, dataset_name):
        x_observed = []
        y_observed = []
        max_accuracy_history = []
        self.exe_params = {
            "model_name": model_name,
            "dataset": dataset_name,
            "benchmark_name": "nl2workflow"
        }
        x, y = self.run_experiment()
        for i in range(self.n_trials):
            new_x = x[i]
            new_y = y[i]
            y_observed.append(new_y)
            x_observed.append(new_x)
            # 记录当前最优的分数
            max_accuracy_history.append(max(y_observed))

        record = {
            "search_space": model_name,
            "dataset": dataset_name,
            "seed": seed_id,
            "x_observed": x_observed,
            "y_observed": y_observed,
            "max_accuracy_history": max_accuracy_history
        }
        return record

    def eval(self, params):
        config = [params[key] for key in params]
        if self.exe_params["benchmark_name"] == "hpob":
            bst_surrogate = xgb.Booster()
            surrogates_dir = os.path.join(get_project_root(), f"data/benchmarks/hpob/saved-surrogates")
            bst_surrogate.load_model(os.path.join(surrogates_dir, f'surrogate-{self.exe_params["model_name"]}-{self.exe_params["dataset"]}.json'))
            new_x_arry = np.array(config)
            x_q = xgb.DMatrix(new_x_arry.reshape(-1, len(config)))
            new_y_arry = bst_surrogate.predict(x_q)
            new_y = float(new_y_arry[0])
        elif self.exe_params["benchmark_name"] == "hpo_bench":
            from data.benchmarks.hpo_bench.tabular_benchmarks import DATASET_MAP, HPOExpRunner
            dataset = DATASET_MAP[self.exe_params["dataset"]][0]
            benchmark = HPOExpRunner(self.exe_params["model_name"], dataset, self.exe_params["seed_id"])
            new_x_pro = {benchmark.hpo_bench.order_list[i]: config[i] for i in range(len(config))}
            new_y = benchmark.evaluate_point(new_x_pro)
        else:
            from data.benchmarks.nl2workflow.eval import evaluate
            new_y = evaluate(config, self.exe_params["model_name"], self.exe_params["dataset"])

        return {"score": new_y}

    def run_experiment(self):
        search_space = self.get_flaml_search_space(self.exe_params["model_name"], self.exe_params["benchmark_name"])
        analysis = tune.run(
            self.eval,
            metric="score",
            mode="max",
            config=search_space,
            num_samples=self.n_trials,
            verbose=0,
        )
        x = []
        y = []
        assert len(analysis.results)==self.n_trials
        for r in analysis.results:
            x.append([analysis.results[r]["config"][key] for key in analysis.results[r]["config"]])
            y.append(analysis.results[r]["score"])
        print(f"y: {y}")
        print(f"x: {x}")
        return x, y

    def get_flaml_search_space(self, model_name, benchmark):
        # tune.randint(lower=1, upper=1000000)
        # tune.choice([1, 2, 3])
        # tune.uniform(lower=0, upper=1)
        if benchmark == "hpob":
            from data.benchmarks.hpob.search_spaces_info import SEARCH_SPACE_INFO
            search_space = SEARCH_SPACE_INFO[model_name]
            search_space_new = {}
            for p in search_space["parameters_name"]:
                if search_space[p]["type"] == "categorical":
                    search_space_new[p] = tune.choice(search_space[p]["categories"])
                elif "float" in search_space[p]['type']:
                    search_space_new[p] = tune.uniform(search_space[p]["low"], search_space[p]["high"])
                elif "int" in search_space[p]['type']:
                    search_space_new[p] = tune.randint(search_space[p]["low"], search_space[p]["high"])
                else:
                    raise ValueError(f"unknown type {search_space[p]['type']}")

        elif benchmark == "hpo_bench":
            with open(os.path.join(get_project_root(), r'data/benchmarks/hpo_bench/hpo_bench.json'), 'r') as f:
                search_space = json.load(f)[model_name]
            search_space_new = {}
            for s in search_space:
                search_space_new[s] = tune.choice(search_space[s])
        else:
            with open(os.path.join(get_project_root(), "data/benchmarks/nl2workflow/search_space.json"), 'r',
                      encoding='utf-8') as f:
                search_space = json.load(f)
            search_space_new = {}
            for s in search_space:
                if s["algorithm"] == model_name:
                    search_space = s["search_space"]
                    break

            for p in search_space:
                if search_space[p]["_type"] == "choice":
                    search_space_new[p] = tune.choice(search_space[p]["_value"])
                elif search_space[p]["_type"] == "uniform":
                    search_space_new[p] = tune.uniform(search_space[p]["_value"][0], search_space[p]["_value"][1])
                elif search_space[p]["_type"] == "randint":
                    search_space_new[p] = tune.randint(search_space[p]["_value"][0], search_space[p]["_value"][1])
                elif search_space[p]["_type"] == "loguniform":
                    search_space_new[p] = tune.loguniform(search_space[p]["_value"][0], search_space[p]["_value"][1])
                else:
                    raise ValueError(f"unknown type {search_space[p]['type']}")

        return search_space_new
if __name__ == '__main__':
    random_search = FLAML()
    random_search.run_nl2workflow_experiment()
    # random_search.run_hpo_bench_experiment()
    # random_search.run_hpob_experiment()