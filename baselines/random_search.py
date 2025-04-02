import random
random.seed(42)
import numpy as np
import xgboost as xgb
import os
import json
from data.benchmarks.hpob.search_spaces_info import SEARCH_SPACE_INFO
from utils.utils import get_project_root
from baselines.baseline import Baseline
class RandomSearch(Baseline):

    def __init__(self):
        super().__init__()
        self.name = f"Random"

    def run_method(self, seed_id, search_space_id, dataset_id):
        # 加载评估代理模型
        surrogate_name = 'surrogate-' + search_space_id + '-' + dataset_id
        bst_surrogate = xgb.Booster()
        bst_surrogate.load_model(self.surrogates_dir + surrogate_name + '.json')

        search_space = SEARCH_SPACE_INFO[search_space_id]
        dim = len(search_space["parameters_name"])
        y_min = self.surrogates_stats[surrogate_name]["y_min"]
        y_max = self.surrogates_stats[surrogate_name]["y_max"]

        x_observed = []
        y_observed = []
        max_accuracy_history = []

        for i in range(self.n_trials):

            # 获取下一个要探索的超参数配置
            new_x = self.random_search(search_space)

            new_x_arry = np.array(new_x)
            # 评估超参数配置
            x_q = xgb.DMatrix(new_x_arry.reshape(-1, dim))
            new_y_arry = bst_surrogate.predict(x_q)
            new_y = float(new_y_arry[0])

            y_observed.append(new_y)
            x_observed.append(new_x)

            best_f = max(y_observed)
            # 记录当前最优的分数
            max_accuracy_history.append(best_f)

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

    def random_search(self, search_space):
        parameters_name = search_space["parameters_name"] #获取的参数列表

        x_new = []
        for parameter_name in parameters_name:
            if "float" in search_space[parameter_name]['type']:
                low = search_space[parameter_name]['low']
                high = search_space[parameter_name]['high']
                random_parameter = round(random.uniform(low, high), 4)
            elif "int" in search_space[parameter_name]['type']:
                low = search_space[parameter_name]['low']
                high = search_space[parameter_name]['high']
                random_parameter = random.randint(low, high)
            else:
                categories = search_space[parameter_name]['categories']
                random_parameter = random.choice(categories)
            x_new.append(random_parameter)
        return x_new

    def run_hpo_bench_method(self, seed_id, model, dataset_name):
        from data.benchmarks.hpo_bench.tabular_benchmarks import DATASET_MAP, HPOExpRunner
        dataset = DATASET_MAP[dataset_name][0]
        benchmark = HPOExpRunner(model, dataset, seed_id)
        with open(os.path.join(get_project_root(), r"data/benchmarks/hpo_bench/hpo_bench.json"), 'r') as f:
            search_space = json.load(f)[model]

        x_observed = []
        y_observed = []
        max_accuracy_history = []

        for i in range(self.n_trials):

            # 获取下一个要探索的超参数配置
            new_x = self.random_search_hpo_bench(search_space)

            # 评估超参数配置
            new_y = benchmark.evaluate_point(new_x)

            # 记录观察到的超参数和分数
            y_observed.append(new_y)
            x_observed.append([new_x[key] for key in new_x])

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

    def random_search_hpo_bench(self, search_space):
        x_new = {}
        for parameter_name in search_space:
            categories = search_space[parameter_name]
            random_parameter = random.choice(categories)
            x_new[parameter_name] = random_parameter
        return x_new

    def run_nl2workflow_method(self, seed_id, model_name, dataset_name):
        from data.benchmarks.nl2workflow.eval import evaluate
        with open(os.path.join(get_project_root(), "data/benchmarks/nl2workflow/search_space.json"), 'r', encoding='utf-8') as f:
            search_space = json.load(f)
        for s in search_space:
            if s["algorithm"] == model_name:
                search_space = s["search_space"]
                break
        x_observed = []
        y_observed = []
        max_accuracy_history = []
        for i in range(self.n_trials):
            # 获取下一个要探索的超参数配置
            new_x = self.random_search_nl2workflow(search_space)
            # 评估超参数配置
            new_y = evaluate(new_x, model_name, dataset_name)
            # 记录观察到的超参数和分数
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

    def random_search_nl2workflow(self, search_space):
        parameters_name = [key for key in search_space]
        x_new = []
        for parameter_name in parameters_name:
            if search_space[parameter_name]['_type'] == "uniform":
                low = search_space[parameter_name]['_value'][0]
                high = search_space[parameter_name]['_value'][1]
                random_parameter = round(random.uniform(low, high), 4)
            elif search_space[parameter_name]['_type']=="randint":
                low = search_space[parameter_name]['_value'][0]
                high = search_space[parameter_name]['_value'][1]
                random_parameter = random.randint(low, high)
            elif search_space[parameter_name]["_type"] == "loguniform":
                low = search_space[parameter_name]['_value'][0]
                high = search_space[parameter_name]['_value'][1]
                random_parameter = random.uniform(low, high)
            elif search_space[parameter_name]["_type"] == "choice":
                categories = search_space[parameter_name]['_value']
                random_parameter = random.choice(categories)
            else:
                raise ValueError("Invalid parameter type")
            x_new.append(random_parameter)
        return x_new

if __name__ == '__main__':
    random_search = RandomSearch()
    random_search.run_nl2workflow_experiment()
    # random_search.run_hpo_bench_experiment()
    # random_search.run_hpob_experiment()