import random
random.seed(42)
import os
import json
from utils.utils import get_project_root
from baselines.baseline import Baseline
from nni.experiment import Experiment
import sys
import time

class NNI(Baseline):

    def __init__(self):
        super().__init__()
        # Bayesian：TPE，Heuristic：Evolution
        self.type = "Evolution"
        self.name = f"NNI_{self.type}"

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
        x, y = self.run_nni_experiment()
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
        from data.benchmarks.hpo_bench.tabular_benchmarks import DATASET_MAP, HPOExpRunner
        dataset = DATASET_MAP[dataset_name][0]
        benchmark = HPOExpRunner(model, dataset, seed_id)
        with open(os.path.join(get_project_root(), r"data/benchmarks/hpo_bench/hpo_bench.json"), 'r') as f:
            search_space = json.load(f)[model]

        x_observed = []
        y_observed = []
        max_accuracy_history = []
        self.exe_params = {
            "model_name": model,
            "dataset": dataset_name,
            "seed_id": seed_id,
            "benchmark_name": "hpo_bench"
        }
        x, y = self.run_nni_experiment()
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
        x, y = self.run_nni_experiment()
        for i in range(len(x)):
            new_x = x[i]
            new_y = y[i]
            y_observed.append(new_y)
            x_observed.append(new_x)
            # 记录当前最优的分数
            max_accuracy_history.append(max(y_observed))

        if len(max_accuracy_history) < self.n_trials:
            max_accuracy_history += [max_accuracy_history[-1]] * (self.n_trials - len(max_accuracy_history))
        record = {
            "search_space": model_name,
            "dataset": dataset_name,
            "seed": seed_id,
            "x_observed": x_observed,
            "y_observed": y_observed,
            "max_accuracy_history": max_accuracy_history
        }
        return record

    def run_nni_experiment(self):
        search_space = self.get_nni_search_space(self.exe_params["model_name"], self.exe_params["benchmark_name"])
        _, result = self.run_experiment(search_space)
        xy = [{r.trialJobId: (r.parameter, r.value)} for r in result]
        x = []
        y = []
        for r in xy:
            for key, value in r.items():
                x.append([value[0][key] for key in value[0]])
                y.append(value[1])
        print(f"y: {y}")
        print(f"x: {x}")
        return x, y

    def get_nni_search_space(self, model_name, benchmark):
        if benchmark == "hpob":
            from data.benchmarks.hpob.search_spaces_info import SEARCH_SPACE_INFO
            search_space = SEARCH_SPACE_INFO[model_name]
            search_space_new = {}
            for p in search_space["parameters_name"]:
                if search_space[p]["type"] == "categorical":
                    search_space_new[p] = {
                        "_type": "choice",
                        "_value": search_space[p]["categories"]
                    }
                elif "float" in search_space[p]['type']:
                    search_space_new[p] = {
                        "_type": "uniform",
                        "_value": [search_space[p]["low"], search_space[p]["high"]]
                    }
                elif "int" in search_space[p]['type']:
                    search_space_new[p] = {
                        "_type": "randint",
                        "_value": [search_space[p]["low"], search_space[p]["high"]]
                    }
                else:
                    raise ValueError(f"unknown type {search_space[p]['type']}")

        elif benchmark == "hpo_bench":
            with open(os.path.join(get_project_root(), r'data/benchmarks/hpo_bench/hpo_bench.json'), 'r') as f:
                search_space = json.load(f)[model_name]
            search_space_new = {}
            for s in search_space:
                search_space_new[s] = {
                    "_type": "choice",
                    "_value": search_space[s]
                }
        else:
            with open(os.path.join(get_project_root(), "data/benchmarks/nl2workflow/search_space.json"), 'r',
                      encoding='utf-8') as f:
                search_space = json.load(f)
            search_space_new = {}
            for s in search_space:
                if s["algorithm"] == model_name:
                    search_space_new = s["search_space"]
                    break
        return search_space_new

    def gen_port(self):
        import socket
        sock = socket.socket()
        sock.bind(("", 0))
        ip, port = sock.getsockname()
        sock.close()
        return port

    def run_experiment(self, search_space, t='30m'):
        experiment = Experiment('local')
        experiment.config.max_experiment_duration = t
        experiment.config.trial_code_directory = os.path.join(get_project_root(),
                                                              r'baselines/nni_results/trial_code_directory')
        if self.exe_params["benchmark_name"] in ["hpob", "hpo_bench"]:
            experiment.config.trial_command = sys.executable + ' ' + f'nni_evaluator.py ' \
                                                                     f'--model_name {self.exe_params["model_name"]} ' \
                                                                     f'--dataset {self.exe_params["dataset"]} ' \
                                                                     f'--seed_id {self.exe_params["seed_id"]} ' \
                                                                     f'--benchmark_name {self.exe_params["benchmark_name"]}'
        else:
            experiment.config.trial_command = sys.executable + ' ' + f'nni_evaluator.py ' \
                                                                     f'--model_name "{self.exe_params["model_name"]}" ' \
                                                                     f'--dataset "{self.exe_params["dataset"]}"'

        experiment.config.tuner.name = self.type
        if self.type == "TPE":
            experiment.config.tuner.class_args['seed'] = 42
        experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
        experiment.config.experiment_working_directory = os.path.join(get_project_root(), r'baselines/nni_results/logs')
        experiment.config.trial_concurrency = 5
        experiment.config.max_trial_number = self.n_trials
        experiment.config.search_space = search_space
        wait_completion = True
        experiment.start(self.gen_port())
        if wait_completion:
            try:
                while True:
                    time.sleep(10)
                    status = experiment.get_status()
                    print(status)
                    if status == 'DONE' or status == 'STOPPED':
                        result = experiment.export_data()
                        best_result = max(result, key=lambda x: x.value)
                        return best_result, result
                    if status == 'ERROR':
                        return None, None
            except KeyboardInterrupt:
                print('KeyboardInterrupt detected')
            finally:
                experiment.stop()
if __name__ == '__main__':
    random_search = NNI()
    # random_search.run_hpo_bench_experiment()
    # random_search.run_hpob_experiment()
    random_search.run_nl2workflow_experiment()