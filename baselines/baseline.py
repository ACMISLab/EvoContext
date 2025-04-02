import json
import os
import time
from utils.logs import logger
from data.benchmarks.hpob.datasets_info import DATASETS_INFO
from utils.utils import get_project_root
from tqdm import tqdm

SEARCH_SPACE_and_DATASETS = {"4796": ["3549", "3918", "9903", "23"], "5527": ["146064", "146065", "9914", "145804", "31", "10101"], "5636": ["146064", "145804", "9914", "146065", "10101", "31"], "5859": ["9983", "31", "37", "3902", "9977", "125923"], "5860": ["14965", "9976", "3493"], "5891": ["9889", "3899", "6566", "9980", "3891", "3492"], "5906": ["9971", "3918"], "5965": ["145836", "9914", "3903", "10101", "9889", "49", "9946"], "5970": ["37", "3492", "9952", "49", "34536", "14951"], "5971": ["10093", "3954", "43", "34536", "9970", "6566"], "6766": ["3903", "146064", "145953", "145804", "31", "10101"], "6767": ["146065", "145804", "146064", "9914", "9967", "31"], "6794": ["145804", "3", "146065", "10101", "9914", "31"], "7607": ["14965", "145976", "3896", "3913", "3903", "9946", "9967"], "7609": ["145854", "3903", "9967", "145853", "34537", "125923", "145878"], "5889": ["9971", "3918"]}

class Baseline:

    def __init__(self):
        self.name = ""
        self.llm_method = False
        self.input_tokens = 0
        self.output_tokens = 0
        self.count = 0

    def run_hpob_experiment(self, seeds=5, n_trials=15, results_path=r"experiments/results/hpob"):
        self.seeds = seeds
        self.n_trials = n_trials
        self.results = []
        self.results_path = os.path.join(get_project_root(), results_path)
        self.surrogates_dir = os.path.join(get_project_root(), r"data/benchmarks/hpob/saved-surrogates/")
        surrogates_file = self.surrogates_dir + "summary-stats.json"
        if os.path.isfile(surrogates_file):
            with open(surrogates_file) as f:
                self.surrogates_stats = json.load(f)

        initialization_data_path = os.path.join(get_project_root(), r"data/benchmarks/hpob/initializations.json")
        with open(initialization_data_path, "r") as f:
            initialization_data = json.load(f)

        # 计算总迭代次数
        total_iterations = 0
        for seed_id in range(self.seeds):
            for search_space_id in SEARCH_SPACE_and_DATASETS:
                for dataset_id in SEARCH_SPACE_and_DATASETS[search_space_id]:
                    if dataset_id not in DATASETS_INFO.keys():
                        continue
                    total_iterations += 1

        errors = []
        with tqdm(total=total_iterations) as pbar:
            begin = time.time()
            for seed_id in range(self.seeds):
                for search_space_id in SEARCH_SPACE_and_DATASETS:
                    for dataset_id in SEARCH_SPACE_and_DATASETS[search_space_id]:
                        if dataset_id not in DATASETS_INFO.keys():
                            continue
                        # modif
                        self.count += 1
                        if self.count < 0:
                            pbar.update(1)
                            continue
                        # modif
                        key = f"{search_space_id}_{dataset_id}_{seed_id}"
                        self.init_x = initialization_data[key]["X"]
                        self.init_y = initialization_data[key]["y"]

                        record = self.run_method(seed_id, search_space_id, dataset_id)
                        self.results.append(record)
                        if self.llm_method:
                            with open(os.path.join(self.results_path, f"{self.name}.json"), "w",
                                      encoding="utf-8") as f:
                                json.dump(self.results, f, indent=4, ensure_ascii=False)
                                logger.info(f"文件保存成功！")
                            # 打开一个文件，如果文件不存在则创建
                            with open(os.path.join(self.results_path, f"{self.name}.txt"), 'w') as file:
                                # 写入一个字符串到文件
                                file.write(f'input_tokens: {self.input_tokens}\n'
                                           f'output_tokens: {self.output_tokens}\n '
                                           f'total_tokens: {self.input_tokens + self.output_tokens}\n'
                                           )
                        if len(errors) > 0:
                            errors_string = "\n".join([f"seed: {error[0]}, search space: {error[1]}, dataset: {error[2]}" for error in errors])
                            with open(os.path.join(self.results_path, f"{self.name}_error.txt"), 'w') as file:
                                file.write(errors_string)
                        pbar.update(1)
            end = time.time()
            time_costs = end - begin

        logger.info(f"Time cost: {time_costs}")
        # 打开一个文件，如果文件不存在则创建
        with open(os.path.join(self.results_path, f"{self.name}_time_costs.txt"), 'w') as file:
            # 写入一个字符串到文件
            file.write(f"Time cost: {time_costs}")
        if not self.llm_method:
            with open(os.path.join(self.results_path, f"{self.name}.json"), "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=4, ensure_ascii=False)

    def run_hpo_bench_experiment(self, seeds=5, n_trials=15, results_path=r"experiments/results/hpo_bench"):
        self.seeds = seeds
        self.n_trials = n_trials
        self.results = []
        self.results_path = os.path.join(get_project_root(), results_path)
        models = ["lr", "svm", "rf", "xgb", "nn"]
        datasets = ["australian", "blood_transfusion", "car", "credit_g", "kc1", "phoneme", "segment", "vehicle"]

        with tqdm(total=len(models) * len(datasets) * self.seeds) as pbar:
            begin = time.time()
            for seed_id in range(self.seeds):
                for model in models:
                    for dataset in datasets:
                        # modif
                        self.count += 1
                        if self.count < 0:
                            pbar.update(1)
                            continue
                        # modif
                        record = self.run_hpo_bench_method(seed_id, model, dataset)
                        self.results.append(record)
                        if self.llm_method:
                            with open(os.path.join(self.results_path, f"{self.name}.json"), "w", encoding="utf-8") as f:
                                json.dump(self.results, f, indent=4, ensure_ascii=False)
                            # 打开一个文件，如果文件不存在则创建
                            with open(os.path.join(self.results_path, f"{self.name}.txt"), 'w') as file:
                                # 写入一个字符串到文件
                                file.write(f'input_tokens: {self.input_tokens}\n'
                                           f'output_tokens: {self.output_tokens}\n '
                                           f'total_tokens: {self.input_tokens + self.output_tokens}\n'
                                           )
                        pbar.update(1)
            end = time.time()
            time_costs = end - begin

        logger.info(f"Time cost: {time_costs}")
        # 打开一个文件，如果文件不存在则创建
        with open(os.path.join(self.results_path, f"{self.name}_time_costs.txt"), 'w') as file:
            # 写入一个字符串到文件
            file.write(f"Time cost: {time_costs}")

        if not self.llm_method:
            with open(os.path.join(self.results_path, f"{self.name}.json"), "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=4, ensure_ascii=False)

    def run_nl2workflow_experiment(self, seeds=5, n_trials=15, results_path=r"experiments/results/nl2workflow"):
        self.seeds = seeds
        self.n_trials = n_trials
        self.results = []
        self.results_path = os.path.join(get_project_root(), results_path)
        task_map = {
                        "XGBoost": ["horse_health_outcomes"],
                        "Random Forest": ["reservation_cancellation", "smoker_status_bio_signals"],
                        "SVM": ["horse_health_outcomes"],
                        "LightGBM": ["obesity_cvd_risk", "reservation_cancellation"],
                        "AdaBoost": ["bank_customer_churn", "kidney_stone"],
                        "KNN": ["horse_health_outcomes", "kidney_stone"],
                        "Logistic Regression": ["kidney_stone"]
                    }
        numbers = 0
        for t in task_map:
            numbers += len(task_map[t])
        with tqdm(total=numbers * self.seeds) as pbar:
            begin = time.time()
            for seed_id in range(self.seeds):
                for model in task_map:
                    for dataset in task_map[model]:
                        # modif
                        self.count += 1
                        if self.count < 0:
                            pbar.update(1)
                            continue
                        # modif
                        record = self.run_nl2workflow_method(seed_id, model, dataset)
                        self.results.append(record)
                        if self.llm_method:
                            with open(os.path.join(self.results_path, f"{self.name}.json"), "w", encoding="utf-8") as f:
                                json.dump(self.results, f, indent=4, ensure_ascii=False)
                            # 打开一个文件，如果文件不存在则创建
                            with open(os.path.join(self.results_path, f"{self.name}.txt"), 'w') as file:
                                # 写入一个字符串到文件
                                file.write(f'input_tokens: {self.input_tokens}\n'
                                           f'output_tokens: {self.output_tokens}\n '
                                           f'total_tokens: {self.input_tokens + self.output_tokens}\n'
                                           )
                        pbar.update(1)
            end = time.time()
            time_costs = end - begin

        logger.info(f"Time cost: {time_costs}")
        # 打开一个文件，如果文件不存在则创建
        with open(os.path.join(self.results_path, f"{self.name}_time_costs.txt"), 'w') as file:
            # 写入一个字符串到文件
            file.write(f"Time cost: {time_costs}")

        if not self.llm_method:
            with open(os.path.join(self.results_path, f"{self.name}.json"), "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=4, ensure_ascii=False)

    def run_method(self, seed_id, search_space_id, dataset_id):
        pass

    def run_hpo_bench_method(self, seed_id, model, dataset):
        pass

    def run_nl2workflow_method(self, seed_id, model, dataset):
        pass

if __name__ == '__main__':
    # 计算总迭代次数
    total_iterations = 0
    for seed_id in range(5):
        for search_space_id in SEARCH_SPACE_and_DATASETS:
            for dataset_id in SEARCH_SPACE_and_DATASETS[search_space_id]:
                if dataset_id not in DATASETS_INFO.keys():
                    continue
                total_iterations += 1
    print(f"Total iterations: {total_iterations}")