from config import MODEL_NAME
import random
random.seed(42)
import re
import os
import openml
from llm.get_llm import get_llm
from utils.utils import get_project_root
from utils.logs import logger
from baselines.baseline import Baseline
import numpy as np
import xgboost as xgb
from data.benchmarks.hpob.search_spaces_info import SEARCH_SPACE_INFO
from data.benchmarks.hpob.datasets_info import DATASETS_INFO
import json
import asyncio

MODEL_NAME_MAP = {
    "Meta-Llama-3-8B-Instruct": "llama3-8b",
    "Meta-Llama-3-70B-Instruct": "llama3-70b",
    "Qwen2.5-14B-Instruct": "qwen2_5-14b",
    "Qwen2.5-72B-Instruct": "qwen2_5-72b",
    "gpt-4o": "gpt-4o",
    "deepseek-reasoner": "deepseek-r1",
    "deepseek-chat": "deepseek-v3",
    "glm-4-9b-chat": "glm4-9b",
}
PROMPT_TEMPLATE = """
# Role
You are an expert in machine learning model hyperparameter configuration.

# Task
Your task is to recommend the best configurations to train a model for a classification dataset.

# Model
{ml_model} Model has {parameters_number} configurable hyper-parameters, i.e., {parameters}.
The allowable ranges for the hyperparameters are:
{configurations_ranges}

# Dataset
{task}

# Examples
The following are examples of performance of a {ml_model} measured in accuracy and the corresponding model hyperparameter configurations.

{cases}

# Output Format
Format strictly follows this template: 
```
{output_format}```

# Remember
1. Give {configurations_numbers} hyperparameter configurations that would enable the model to outperform the above examples.
2. Please give the {configurations_numbers} configurations in a strictly direct output format, no nonsense!
"""

class LLM_Few_Shot(Baseline):
    """
    不基于历史数据
    """
    def __init__(self):
        super().__init__()
        self.llm = get_llm()
        self.llm_method = True
        temperatures = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        self.t = 9
        self.temperature = temperatures[self.t]
        self.top_K = 3
        self.name = f"LLM_Few_Shot_temperature{self.t}({MODEL_NAME_MAP[MODEL_NAME]})"

    def run_method(self, seed_id, search_space_id, dataset_id):
        # 加载评估代理模型
        surrogate_name = 'surrogate-' + search_space_id + '-' + dataset_id
        bst_surrogate = xgb.Booster()
        bst_surrogate.load_model(self.surrogates_dir + surrogate_name + '.json')
        search_space = SEARCH_SPACE_INFO[search_space_id]
        dataset = DATASETS_INFO[dataset_id]
        dim = len(search_space["parameters_name"])
        y_min = self.surrogates_stats[surrogate_name]["y_min"]
        y_max = self.surrogates_stats[surrogate_name]["y_max"]

        x_observed = []
        y_observed = []
        max_accuracy_history = []
        self.benchmark = "hpob"
        configs = self.get_configs(search_space_id, dataset)
        for i in range(self.n_trials):
            if i > len(configs)-1:
                continue
            new_x = configs[i]

            # 获取下一个要探索的超参数配置
            try:
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

            except Exception as e:
                logger.info(f"config: {new_x}, Error:{e}")
                continue

            if len(configs) <= i + 1:
                max_accuracy_history += [max(y_observed)] * (self.n_trials - i - 1)
                break

        logger.info(f"y_observed: {len(y_observed)}")
        assert len(y_observed) > 0
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
        parameters_name = [key for key in search_space.keys()]
        x_observed = []
        y_observed = []
        max_accuracy_history = []
        self.init_x = []
        self.init_y = []
        init_configs = benchmark.generate_initialization()
        for config in init_configs:
            x = [config[key] for key in config.keys()]
            y = benchmark.evaluate_point(config)
            self.init_x.append(x)
            self.init_y.append(y)
        self.benchmark = "hpo_bench"
        configs = self.get_configs(model, str(DATASET_MAP[dataset_name][1]))
        for i in range(self.n_trials):
            if i > len(configs) - 1:
                continue
            new_x = configs[i]
            new_x_eval = {parameters_name[j]: new_x[j] for j in range(len(parameters_name))}

            # 评估超参数配置
            try:
                new_y = benchmark.evaluate_point(new_x_eval)
                # 记录观察到的超参数和分数
                y_observed.append(new_y)
                x_observed.append(new_x)
                # 记录当前最优的分数
                max_accuracy_history.append(max(y_observed))

            except Exception as e:
                print(new_x_eval)
                print(f"Error:{e}")
                continue

            if len(configs) <= i + 1:
                max_accuracy_history += [max(y_observed)] * (self.n_trials - i - 1)
                break

        logger.info(f"y_observed: {len(y_observed)}")
        assert len(y_observed) > 0
        record = {
            "search_space": model,
            "dataset": dataset,
            "seed": seed_id,
            "x_observed": x_observed,
            "y_observed": y_observed,
            "max_accuracy_history": max_accuracy_history
        }

        return record

    def run_nl2workflow_method(self, seed_id, model, dataset_name):
        from data.benchmarks.nl2workflow.eval import evaluate
        with open(os.path.join(get_project_root(), "data/benchmarks/nl2workflow/search_space.json"), 'r', encoding='utf-8') as f:
            search_space = json.load(f)
        for s in search_space:
            if s["algorithm"] == model:
                self.search_space = s["search_space"]
                break
        x_observed = []
        y_observed = []
        max_accuracy_history = []

        self.init_x = []
        self.init_y = []

        # configs, ys = get_init_configs(model=model, dataset=dataset_name, seed=seed_id)
        # for i in range(len(configs)):
        #     x = configs[i]
        #     y = ys[i]
        #     self.init_x.append(x)
        #     self.init_y.append(y)

        for i in range(5):
            new_x = self.random_search_nl2workflow(self.search_space)
            self.init_y.append(evaluate(new_x, model, dataset_name))
            self.init_x.append(new_x)

        self.benchmark = "nl2workflow"
        configs = self.get_configs(model, dataset_name)
        for i in range(self.n_trials):
            if i > len(configs)-1:
                continue
            new_x = configs[i]
            # 评估超参数配置
            try:
                new_y = evaluate(new_x, model, dataset_name)
                # 记录观察到的超参数和分数
                y_observed.append(new_y)
                x_observed.append(new_x)
                # 记录当前最优的分数
                max_accuracy_history.append(max(y_observed))
            except Exception as e:
                print(new_x)
                print(f"Error:{e}")
                continue
            if len(configs) <= i+1:
                max_accuracy_history += [max(y_observed)] * (self.n_trials - i - 1)
                break
        logger.info(f"y_observed: {len(y_observed)}")
        assert len(y_observed) > 0
        record = {
            "search_space": model,
            "dataset": dataset_name,
            "seed": seed_id,
            "x_observed": x_observed,
            "y_observed": y_observed,
            "max_accuracy_history": max_accuracy_history
        }

        return record

    async def predict(self, messages, temperature, top_p):
        coroutines = []
        for m in messages:
            coroutines.append(self.llm.async_predict(messages=m, temperature=temperature, top_p=top_p))

        tasks = [asyncio.create_task(c) for c in coroutines]
        results = []
        llm_response = await asyncio.gather(*tasks)
        for idx, response in enumerate(llm_response):
            if response[0] is not None:
                results.append(response[0])
            if response[1] is not None:
                self.input_tokens += response[1]
                self.output_tokens += response[2]

        return results

    def get_configs(self, search_space_id, dataset):
        eval_configs = []
        configs = []

        # 一次LLM返回的配置数量
        k = self.top_K

        msg = self.get_prompt(search_space_id, dataset, k)

        messages = []
        for i in range(int(self.n_trials / k)):
            messages.append(msg)
        responses = asyncio.run(self.predict(messages=messages, temperature=self.temperature, top_p=1))
        logger.info(f"llm end:{len(responses)}")
        for idx, response in enumerate(responses):
            logger.info(f"llm response: {idx}")
            logger.info(response)
            try_times = 0
            while try_times < 5:
                try:
                    suggest_config, suggest_eval_config = self.parse_configs(response, k)
                    if len(suggest_eval_config) > 0:
                        break
                    else:
                        raise Exception("配置数量为0")
                except Exception as e:
                    logger.error(f"解析配置失败，错误信息为：{e}")
                    try_times += 1
                    response, message, input_tokens, output_tokens, price = self.llm.predict(messages=msg,
                                                                                             temperature=self.temperature, top_p=1)
                    self.input_tokens += input_tokens
                    self.output_tokens += output_tokens

            if try_times >= 5:
                continue
            for i in range(len(suggest_eval_config)):
                eval_configs.append([suggest_eval_config[i][key] for key in suggest_eval_config[i]])
            if len(eval_configs) >= self.n_trials:
                break
        logger.info(f"获得的配置数为：{len(eval_configs)}")
        return eval_configs

    def get_prompt(self, search_space_id, dataset, k=5):
        if self.benchmark == "hpob":
            search_space = SEARCH_SPACE_INFO[search_space_id]
            TASK_DESCRIPTION_TEMPLATE = """The dataset name is "{dataset_name}".
The dataset contains {classes_number} classes, {num_samples} instances, {features_number} features, {numeric_features_number} numeric features, {categorical_features_number} categorical features.
The target variable has the following characteristics: 
{target_characteristics}."""
            ml_model = search_space["model_name"]
            parameters_name = search_space["parameters_name"]
            target_characteristics = ".".join(
                [f'The "{i}" class size is {dataset["target_characteristics"][i][1]}' for i in
                 dataset["target_characteristics"]])
            task_description = TASK_DESCRIPTION_TEMPLATE.format(
                dataset_name=dataset["name"],
                classes_number=len(dataset["target_characteristics"]),
                num_samples=dataset["num_samples"],
                features_number=dataset["numeric_features"] + dataset["categorical_features"],
                numeric_features_number=dataset["numeric_features"],
                categorical_features_number=dataset["categorical_features"],
                target_characteristics=target_characteristics
            )
            configurations_ranges = ""
            for parameter in parameters_name:
                if search_space[parameter]["type"] == "categorical":
                    configurations_ranges += f"- {parameter}: It is a categorical variable, must take value in {search_space[parameter]['categories']}.\n"
                else:
                    configurations_ranges += f"- {parameter}: It is a continuous variable, must take value in {search_space[parameter]['low']} to {search_space[parameter]['high']}.\n"

            cases_string = ""

            for i in range(len(self.init_x)):
                case_str = ""
                for j in range(len(parameters_name)):
                    if j == len(parameters_name) - 1:
                        case_str += f"{parameters_name[j]} is {self.init_x[i][j]}."
                    else:
                        case_str += f"{parameters_name[j]} is {self.init_x[i][j]}. "

                cases_string += f"Configuration {i + 1}: {case_str}\n"
                cases_string += f"Performance {i + 1}: {self.init_y[i]}\n"
            logger.info(cases_string)

            output_format = ""
            for i in range(k):
                output_format += f"Configuration {i + 1}" + ": {{parameter_1_name}} is {{parameter_1_value}}. {{parameter_2_name}} is {{parameter_2_value}}...{{parameter_n_name}} is {{parameter_n_value}}.\n"

            prompt = PROMPT_TEMPLATE.format(
                ml_model=ml_model,
                parameters_number=len(parameters_name),
                parameters=','.join(parameters_name),
                task=task_description,
                output_format=output_format,
                configurations_numbers=k,
                cases=cases_string,
                configurations_ranges=configurations_ranges
            )
            messages = [
                {"role": "user", "content": prompt}
            ]
            return messages
        elif self.benchmark == "nl2workflow":
            TASK_DESCRIPTION_TEMPLATE = """The dataset name is "{dataset_name}".
The dataset contains {classes_number} classes, {num_samples} instances, {features_number} features, {numeric_features_number} numeric features, {categorical_features_number} categorical features.
The target variable has the following characteristics: 
{target_characteristics}."""
            dataset_info_path = os.path.join(get_project_root(),
                                             r"data/benchmarks/nl2workflow/datasets/datasets_info.json")
            with open(dataset_info_path, 'r', encoding='utf-8') as f:
                dataset_info = json.load(f)[dataset]
            task_description = TASK_DESCRIPTION_TEMPLATE.format(
                dataset_name=dataset,
                classes_number=dataset_info["classes_number"],
                num_samples=dataset_info["num_samples"],
                features_number=dataset_info["features_number"],
                numeric_features_number=dataset_info["numeric_features_number"],
                categorical_features_number=dataset_info["categorical_features_number"],
                target_characteristics=dataset_info["target_characteristics"]
            )
            parameters_name = [key for key in self.search_space]
            configurations_ranges = ""
            for parameter in parameters_name:
                if self.search_space[parameter]["_type"] == "choice":
                    configurations_ranges += f"- {parameter}: It is a categorical variable, must take value in {self.search_space[parameter]['_value']}.\n"
                else:
                    configurations_ranges += f"- {parameter}: It is a continuous variable, must take value in {self.search_space[parameter]['_value'][0]} to {self.search_space[parameter]['_value'][1]}.\n"

            cases_string = ""
            for i in range(len(self.init_x)):
                case_str = ""
                for j in range(len(parameters_name)):
                    if j == len(parameters_name) - 1:
                        case_str += f"{parameters_name[j]} is {self.init_x[i][j]}."
                    else:
                        case_str += f"{parameters_name[j]} is {self.init_x[i][j]}. "

                cases_string += f"Configuration {i + 1}: {case_str}\n"
                cases_string += f"Performance {i + 1}: {self.init_y[i]}\n"

            logger.info(cases_string)
            output_format = ""
            for i in range(k):
                output_format += f"Configuration {i + 1}" + ": {{parameter_1_name}} is {{parameter_1_value}}. {{parameter_2_name}} is {{parameter_2_value}}...{{parameter_n_name}} is {{parameter_n_value}}.\n"

            prompt = PROMPT_TEMPLATE.format(
                ml_model=search_space_id,
                parameters_number=len(parameters_name),
                parameters=','.join(parameters_name),
                task=task_description,
                output_format=output_format,
                configurations_numbers=k,
                cases=cases_string,
                configurations_ranges=configurations_ranges
            )
            messages = [
                {"role": "user", "content": prompt}
            ]
            print(prompt)
            return messages

        else:
            TASK_DESCRIPTION_TEMPLATE = """The dataset name is "{dataset_name}".
        The dataset contains {classes_number} classes, {num_samples} instances, {features_number} features, {numeric_features_number} numeric features, {categorical_features_number} categorical features."""
            DATASET_MAP = {
                "3": "kr-vs-kp",
                "14965": "bank-mark..",
                "9977": "nomao",
                "9952": "phoneme",
                "3917": "kc1",
                "146818": "australian",
                "10101": "blood_transfusion",
                "146821": "car",
                "146822": "segment",
                "31": "credit_g",
                "53": "vehicle"
            }
            MODEL_MAP = {
                'rf': 'Random Forest',
                'nn': 'Multilayer Perceptron',
                'xgb': 'XGBoost',
                "svm": 'SVM',
                "lr": "Logistic Regression"
            }
            order_list_info = {
                "nn": ["alpha", "batch_size", "depth", "learning_rate_init", "width"],
                "rf": ["max_depth", "max_features", "min_samples_leaf", "min_samples_split"],
                "xgb": ["colsample_bytree", "eta", "max_depth", "reg_lambda"],
                "svm": ["C", "gamma"],
                "lr": ["alpha", "eta0"]
            }
            dataset_id = str(dataset)
            task = openml.tasks.get_task(int(dataset_id))
            dataset_ = task.get_dataset()
            X, y, categorical_mask, _ = dataset_.get_data(target=dataset_.default_target_attribute)

            task_description = TASK_DESCRIPTION_TEMPLATE.format(
                dataset_name=DATASET_MAP[dataset],
                classes_number=len(np.unique(y)),
                num_samples=X.shape[0],
                features_number=X.shape[1],
                numeric_features_number=X.shape[1] - len(categorical_mask),
                categorical_features_number=len(categorical_mask)
            )

            output_format = ""
            for i in range(k):
                output_format += f"Configuration {i + 1}" + ": {{parameter_1_name}} is {{parameter_1_value}}. {{parameter_2_name}} is {{parameter_2_value}}...{{parameter_n_name}} is {{parameter_n_value}}.\n"

            with open(os.path.join(get_project_root(), r"data/benchmarks/hpo_bench/hpo_bench.json"), 'r') as f:
                search_space = json.load(f)[search_space_id]

            configurations_ranges = ""
            for parameter in order_list_info[search_space_id]:
                configurations_ranges += f"- {parameter}: It is a categorical variable, must take value in {search_space[parameter]}.\n"
            cases_string = ""
            parameters_name = order_list_info[search_space_id]
            for i in range(len(self.init_x)):
                case_str = ""
                for j in range(len(parameters_name)):
                    if j == len(parameters_name) - 1:
                        case_str += f"{parameters_name[j]} is {self.init_x[i][j]}."
                    else:
                        case_str += f"{parameters_name[j]} is {self.init_x[i][j]}. "

                cases_string += f"Configuration {i + 1}: {case_str}\n"
                cases_string += f"Performance {i + 1}: {self.init_y[i]}\n"
            logger.info(f"\n{cases_string}")
            prompt = PROMPT_TEMPLATE.format(
                ml_model=MODEL_MAP[search_space_id],
                parameters_number=len(parameters_name),
                parameters=','.join(order_list_info[search_space_id]),
                task=task_description,
                output_format=output_format,
                configurations_numbers=k,
                configurations_ranges=configurations_ranges,
                cases=cases_string
            )
            messages = [
                {"role": "user", "content": prompt}
            ]
            return messages

    def parse_configs(self, response, topk=5):
        pattern_0 = re.compile(r"Configuration \d+: (.*)\.\n")
        suggest_configs = []
        configs = []
        groups = re.findall(pattern_0, response + "\n")
        for t in groups[:topk]:
            configs.append(t)
            kvs = t.split(". ")
            if len(kvs)==1:
                kvs = t.split(", ")
            config = {}
            for kv in kvs:
                _k, v = kv.strip().split(" is ")
                _k = _k.replace('`', '')
                try:
                    config[_k] = float(v)
                except:
                    config[_k] = v
            suggest_configs.append(config)

        if len(suggest_configs) == 0:
            pattern_0 = re.compile(r"Configuration: (.*)\.\n")
            suggest_configs = []
            configs = []
            groups = re.findall(pattern_0, response + "\n")
            for t in groups[:topk]:
                configs.append(t)
                kvs = t.split(". ")
                if len(kvs) == 1:
                    kvs = t.split(", ")
                config = {}
                for kv in kvs:
                    _k, v = kv.strip().split(" is ")
                    _k = _k.replace('`', '')
                    try:
                        config[_k] = float(v)
                    except:
                        config[_k] = v
                suggest_configs.append(config)
        if len(suggest_configs) == 0:
            pattern_0 = re.compile(r"Configuration \d+:\n(.*)\.\n")
            suggest_configs = []
            configs = []
            groups = re.findall(pattern_0, response + "\n")
            for t in groups[:topk]:
                configs.append(t)
                kvs = t.split(". ")
                if len(kvs) == 1:
                    kvs = t.split(", ")
                config = {}
                for kv in kvs:
                    _k, v = kv.strip().split(" is ")
                    _k = _k.replace('`', '')
                    try:
                        config[_k] = float(v)
                    except:
                        config[_k] = v
                suggest_configs.append(config)

        return configs, suggest_configs

    def random_search_nl2workflow(self, search_space):
        parameters_name = [key for key in search_space]
        x_new = []
        for parameter_name in parameters_name:
            if search_space[parameter_name]['_type'] == "uniform":
                low = search_space[parameter_name]['_value'][0]
                high = search_space[parameter_name]['_value'][1]
                random_parameter = round(random.uniform(low, high), 4)
            elif search_space[parameter_name]['_type'] == "randint":
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
    llm_fs = LLM_Few_Shot()
    llm_fs.run_nl2workflow_method(0,"XGBoost","smoker_status_bio_signals")