import json
import re
import asyncio
import random
random.seed(42)
import openml
from scipy.stats import norm
import numpy as np
import pandas as pd
import xgboost as xgb
from data.benchmarks.hpob.search_spaces_info import SEARCH_SPACE_INFO
from data.benchmarks.hpob.datasets_info import DATASETS_INFO
from utils.logs import logger
from llm.get_llm import get_llm
from prompts.llambo_prompts import CANDIDATE_SAMPLER_PROMPT, SURROGATE_MODEL_PROMPT
from baselines.baseline import Baseline
from data.benchmarks.hpo_bench.tabular_benchmarks import DATASET_MAP, MODEL_MAP, HPOExpRunner
import os
from utils.utils import get_project_root
from config import MODEL_NAME
MODEL_NAME_MAP = {
    "Meta-Llama-3-8B-Instruct": "llama3-8b",
    "Meta-Llama-3-70B-Instruct": "llama3-70b",
    "Qwen2.5-14B-Instruct": "qwen2_5-14b",
    "Qwen2.5-72B-Instruct": "qwen2_5-72b"
}
NUMBERS = 10

class LLAMBO(Baseline):

    def __init__(self, alpha=0.1):
        super().__init__()
        self.name = f"LLAMBO({MODEL_NAME_MAP[MODEL_NAME]})"
        self.llm_method = True
        self.alpha = alpha
        self.llm = get_llm()

    def run_method(self, seed_id, search_space_id, dataset_id):

        # 加载评估代理模型
        surrogate_name = 'surrogate-' + search_space_id + '-' + dataset_id
        bst_surrogate = xgb.Booster()
        bst_surrogate.load_model(self.surrogates_dir + surrogate_name + '.json')
        self.bst_surrogate = bst_surrogate

        search_space = SEARCH_SPACE_INFO[search_space_id]
        self.search_space = search_space
        dataset = DATASETS_INFO[dataset_id]
        self.dataset = dataset
        self.dim = len(search_space["parameters_name"])
        y_min = self.surrogates_stats[surrogate_name]["y_min"]
        y_max = self.surrogates_stats[surrogate_name]["y_max"]

        x_observed = []
        y_observed = []
        max_accuracy_history = []
        print(seed_id, search_space_id, dataset_id)
        for i in range(self.n_trials):
            logger.info(f"i: {i}")
            # 获取K个超参数配置
            candidate_points = self.get_candidate_points(x_observed+self.init_x, y_observed+self.init_y, alpha=self.alpha)
            logger.info(f"candidate_points: \n{candidate_points}")
            if len(candidate_points) != 0:
                # 用LLM充当代理模型，去评估K个超参数配置的性能
                candidate_points_score = self.evaluate_candidate_points(x_observed+self.init_x, y_observed+self.init_y, candidate_points)
                logger.info(f"candidate_points_score: \n{candidate_points_score}")
                # 选择本次要评估的一个超参数配置
                new_x = self.max_acq_function(y_observed+self.init_y, candidate_points, candidate_points_score)
                logger.info(f"new_x: \n{new_x}")
                new_x_arry = np.array(new_x)
                # 评估超参数配置
                x_q = xgb.DMatrix(new_x_arry.reshape(-1, self.dim))
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

    def run_hpo_bench_method(self, seed_id, model, dataset_name):
        dataset = DATASET_MAP[dataset_name][0]
        benchmark = HPOExpRunner(model, dataset, seed_id)
        with open(os.path.join(get_project_root(), r"data/benchmarks/hpo_bench/hpo_bench.json"), 'r') as f:
            search_space = json.load(f)[model]
        parameters_name = [key for key in search_space.keys()]

        x_observed = []
        y_observed = []
        max_accuracy_history = []
        init_configs = benchmark.generate_initialization()
        init_x = []
        init_y = []
        for config in init_configs:
            x = [config[key] for key in config.keys()]
            init_x.append(x)

            y = benchmark.evaluate_point(config)
            init_y.append(y)

        for i in range(self.n_trials):
            logger.info(f"i: {i}")
            # 获取K个超参数配置
            candidate_points = self.get_candidate_points_bench(x_observed+init_x, y_observed+init_y, model, dataset_name, alpha=self.alpha)
            logger.info(f"candidate_points: \n{candidate_points}")
            if len(candidate_points) != 0:
                # 用LLM充当代理模型，去评估K个超参数配置的性能
                candidate_points_score = self.evaluate_candidate_points_bench(x_observed+init_x, y_observed+init_y, candidate_points, model, dataset_name)
                logger.info(f"candidate_points_score: \n{candidate_points_score}")
                # 选择本次要评估的一个超参数配置
                new_x = self.max_acq_function(y_observed+init_y, candidate_points, candidate_points_score)
                logger.info(f"new_x: \n{new_x}")
                evaluate_point = {}
                for idx, name in enumerate(parameters_name):
                    evaluate_point[name] = new_x[idx]
                new_y = benchmark.evaluate_point(evaluate_point)
                # new_y = float(new_y_arry)
                y_observed.append(new_y)
                x_observed.append(new_x)
            best_f = max(y_observed)
            # 记录当前最优的分数
            max_accuracy_history.append(best_f)

        record = {
            "search_space": model,
            "dataset": dataset_name,
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
        init_x = []
        init_y = []
        for i in range(5):
            new_x = self.random_search_nl2workflow(self.search_space)
            init_y.append(evaluate(new_x, model, dataset_name))
            init_x.append(new_x)

        for i in range(self.n_trials):
            logger.info(f"i: {i}")
            # 获取K个超参数配置
            candidate_points = self.get_candidate_points_nl2workflow(x_observed+init_x, y_observed+init_y, model, dataset_name, alpha=self.alpha)
            logger.info(f"candidate_points: \n{candidate_points}")
            if len(candidate_points) != 0:
                # 用LLM充当代理模型，去评估K个超参数配置的性能
                candidate_points_score = self.evaluate_candidate_points_nl2workflow(x_observed+init_x, y_observed+init_y, candidate_points, model, dataset_name)
                logger.info(f"candidate_points_score: \n{candidate_points_score}")
                # 选择本次要评估的一个超参数配置
                new_x = self.max_acq_function(y_observed+init_y, candidate_points, candidate_points_score)
                logger.info(f"new_x: \n{new_x}")
                new_y = evaluate(new_x, model, dataset_name)
                # new_y = float(new_y_arry)
                y_observed.append(new_y)
                x_observed.append(new_x)
            best_f = max(y_observed)
            # 记录当前最优的分数
            max_accuracy_history.append(best_f)

        record = {
            "search_space": model,
            "dataset": dataset_name,
            "seed": seed_id,
            "x_observed": x_observed,
            "y_observed": y_observed,
            "max_accuracy_history": max_accuracy_history
        }

        return record

    async def predict(self, messages, temperature, max_tokens, top_p):
        coroutines = []
        for message in messages:
            coroutines.append(self.llm.async_predict(messages=message, temperature=temperature, top_p=top_p))

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

    def get_candidate_points(self, x_observed, y_observed, alpha=0.1):
        # return [[0.6666632854092085, 0.4999907956696741, 0.5222445163303155], [0.499990796, 0.333328261, 0.6666632854092085], [0.6666632854092085, 0.4999907956696741, 0.8333297642456549], [0.6666632854092085, 0.4999907956696741, 0.49999539800000004], [0.499990796, 0.333328261, 0.666663285], [0.499990796, 0.666663285, 0.833331549], [0.6666632854092085, 0.3333282611874454, 0.4999953985575123], [0.499990796, 0.33332319, 0.666663285], [0.6666632854092085, 0.4999907956696741, 0.5576119564475298], [0.499990796, 0.33332319, 0.6666632854092085]]

        ranges = np.abs(np.max(y_observed) - np.min(y_observed))
        if ranges == 0:
            ranges = 0.1*np.abs(np.max(y_observed))
        alpha_range = [0.1, 1e-2, 1e-3, -1e-3, -1e-2, 1e-1]
        observed_best = np.max(y_observed)
        observed_worst = np.min(y_observed)
        desired_fval = observed_best + alpha * ranges

        if desired_fval >= .9999:  # accuracy can't be greater than 1
            for alpha_ in alpha_range:
                if alpha_ < alpha:
                    alpha = alpha_  # new alpha
                    desired_fval = observed_best + alpha * ranges
                    break
        if desired_fval >= .9999:
            desired_fval = 1.0
        # 构造提示词
        search_space = self.search_space
        dataset = self.dataset
        dataset["metric"] = "accuracy"
        hyperparameters_info = [search_space[p] for p in search_space["parameters_name"]]


        observed_configs = pd.DataFrame(x_observed, columns=search_space["parameters_name"])
        observed_scores = pd.DataFrame(y_observed, columns=["score"])
        cases = pd.concat([observed_configs, observed_scores], axis=1)
        # cases = cases.sort_values(by="score", ascending=True)
        def get_cases_str(cases):
            # 构造示例字符串
            cases_str = ""
            for index, row in cases.iterrows():
                row_col = row.drop("score")
                c_str = ", ".join([f"{k}: {v}" for k, v in row_col.items()])
                cases_str += f"\nPerformance: {row['score']:.6f}\nHyperparameter configuration: ## {c_str} ##"
            prompt = CANDIDATE_SAMPLER_PROMPT.format(
                model=search_space["model_name"],
                metric=dataset["metric"],
                task=dataset["type"],
                classes_number=len(dataset["target_characteristics"]),
                samples_number=dataset["num_samples"],
                features_number=dataset["categorical_features"] + dataset["numeric_features"],
                categorical_features_number=dataset["categorical_features"],
                continuous_features_number=dataset["numeric_features"],
                hyperparameters_info=json.dumps(hyperparameters_info, ensure_ascii=False),
                target_score=desired_fval,
                cases=cases_str
            )
            msg = []
            # msg = [[{"role": "system", "content": "You are an AI assistant that helps people find information."},
            #         {"role": "user", "content": prompt}] * 5]
            for i in range(NUMBERS):
                msg.append([
                    {"role": "system", "content": "You are an AI assistant that helps people find information."},
                    {"role": "user", "content": prompt}
                ])
            return msg
        messages = []
        if len(self.search_space["parameters_name"]) > 5:
            cases_numbers = 5
        else:
            cases_numbers = 10
        for i in range(2):
            np.random.seed(i)
            shuffled_indices = np.random.permutation(cases.index)
            new_cases = cases.loc[shuffled_indices]
            if len(new_cases) > cases_numbers:
                new_cases = new_cases.head(cases_numbers)
            msg = get_cases_str(new_cases)
            messages.append(msg)

        # 最佳的cases_numbers个示例
        top5 = cases.sort_values(by="score", ascending=True)
        messages.append(get_cases_str(top5.head(cases_numbers)))

        retry = 0
        filtered_candidate_points = []
        while len(filtered_candidate_points) < NUMBERS:
            retry += 1
            if retry % 3 == 0:
                llm_responses = asyncio.run(
                    self.predict(messages=messages[2], temperature=0.8, max_tokens=500, top_p=0.95))
            elif retry % 3 == 1:
                llm_responses = asyncio.run(
                    self.predict(messages=messages[0], temperature=0.8, max_tokens=500, top_p=0.95))
            else:
                llm_responses = asyncio.run(
                    self.predict(messages=messages[1], temperature=0.8, max_tokens=500, top_p=0.95))

            for content in llm_responses:
                # 解析候选点
                candidate_point = self.parse_candidate_points(content, cases.columns)
                # 检查候选点是否合法
                if candidate_point not in x_observed+filtered_candidate_points and len(candidate_point) == (len(cases.columns)-1):
                    filtered_candidate_points.append(candidate_point)
            logger.info(f'Number of proposed candidate points: {retry*NUMBERS}')
            logger.info(f'Number of accepted candidate points: {len(filtered_candidate_points)}')
            if retry > 6:
                if len(filtered_candidate_points) <= 0:
                    filtered_candidate_points = []
                    logger.info('LLM failed to generate candidate points')
                break

        return filtered_candidate_points

    def get_candidate_points_nl2workflow(self, x_observed, y_observed, model, dataset_name, alpha=0.1):
        # return [[0.6666632854092085, 0.4999907956696741, 0.5222445163303155], [0.499990796, 0.333328261, 0.6666632854092085], [0.6666632854092085, 0.4999907956696741, 0.8333297642456549], [0.6666632854092085, 0.4999907956696741, 0.49999539800000004], [0.499990796, 0.333328261, 0.666663285], [0.499990796, 0.666663285, 0.833331549], [0.6666632854092085, 0.3333282611874454, 0.4999953985575123], [0.499990796, 0.33332319, 0.666663285], [0.6666632854092085, 0.4999907956696741, 0.5576119564475298], [0.499990796, 0.33332319, 0.6666632854092085]]

        ranges = np.abs(np.max(y_observed) - np.min(y_observed))
        if ranges == 0:
            ranges = 0.1*np.abs(np.max(y_observed))
        alpha_range = [0.1, 1e-2, 1e-3, -1e-3, -1e-2, 1e-1]
        observed_best = np.max(y_observed)
        observed_worst = np.min(y_observed)
        desired_fval = observed_best + alpha * ranges

        if desired_fval >= .9999:  # accuracy can't be greater than 1
            for alpha_ in alpha_range:
                if alpha_ < alpha:
                    alpha = alpha_  # new alpha
                    desired_fval = observed_best + alpha * ranges
                    break
        if desired_fval >= .9999:
            desired_fval = 1.0
        # 构造提示词
        dataset_info_path = os.path.join(get_project_root(),
                                         r"data/benchmarks/nl2workflow/datasets/datasets_info.json")
        with open(dataset_info_path, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)[dataset_name]
        parameters_name = [key for key in self.search_space]
        configurations_ranges = ""
        for parameter in parameters_name:
            if self.search_space[parameter]["_type"] == "choice":
                configurations_ranges += f"- {parameter}: It is a categorical variable, must take value in {self.search_space[parameter]['_value']}.\n"
            else:
                configurations_ranges += f"- {parameter}: It is a continuous variable, must take value in {self.search_space[parameter]['_value'][0]} to {self.search_space[parameter]['_value'][1]}.\n"

        observed_configs = pd.DataFrame(x_observed, columns=parameters_name)
        observed_scores = pd.DataFrame(y_observed, columns=["score"])
        cases = pd.concat([observed_configs, observed_scores], axis=1)
        # cases = cases.sort_values(by="score", ascending=True)
        def get_cases_str(cases):
            # 构造示例字符串
            cases_str = ""
            for index, row in cases.iterrows():
                row_col = row.drop("score")
                c_str = ", ".join([f"{k}: {v}" for k, v in row_col.items()])
                cases_str += f"\nPerformance: {row['score']:.6f}\nHyperparameter configuration: ## {c_str} ##"
            prompt = CANDIDATE_SAMPLER_PROMPT.format(
                model=model,
                metric="accuracy",
                task="classification",
                classes_number=dataset_info["classes_number"],
                samples_number=dataset_info["num_samples"],
                features_number=dataset_info["features_number"],
                categorical_features_number=dataset_info["categorical_features_number"],
                continuous_features_number=dataset_info["numeric_features_number"],
                hyperparameters_info=configurations_ranges,
                target_score=desired_fval,
                cases=cases_str
            )
            msg = []
            # msg = [[{"role": "system", "content": "You are an AI assistant that helps people find information."},
            #         {"role": "user", "content": prompt}] * 5]
            for i in range(NUMBERS):
                msg.append([
                    {"role": "system", "content": "You are an AI assistant that helps people find information."},
                    {"role": "user", "content": prompt}
                ])
            return msg
        messages = []
        if len(parameters_name) > 5:
            cases_numbers = 5
        else:
            cases_numbers = 10
        for i in range(2):
            np.random.seed(i)
            shuffled_indices = np.random.permutation(cases.index)
            new_cases = cases.loc[shuffled_indices]
            if len(new_cases) > cases_numbers:
                new_cases = new_cases.head(cases_numbers)
            msg = get_cases_str(new_cases)
            messages.append(msg)

        # 最佳的cases_numbers个示例
        top5 = cases.sort_values(by="score", ascending=True)
        messages.append(get_cases_str(top5.head(cases_numbers)))

        retry = 0
        filtered_candidate_points = []
        while len(filtered_candidate_points) < NUMBERS:
            retry += 1
            if retry % 3 == 0:
                llm_responses = asyncio.run(
                    self.predict(messages=messages[2], temperature=0.8, max_tokens=500, top_p=0.95))
            elif retry % 3 == 1:
                llm_responses = asyncio.run(
                    self.predict(messages=messages[0], temperature=0.8, max_tokens=500, top_p=0.95))
            else:
                llm_responses = asyncio.run(
                    self.predict(messages=messages[1], temperature=0.8, max_tokens=500, top_p=0.95))

            for content in llm_responses:
                # 解析候选点
                candidate_point = self.parse_candidate_points(content, cases.columns)
                # 检查候选点是否合法
                if candidate_point not in x_observed+filtered_candidate_points and len(candidate_point) == (len(cases.columns)-1):
                    filtered_candidate_points.append(candidate_point)
            logger.info(f'Number of proposed candidate points: {retry*NUMBERS}')
            logger.info(f'Number of accepted candidate points: {len(filtered_candidate_points)}')
            if retry > 6:
                if len(filtered_candidate_points) <= 0:
                    filtered_candidate_points = []
                    logger.info('LLM failed to generate candidate points')
                break

        return filtered_candidate_points

    def get_candidate_points_bench(self, x_observed, y_observed, model, dataset_name, alpha=0.1):

        # return [[0.6666632854092085, 0.4999907956696741, 0.5222445163303155], [0.499990796, 0.333328261, 0.6666632854092085], [0.6666632854092085, 0.4999907956696741, 0.8333297642456549], [0.6666632854092085, 0.4999907956696741, 0.49999539800000004], [0.499990796, 0.333328261, 0.666663285], [0.499990796, 0.666663285, 0.833331549], [0.6666632854092085, 0.3333282611874454, 0.4999953985575123], [0.499990796, 0.33332319, 0.666663285], [0.6666632854092085, 0.4999907956696741, 0.5576119564475298], [0.499990796, 0.33332319, 0.6666632854092085]]

        ranges = np.abs(np.max(y_observed) - np.min(y_observed))
        if ranges == 0:
            ranges = 0.1*np.abs(np.max(y_observed))
        alpha_range = [0.1, 1e-2, 1e-3, -1e-3, -1e-2, 1e-1]
        observed_best = np.max(y_observed)
        observed_worst = np.min(y_observed)
        desired_fval = observed_best + alpha * ranges

        if desired_fval >= .9999:  # accuracy can't be greater than 1
            for alpha_ in alpha_range:
                if alpha_ < alpha:
                    alpha = alpha_  # new alpha
                    desired_fval = observed_best + alpha * ranges
                    break
        if desired_fval >= .9999:
            desired_fval = 1.0
        # 构造提示词
        with open(os.path.join(get_project_root(), r"data/benchmarks/hpo_bench/hpo_bench.json"), 'r') as f:
            hyperparameter_constraints = json.load(f)[model]
        task = openml.tasks.get_task(DATASET_MAP[dataset_name][1])
        dataset_ = task.get_dataset()
        X, y, categorical_mask, _ = dataset_.get_data(target=dataset_.default_target_attribute)
        hyperparameter_constraints_str = ""
        for key, value in hyperparameter_constraints.items():
            hyperparameter_constraints_str += f"The range of values of '{key}' is: {value}\n"

        parameters_names = [parameters_name for parameters_name in hyperparameter_constraints]
        observed_configs = pd.DataFrame(x_observed, columns=parameters_names)
        observed_scores = pd.DataFrame(y_observed, columns=["score"])
        cases = pd.concat([observed_configs, observed_scores], axis=1)
        # cases = cases.sort_values(by="score", ascending=True)
        def get_cases_str(cases):
            # 构造示例字符串
            cases_str = ""
            for index, row in cases.iterrows():
                row_col = row.drop("score")
                c_str = ", ".join([f"{k}: {v}" for k, v in row_col.items()])
                cases_str += f"\nPerformance: {row['score']:.6f}\nHyperparameter configuration: ## {c_str} ##"

            prompt = CANDIDATE_SAMPLER_PROMPT.format(
                model=MODEL_MAP[model],
                metric="accuracy",
                task='classification',
                classes_number=len(np.unique(y)),
                samples_number=X.shape[0],
                features_number=X.shape[1],
                categorical_features_number=len(categorical_mask),
                continuous_features_number=X.shape[1] - len(categorical_mask),
                hyperparameters_info=hyperparameter_constraints_str,
                target_score=desired_fval,
                cases=cases_str
            )
            msg = []
            # msg = [[{"role": "system", "content": "You are an AI assistant that helps people find information."},
            #         {"role": "user", "content": prompt}] * 5]
            for i in range(NUMBERS):
                msg.append([
                    {"role": "system", "content": "You are an AI assistant that helps people find information."},
                    {"role": "user", "content": prompt}
                ])
            return msg
        messages = []
        if len(parameters_names) > 5:
            cases_numbers = 5
        else:
            cases_numbers = 10
        for i in range(2):
            np.random.seed(i)
            shuffled_indices = np.random.permutation(cases.index)
            new_cases = cases.loc[shuffled_indices]
            if len(new_cases) > cases_numbers:
                new_cases = new_cases.head(cases_numbers)
            msg = get_cases_str(new_cases)
            messages.append(msg)

        # 最佳的cases_numbers个示例
        top5 = cases.sort_values(by="score", ascending=True)
        messages.append(get_cases_str(top5.head(cases_numbers)))

        retry = 0
        filtered_candidate_points = []
        while len(filtered_candidate_points) < NUMBERS:
            retry += 1
            if retry % 3 == 0:
                llm_responses = asyncio.run(
                    self.predict(messages=messages[2], temperature=0.8, max_tokens=500, top_p=0.95))
            elif retry % 3 == 1:
                llm_responses = asyncio.run(
                    self.predict(messages=messages[0], temperature=0.8, max_tokens=500, top_p=0.95))
            else:
                llm_responses = asyncio.run(
                    self.predict(messages=messages[1], temperature=0.8, max_tokens=500, top_p=0.95))

            for content in llm_responses:
                # 解析候选点
                candidate_point = self.parse_candidate_points(content, cases.columns)
                # 检查候选点是否合法
                if candidate_point not in x_observed+filtered_candidate_points and len(candidate_point) == (len(cases.columns)-1):
                    filtered_candidate_points.append(candidate_point)
            logger.info(f'Number of proposed candidate points: {retry*NUMBERS}')
            logger.info(f'Number of accepted candidate points: {len(filtered_candidate_points)}')
            if retry > 6:
                if len(filtered_candidate_points) <= 0:
                    filtered_candidate_points = []
                    logger.info('LLM failed to generate candidate points')
                break

        return filtered_candidate_points

    def parse_candidate_points(self, content, columns):
        # 解析候选点
        candidate_point = []
        try:
            for column in columns:
                pattern = column + r":\s*([\d.]+|\w+)"
                match = re.search(pattern, content)
                if match:
                    try:candidate_point.append(float(match.group(1)))
                    except:candidate_point.append(match.group(1))
        except:
            logger.info(f'Failed to parse candidate points: {content}')
        return candidate_point

    def evaluate_candidate_points(self, x_observed, y_observed, candidate_points):
        # return [[0.9239, 0.9239, 0.9239, 0.9239, 0.9239], [0.9239, 0.9239, 0.9239, 0.9239, 0.9239], [0.9239, 0.9239, 0.9239, 0.9239, 0.9239], [0.9239, 0.9239, 0.9239, 0.9239, 0.9239], [0.9239, 0.9239, 0.9239, 0.9239, 0.9239], [0.9239, 0.9239, 0.9239, 0.9239, 0.9239], [0.9239, 0.9239, 0.9239, 0.9239, 0.9239], [0.9239, 0.9239, 0.9239, 0.9239, 0.9239], [0.9239, 0.9239, 0.9239, 0.9239, 0.9239], [0.9239, 0.9239, 0.9239, 0.9239, 0.9239]]

        # 构造提示词
        search_space = self.search_space
        dataset = self.dataset
        dataset["metric"] = "accuracy"
        observed_configs = pd.DataFrame(x_observed, columns=search_space["parameters_name"])
        observed_scores = pd.DataFrame(y_observed, columns=["score"])
        cases = pd.concat([observed_configs, observed_scores], axis=1)
        # cases = cases.sort_values(by="score", ascending=True)


        candidate_points_score = []
        for candidate_point in candidate_points:
            scores = []
            messages = []
            for i in range(5):
                np.random.seed(i)
                shuffled_indices = np.random.permutation(cases.index)
                new_cases = cases.loc[shuffled_indices]
                if len(new_cases) > 10:
                    new_cases = new_cases.head(10)
                # 构造示例字符串
                cases_str = ""
                for index, row in new_cases.iterrows():
                    row_col = row.drop("score")
                    c_str = ", ".join([f"{k}: {v}" for k, v in row_col.items()])
                    cases_str += f"\nHyperparameter configuration: {c_str}\nPerformance: ## {row['score']} ##"
                evaluated_configuration = ", ".join(
                    [f"{k}: {v}" for k, v in zip(search_space["parameters_name"], candidate_point)])
                prompt = SURROGATE_MODEL_PROMPT.format(
                    model=search_space["model_name"],
                    metric=dataset["metric"],
                    task=dataset["type"],
                    classes_number=len(dataset["target_characteristics"]),
                    samples_number=dataset["num_samples"],
                    features_number=dataset["categorical_features"] + dataset["numeric_features"],
                    categorical_features_number=dataset["categorical_features"],
                    continuous_features_number=dataset["numeric_features"],
                    evaluated_configuration=evaluated_configuration,
                    cases=cases_str
                )
                messages.append([
                    {"role": "system", "content": "You are an AI assistant that helps people find information."},
                    {"role": "user", "content": prompt}
                ])
            llm_responses = asyncio.run(self.predict(messages=messages, temperature=0.8, max_tokens=500, top_p=0.95))
            for content in llm_responses:
                # 解析
                try:
                    scores.append(float(content))
                except:
                    score = re.findall(r"## (-?[\d.]+) ##", content)
                    if len(score) == 1:
                        scores.append(float(score[0]))
                    else:
                        scores.append(np.nan)
            candidate_points_score.append(scores)

        return candidate_points_score

    def evaluate_candidate_points_nl2workflow(self, x_observed, y_observed, candidate_points, model, dataset_name):
        # return [[0.9239, 0.9239, 0.9239, 0.9239, 0.9239], [0.9239, 0.9239, 0.9239, 0.9239, 0.9239], [0.9239, 0.9239, 0.9239, 0.9239, 0.9239], [0.9239, 0.9239, 0.9239, 0.9239, 0.9239], [0.9239, 0.9239, 0.9239, 0.9239, 0.9239], [0.9239, 0.9239, 0.9239, 0.9239, 0.9239], [0.9239, 0.9239, 0.9239, 0.9239, 0.9239], [0.9239, 0.9239, 0.9239, 0.9239, 0.9239], [0.9239, 0.9239, 0.9239, 0.9239, 0.9239], [0.9239, 0.9239, 0.9239, 0.9239, 0.9239]]

        # 构造提示词
        dataset_info_path = os.path.join(get_project_root(),
                                         r"data/benchmarks/nl2workflow/datasets/datasets_info.json")
        with open(dataset_info_path, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)[dataset_name]
        parameters_name = [key for key in self.search_space]
        observed_configs = pd.DataFrame(x_observed, columns=parameters_name)
        observed_scores = pd.DataFrame(y_observed, columns=["score"])
        cases = pd.concat([observed_configs, observed_scores], axis=1)
        candidate_points_score = []
        for candidate_point in candidate_points:
            scores = []
            messages = []
            for i in range(5):
                np.random.seed(i)
                shuffled_indices = np.random.permutation(cases.index)
                new_cases = cases.loc[shuffled_indices]
                if len(new_cases) > 10:
                    new_cases = new_cases.head(10)
                # 构造示例字符串
                cases_str = ""
                for index, row in new_cases.iterrows():
                    row_col = row.drop("score")
                    c_str = ", ".join([f"{k}: {v}" for k, v in row_col.items()])
                    cases_str += f"\nHyperparameter configuration: {c_str}\nPerformance: ## {row['score']} ##"
                evaluated_configuration = ", ".join(
                    [f"{k}: {v}" for k, v in zip(parameters_name, candidate_point)])
                prompt = SURROGATE_MODEL_PROMPT.format(
                    model=model,
                    metric="accuracy",
                    task="classification",
                    classes_number=dataset_info["classes_number"],
                    samples_number=dataset_info["num_samples"],
                    features_number=dataset_info["features_number"],
                    categorical_features_number=dataset_info["categorical_features_number"],
                    continuous_features_number=dataset_info["numeric_features_number"],
                    evaluated_configuration=evaluated_configuration,
                    cases=cases_str
                )
                messages.append([
                    {"role": "system", "content": "You are an AI assistant that helps people find information."},
                    {"role": "user", "content": prompt}
                ])
            llm_responses = asyncio.run(self.predict(messages=messages, temperature=0.8, max_tokens=500, top_p=0.95))
            for content in llm_responses:
                # 解析
                try:
                    scores.append(float(content))
                except:
                    score = re.findall(r"## (-?[\d.]+) ##", content)
                    if len(score) == 1:
                        scores.append(float(score[0]))
                    else:
                        scores.append(np.nan)
            candidate_points_score.append(scores)

        return candidate_points_score

    def evaluate_candidate_points_bench(self, x_observed, y_observed, candidate_points, model, dataset_name):
        # return [[0.9239, 0.9239, 0.9239, 0.9239, 0.9239], [0.9239, 0.9239, 0.9239, 0.9239, 0.9239], [0.9239, 0.9239, 0.9239, 0.9239, 0.9239], [0.9239, 0.9239, 0.9239, 0.9239, 0.9239], [0.9239, 0.9239, 0.9239, 0.9239, 0.9239], [0.9239, 0.9239, 0.9239, 0.9239, 0.9239], [0.9239, 0.9239, 0.9239, 0.9239, 0.9239], [0.9239, 0.9239, 0.9239, 0.9239, 0.9239], [0.9239, 0.9239, 0.9239, 0.9239, 0.9239], [0.9239, 0.9239, 0.9239, 0.9239, 0.9239]]

        # 构造提示词
        with open(os.path.join(get_project_root(), r"data/benchmarks/hpo_bench/hpo_bench.json"), 'r') as f:
            hyperparameter_constraints = json.load(f)[model]
        task = openml.tasks.get_task(DATASET_MAP[dataset_name][1])
        dataset_ = task.get_dataset()
        X, y, categorical_mask, _ = dataset_.get_data(target=dataset_.default_target_attribute)
        hyperparameter_constraints_str = ""
        for key, value in hyperparameter_constraints.items():
            hyperparameter_constraints_str += f"The range of values of '{key}' is: {value}\n"

        parameters_names = [parameters_name for parameters_name in hyperparameter_constraints]

        observed_configs = pd.DataFrame(x_observed, columns=parameters_names)
        observed_scores = pd.DataFrame(y_observed, columns=["score"])
        cases = pd.concat([observed_configs, observed_scores], axis=1)
        # cases = cases.sort_values(by="score", ascending=True)


        candidate_points_score = []
        for candidate_point in candidate_points:
            scores = []
            messages = []
            for i in range(5):
                np.random.seed(i)
                shuffled_indices = np.random.permutation(cases.index)
                new_cases = cases.loc[shuffled_indices]
                if len(new_cases) > 10:
                    new_cases = new_cases.head(10)
                # 构造示例字符串
                cases_str = ""
                for index, row in new_cases.iterrows():
                    row_col = row.drop("score")
                    c_str = ", ".join([f"{k}: {v}" for k, v in row_col.items()])
                    cases_str += f"\nHyperparameter configuration: {c_str}\nPerformance: ## {row['score']} ##"
                evaluated_configuration = ", ".join(
                    [f"{k}: {v}" for k, v in zip(parameters_names, candidate_point)])
                prompt = SURROGATE_MODEL_PROMPT.format(
                    model=MODEL_MAP[model],
                    metric="accuracy",
                    task='classification',
                    classes_number=len(np.unique(y)),
                    samples_number=X.shape[0],
                    features_number=X.shape[1],
                    categorical_features_number=len(categorical_mask),
                    continuous_features_number=X.shape[1] - len(categorical_mask),
                    evaluated_configuration=evaluated_configuration,
                    cases=cases_str
                )
                messages.append([
                    {"role": "system", "content": "You are an AI assistant that helps people find information."},
                    {"role": "user", "content": prompt}
                ])
            llm_responses = asyncio.run(self.predict(messages=messages, temperature=0.8, max_tokens=500, top_p=0.95))
            for content in llm_responses:
                try_times = 0
                while try_times < 3:
                    try:
                        # 解析
                        try:
                            scores.append(float(content))
                        except:
                            score = re.findall(r"## (-?[\d.]+) ##", content)
                            if len(score) == 1:
                                scores.append(float(score[0]))
                            else:
                                scores.append(np.nan)
                        break
                    except Exception as e:
                        logger.info(f"Error: {e}")

                        content, message, input_tokens, output_tokens, price = self.llm.predict(messages=messages[0],
                                                                                                temperature=0.8,
                                                                                                top_p=0.95)
                        self.input_tokens += input_tokens
                        self.output_tokens += output_tokens

                    try_times += 1



            candidate_points_score.append(scores)

        return candidate_points_score

    def max_acq_function(self, y_observed, candidate_points, candidate_points_score):
        """
        贝叶斯优化算法中的采样函数（Acquisition Function）：这个函数的作用是用来选择一组可能的超参数
        :return:
        """
        all_preds = np.array(candidate_points_score).astype(float)
        y_mean = np.nanmean(all_preds, axis=1)
        y_std = np.nanstd(all_preds, axis=1)
        # Capture failed calls - impute None with average predictions
        y_mean[np.isnan(y_mean)]  = np.nanmean(y_mean)
        y_std[np.isnan(y_std)]  = np.nanmean(y_std)
        y_std[y_std<1e-5] = 1e-5  # replace small values to avoid division by zero
        y_mean = np.array(y_mean)
        y_std = np.array(y_std)
        # Capture failed calls - impute None with average predictions
        best_fval = np.max(y_observed)
        delta = y_mean - best_fval
        with np.errstate(divide='ignore'):  # handle y_std=0 without warning
            Z = delta/y_std
        ei = np.where(y_std>0, delta * norm.cdf(Z) + y_std * norm.pdf(Z), 0)
        best_point_index = np.argmax(ei)
        best_point = candidate_points[best_point_index]
        return best_point

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
    llambo = LLAMBO()
    llambo.run_nl2workflow_experiment()


