from deap import base, creator, tools
import numpy as np
import random
random.seed(42)
import xgboost as xgb
import os
import openml
from llm.get_llm import get_llm
from utils.utils import get_project_root
from utils.logs import logger
from baselines.baseline import Baseline
from data.benchmarks.hpob.search_spaces_info import SEARCH_SPACE_INFO
from data.benchmarks.hpob.datasets_info import DATASETS_INFO
import json
import asyncio
from config import MODEL_NAME
import re

MODEL_NAME_MAP = {
    "Meta-Llama-3-8B-Instruct": "llama3-8b",
    "Meta-Llama-3-70B-Instruct": "llama3-70b",
    "Qwen2.5-14B-Instruct": "qwen2_5-14b",
    "Qwen2.5-72B-Instruct": "qwen2_5-72b",
    "qwen2.5-72b-instruct": "qwen2_5-72b",
    "gpt-4o": "gpt-4o",
    "deepseek-reasoner": "deepseek-r1",
    "deepseek-chat": "deepseek-v3",
    "glm-4-9b-chat": "glm4-9b",
}

PROMPT_TEMPLATE_fewshot = """
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
3. Different hyperparameter configurations must be given. Do not repeat the same configuration.
"""

PROMPT_TEMPLATE_zeroshot = """
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

# Output Format
Format strictly follows this template: 
```
{output_format}```

# Remember
1. Please give the {configurations_numbers} configurations in a strictly direct output format, no nonsense!
2. Different hyperparameter configurations must be given. Do not repeat the same configuration.
"""

class Evo_Context(Baseline):

    def __init__(self, population_size=5, mutation_rate=0.8, cross_rate=0.2):
        super().__init__()
        self.population_size = population_size  # 种群的大小
        self.mutation_rate = mutation_rate  # 变异率
        self.cross_rate = cross_rate #交叉概率
        self.llm = get_llm()
        self.llm_method = True

        # 5或7
        self.top_K = 5
        # self.best = 3
        # 0，0.1，0.3，0.5，0.7，0.9
        temperatures = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        self.t = 0
        self.temperature = temperatures[self.t]
        # 冷启动方式, llm or random
        self.cold_starting = "llm"
        # 示例数据文本化
        self.textualize = True
        #示例集的更新方式,0: 所有历史优化数据 + 进化出来的配置, 1: 进化出来的配置 + best, 2: 进化出来的配置
        self.type = 1
        #- 初始示例集, 0:直接让LLM生成初始示例集, 1:基于少量历史数据让LLM生成初始示例集, 2: 少量历史数据作为初始示例集
        self.starting = 2
        # 提示词中是否提供分数
        self.is_score = True
        # 示例排序方式，0: 从大到小, 1: 从小到大, 2: 随机
        self.sort = 0

        self.evolute_type = "evo"
        self.is_deduplicate = True
        self.n = 1
        self.is_verify = False

        self.name = f"evo_context_t{self.t}_type{self.type}_starting{self.starting}_{self.evolute_type}_score{self.is_score}_sort{self.sort}({MODEL_NAME_MAP[MODEL_NAME]})"
        self.toolbox = self.initialize_toolbox()

    def run_method(self, seed_id, search_space_id, dataset_id):
        # 加载评估代理模型
        surrogate_name = 'surrogate-' + search_space_id + '-' + dataset_id
        bst_surrogate = xgb.Booster()
        bst_surrogate.load_model(self.surrogates_dir + surrogate_name + '.json')
        self.bst_surrogate = bst_surrogate
        dataset = DATASETS_INFO[dataset_id]
        search_space = SEARCH_SPACE_INFO[search_space_id]
        self.search_space = search_space
        self.search_space_id = search_space_id
        self.dim = len(search_space["parameters_name"])
        y_min = self.surrogates_stats[surrogate_name]["y_min"]
        y_max = self.surrogates_stats[surrogate_name]["y_max"]

        x_observed = []
        y_observed = []
        max_accuracy_history = []

        self.history_configs = []
        self.benchmark = "hpob"
        init_x = []
        init_y = []
        for i in range(len(self.init_x)):
            x = self.init_x[i]
            individual = creator.Individual(x)
            individual.fitness.values = self.init_y[i],
            init_y.append(individual.fitness.values[0])
            init_x.append(individual)

        if self.starting != 2:
            if self.starting == 1:
                self.cases, _ = self.get_context_cases(init_x + x_observed, fewshot=True)
                configs = self.get_configs(search_space_id, dataset, k=self.top_K - 2)
            else:
                self.cases = []
                if self.cold_starting == "llm":
                    configs = self.get_configs(search_space_id, dataset, k=self.top_K-2)
                else:
                    configs = []
                    for i in range(self.top_K-2):
                        configs.append(self.random_search(search_space))
            self.history_configs += configs
            for c in configs:
                try:
                    individual = creator.Individual(c)
                    individual.fitness.values = self.evaluate(c)
                    y_observed.append(individual.fitness.values[0])
                    x_observed.append(individual)
                    best_f = max(y_observed)
                    # 记录当前最优的分数
                    max_accuracy_history.append(best_f)
                except Exception as e:
                    logger.info(f"config: {c}, Error:{e}")
                    continue

        logger.info(f"初始种群: {len(x_observed)}")

        for i in range(int(self.n_trials/self.top_K*2)):
            logger.info(f"第{i}轮")
            if self.starting != 0:
                self.cases, evo_configs = self.get_context_cases(init_x + x_observed)
            else:
                self.cases, evo_configs = self.get_context_cases(x_observed)
            self.history_configs += evo_configs
            for c in evo_configs:
                y_observed.append(c.fitness.values[0])
                x_observed.append(c)
                max_accuracy_history.append(max(y_observed))
                if len(max_accuracy_history) >= self.n_trials:
                    break
            if len(max_accuracy_history) >= self.n_trials:
                break
            configs = self.get_configs(search_space_id, dataset, self.top_K-len(evo_configs))
            self.history_configs += configs
            for c in configs:
                if i > len(configs)-1:
                    continue
                # 获取下一个要探索的超参数配置
                try:
                    individual = creator.Individual(c)
                    individual.fitness.values = self.evaluate(c)
                    y_observed.append(individual.fitness.values[0])
                    x_observed.append(individual)
                    # 记录当前最优的分数
                    max_accuracy_history.append(max(y_observed))
                    if len(max_accuracy_history) >= self.n_trials:
                        break
                except Exception as e:
                    logger.info(f"config: {c}, Error:{e}")
                    continue
            if len(max_accuracy_history) >= self.n_trials:
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
        from utils.utils import get_project_root
        import os
        import json
        from data.benchmarks.hpo_bench.tabular_benchmarks import DATASET_MAP, HPOExpRunner
        dataset = DATASET_MAP[dataset_name][0]
        benchmark = HPOExpRunner(model, dataset, seed_id)
        self.hpobench_evaluate = benchmark
        with open(os.path.join(get_project_root(), r"data/benchmarks/hpo_bench/hpo_bench.json"), 'r') as f:
            search_space = json.load(f)[model]
        self.search_space = search_space
        self.search_space_id = model
        parameters_name = [key for key in search_space.keys()]
        self.parameters_name = parameters_name
        x_observed = []
        y_observed = []
        max_accuracy_history = []
        self.history_configs = []
        self.benchmark = "hpo_bench"

        init_configs = benchmark.generate_initialization()
        init_x = []
        init_y = []
        for i in range(len(init_configs)):
            config = init_configs[i]
            x = [config[key] for key in config.keys()]
            individual = creator.Individual(x)
            individual.fitness.values = self.evaluate(x)
            init_y.append(individual.fitness.values[0])
            init_x.append(individual)
        if self.starting != 2:
            if self.starting == 1:
                self.cases, _ = self.get_context_cases(init_x + x_observed, fewshot=True)
                configs = self.get_configs(model, str(DATASET_MAP[dataset_name][1]), self.top_K - 2)
            else:
                self.cases = []
                if self.cold_starting=="llm":
                    configs = self.get_configs(model, str(DATASET_MAP[dataset_name][1]), self.top_K-2)
                else:
                    configs = []
                    for i in range(self.top_K-2):
                        configs.append(self.random_search(self.search_space))
            self.history_configs += configs
            for c in configs:
                try:
                    individual = creator.Individual(c)
                    individual.fitness.values = self.evaluate(c)
                    y_observed.append(individual.fitness.values[0])
                    x_observed.append(individual)
                    # 记录当前最优的分数
                    max_accuracy_history.append(max(y_observed))
                except Exception as e:
                    logger.info(f"config: {c}, Error:{e}")
                    continue

        logger.info(f"初始种群: {len(x_observed)}")


        for i in range(int(self.n_trials/self.top_K*2)):
            logger.info(f"第{i}轮")
            self.cases, evo_configs = self.get_context_cases(init_x + x_observed)
            self.history_configs += evo_configs
            for c in evo_configs:
                y_observed.append(c.fitness.values[0])
                x_observed.append(c)
                max_accuracy_history.append(max(y_observed))
                if len(max_accuracy_history) >= self.n_trials:
                    break
            if len(max_accuracy_history) >= self.n_trials:
                break
            configs = self.get_configs(model, str(DATASET_MAP[dataset_name][1]), self.top_K-len(evo_configs))
            self.history_configs += configs
            for c in configs:
                if i > len(configs)-1:
                    continue

                # 获取下一个要探索的超参数配置
                try:
                    individual = creator.Individual(c)
                    individual.fitness.values = self.evaluate(c)
                    y_observed.append(individual.fitness.values[0])
                    x_observed.append(individual)
                    # 记录当前最优的分数
                    max_accuracy_history.append(max(y_observed))
                    if len(max_accuracy_history) >= self.n_trials:
                        break
                except Exception as e:
                    logger.info(f"config: {c}, Error:{e}")
                    continue
            if len(max_accuracy_history) >= self.n_trials:
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

    def run_nl2workflow_method(self, seed_id, model, dataset_name):
        with open(os.path.join(get_project_root(), "data/benchmarks/nl2workflow/search_space.json"), 'r', encoding='utf-8') as f:
            search_space = json.load(f)
        for s in search_space:
            if s["algorithm"] == model:
                self.search_space = s["search_space"]
                break
        self.eval_params = (model, dataset_name)
        x_observed = []
        y_observed = []
        max_accuracy_history = []
        self.history_configs = []
        self.benchmark = "nl2workflow"
        init_x = []
        init_y = []
        for i in range(self.population_size):
            new_x = self.random_search(self.search_space)
            individual = creator.Individual(new_x)
            individual.fitness.values = self.evaluate(new_x)
            init_y.append(individual.fitness.values[0])
            init_x.append(individual)

        if self.starting != 2:
            if self.starting == 1:
                self.cases, _ = self.get_context_cases(init_x + x_observed, fewshot=True)
                configs = self.get_configs(model, dataset_name, self.top_K - 2)
            else:
                self.cases = []
                if self.cold_starting == "llm":
                    configs = self.get_configs(model, dataset_name, self.top_K-2)
                else:
                    configs = []
                    for i in range(self.top_K-2):
                        configs.append(self.random_search(self.search_space))

            self.history_configs += configs
            for c in configs:
                try:
                    individual = creator.Individual(c)
                    individual.fitness.values = self.evaluate(c)
                    y_observed.append(individual.fitness.values[0])
                    x_observed.append(individual)
                    # 记录当前最优的分数
                    max_accuracy_history.append(max(y_observed))
                except Exception as e:
                    logger.info(f"config: {c}, Error:{e}")
                    continue

        logger.info(f"初始种群: {len(x_observed)}")


        for i in range(int(self.n_trials/self.top_K*2)):
            logger.info(f"第{i}轮")
            self.cases, evo_configs = self.get_context_cases(init_x + x_observed)
            self.history_configs += evo_configs
            for c in evo_configs:
                y_observed.append(c.fitness.values[0])
                x_observed.append(c)
                max_accuracy_history.append(max(y_observed))
                if len(max_accuracy_history) >= self.n_trials:
                    break
            if len(max_accuracy_history) >= self.n_trials:
                break
            configs = self.get_configs(model, dataset_name, self.top_K-len(evo_configs))
            self.history_configs += configs
            for c in configs:
                if i > len(configs)-1:
                    continue

                # 获取下一个要探索的超参数配置
                try:
                    individual = creator.Individual(c)
                    individual.fitness.values = self.evaluate(c)
                    y_observed.append(individual.fitness.values[0])
                    x_observed.append(individual)
                    # 记录当前最优的分数
                    max_accuracy_history.append(max(y_observed))
                    if len(max_accuracy_history) >= self.n_trials:
                        break
                except Exception as e:
                    logger.info(f"config: {c}, Error:{e}")
                    continue
            if len(max_accuracy_history) >= self.n_trials:
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

    def initialize_toolbox(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        toolbox.register("mate", tools.cxUniform, indpb=self.cross_rate)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=self.mutation_rate)
        toolbox.register("select", tools.selRoulette, k=2, fit_attr='fitness')
        return toolbox

    def evaluate(self, new_x):
        if self.benchmark == "hpob":
            new_x_arry = np.array(new_x)
            x_q = xgb.DMatrix(new_x_arry.reshape(-1, self.dim))
            new_y_arry = self.bst_surrogate.predict(x_q)
            new_y = float(new_y_arry[0])
            return new_y,
        elif self.benchmark == "hpo_bench":
            new_x_pro = {self.parameters_name[i]: new_x[i] for i in range(len(new_x))}
            return self.hpobench_evaluate.evaluate_point(new_x_pro),
        else:
            from data.benchmarks.nl2workflow.eval import evaluate
            return evaluate(new_x, self.eval_params[0], self.eval_params[1]),

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

    def get_configs(self, search_space_id, dataset, k):
        eval_configs = []
        # 一次LLM返回的配置数量
        msgs = []
        for i in range(1):
            msgs.append(self.get_prompt(search_space_id, dataset, k=k*self.n))

        responses = asyncio.run(self.predict(messages=msgs, temperature=self.temperature, top_p=1))
        logger.info(f"llm end:{len(responses)}")
        for idx, response in enumerate(responses):
            logger.info(f"llm response: {idx}")
            logger.info(response)
            try_times = 0
            while try_times < 5:
                try_times += 1
                try:
                    # 即使生成k*self.n，也只是获取前k个配置
                    suggest_config, suggest_eval_config = self.parse_configs(response, topk=k)
                    if len(suggest_eval_config) > 0:
                            for i in range(len(suggest_eval_config)):
                                if self.is_verify:
                                    suggest_eval_config[i] = self.verify(
                                        [suggest_eval_config[i][key] for key in suggest_eval_config[i]],
                                        search_space_id)
                                else:
                                    suggest_eval_config[i] = [suggest_eval_config[i][key] for key in suggest_eval_config[i]]
                                if not self.is_deduplicate:
                                    if suggest_eval_config[i] not in eval_configs and (
                                            suggest_eval_config[i] not in self.history_configs):
                                        eval_configs.append(suggest_eval_config[i])
                                    else:
                                        logger.info(f"重复: {suggest_eval_config[i]}")
                                else:
                                    eval_configs.append(suggest_eval_config[i])

                                if len(eval_configs) >= k:
                                    break
                                else:
                                    logger.info(f"获得的配置数为：{len(eval_configs)}")
                            if len(eval_configs) >= k:
                                break
                    else:
                        raise Exception("配置数量为0")
                except Exception as e:
                    logger.error(f"解析配置失败，错误信息为：{e}")
                if len(eval_configs) >= k:
                    break
                response, message, input_tokens, output_tokens, price = self.llm.predict(messages=msgs[idx],
                                                                                         temperature=0.7, top_p=1)
                self.input_tokens += input_tokens
                self.output_tokens += output_tokens

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
                if self.textualize:
                    if not self.is_score:
                        for i in range(len(self.cases)):
                            case_str = ""
                            for j in range(len(parameters_name)):
                                if j == len(parameters_name) - 1:
                                    case_str += f"{parameters_name[j]} is {self.cases[i][j]}."
                                else:
                                    case_str += f"{parameters_name[j]} is {self.cases[i][j]}. "
                            cases_string += f"Configuration {i + 1}: {case_str}\n"
                    else:
                        for i in range(len(self.cases)):
                            case_str = ""
                            for j in range(len(parameters_name)):
                                if j == len(parameters_name) - 1:
                                    case_str += f"{parameters_name[j]} is {self.cases[i][j]}."
                                else:
                                    case_str += f"{parameters_name[j]} is {self.cases[i][j]}. "

                            cases_string += f"Configuration {i + 1}: {case_str}\n"
                            cases_string += f"Performance {i + 1}: {self.cases[i].fitness.values[0]}\n"
                else:
                    cases_string = json.dumps(self.cases, ensure_ascii=False)

                logger.info(f"cases_string:\n{cases_string}")
                output_format = ""
                for i in range(k):
                    output_format += f"Configuration {i + 1}" + ": {{parameter_1_name}} is {{parameter_1_value}}. {{parameter_2_name}} is {{parameter_2_value}}...{{parameter_n_name}} is {{parameter_n_value}}.\n"

                if self.starting==0 and len(self.cases)==0:
                    prompt = PROMPT_TEMPLATE_zeroshot.format(
                        ml_model=ml_model,
                        parameters_number=len(parameters_name),
                        parameters=','.join(parameters_name),
                        task=task_description,
                        output_format=output_format,
                        configurations_numbers=k,
                        configurations_ranges=configurations_ranges
                    )
                else:
                    prompt = PROMPT_TEMPLATE_fewshot.format(
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
                dataset_info_path = os.path.join(get_project_root(), r"data/benchmarks/nl2workflow/datasets/datasets_info.json")
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
                if self.textualize:
                    if not self.is_score:
                        for i in range(len(self.cases)):
                            case_str = ""
                            for j in range(len(parameters_name)):
                                if j == len(parameters_name) - 1:
                                    case_str += f"{parameters_name[j]} is {self.cases[i][j]}."
                                else:
                                    case_str += f"{parameters_name[j]} is {self.cases[i][j]}. "
                            cases_string += f"Configuration {i + 1}: {case_str}\n"
                    else:
                        for i in range(len(self.cases)):
                            case_str = ""
                            for j in range(len(parameters_name)):
                                if j == len(parameters_name) - 1:
                                    case_str += f"{parameters_name[j]} is {self.cases[i][j]}."
                                else:
                                    case_str += f"{parameters_name[j]} is {self.cases[i][j]}. "

                            cases_string += f"Configuration {i + 1}: {case_str}\n"
                            cases_string += f"Performance {i + 1}: {self.cases[i].fitness.values[0]}\n"
                else:
                    cases_string = json.dumps(self.cases, ensure_ascii=False)
                logger.info(f"cases_string:\n{cases_string}")
                output_format = ""
                for i in range(k):
                    output_format += f"Configuration {i + 1}" + ": {{parameter_1_name}} is {{parameter_1_value}}. {{parameter_2_name}} is {{parameter_2_value}}...{{parameter_n_name}} is {{parameter_n_value}}.\n"

                if self.starting == 0 and len(self.cases) == 0:
                    prompt = PROMPT_TEMPLATE_zeroshot.format(
                        ml_model=search_space_id,
                        parameters_number=len(parameters_name),
                        parameters=','.join(parameters_name),
                        task=task_description,
                        output_format=output_format,
                        configurations_numbers=k,
                        configurations_ranges=configurations_ranges
                    )
                else:
                    prompt = PROMPT_TEMPLATE_fewshot.format(
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
                parameters_name = self.parameters_name
                with open(os.path.join(get_project_root(), r"data/benchmarks/hpo_bench/hpo_bench.json"), 'r') as f:
                    search_space = json.load(f)[search_space_id]

                configurations_ranges = ""
                for parameter in order_list_info[search_space_id]:
                    configurations_ranges += f"- {parameter}: It is a categorical variable, must take value in {search_space[parameter]}.\n"

                cases_string = ""
                if self.textualize:
                    if not self.is_score:
                        for i in range(len(self.cases)):
                            case_str = ""
                            for j in range(len(parameters_name)):
                                if j == len(parameters_name) - 1:
                                    case_str += f"{parameters_name[j]} is {self.cases[i][j]}."
                                else:
                                    case_str += f"{parameters_name[j]} is {self.cases[i][j]}. "
                            cases_string += f"Configuration {i + 1}: {case_str}\n"
                    else:
                        for i in range(len(self.cases)):
                            case_str = ""
                            for j in range(len(parameters_name)):
                                if j == len(parameters_name) - 1:
                                    case_str += f"{parameters_name[j]} is {self.cases[i][j]}."
                                else:
                                    case_str += f"{parameters_name[j]} is {self.cases[i][j]}. "

                            cases_string += f"Configuration {i + 1}: {case_str}\n"
                            cases_string += f"Performance {i + 1}: {self.cases[i].fitness.values[0]}\n"
                else:
                    cases_string = json.dumps(self.cases, ensure_ascii=False)
                logger.info(f"cases_string:\n{cases_string}")
                output_format = ""
                for i in range(k):
                    output_format += f"Configuration {i + 1}" + ": {{parameter_1_name}} is {{parameter_1_value}}. {{parameter_2_name}} is {{parameter_2_value}}...{{parameter_n_name}} is {{parameter_n_value}}.\n"

                if self.starting == 0 and len(self.cases) == 0:
                    prompt = PROMPT_TEMPLATE_zeroshot.format(
                        ml_model=MODEL_MAP[search_space_id],
                        parameters_number=len(parameters_name),
                        parameters=','.join(parameters_name),
                        task=task_description,
                        output_format=output_format,
                        configurations_numbers=k,
                        configurations_ranges=configurations_ranges
                    )
                else:
                    prompt = PROMPT_TEMPLATE_fewshot.format(
                        ml_model=MODEL_MAP[search_space_id],
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

    def verify(self, config, search_space_id):
        "验证超参数配置是否符合范围，若不符合则做出相应处理"
        new_x = []
        if self.benchmark=="hpob":
            search_space = SEARCH_SPACE_INFO[search_space_id]
            for index, value in enumerate(search_space["parameters_name"]):
                parameter = search_space[value]
                if parameter["type"] == "categorical":
                    # 如果x的值不在search_space[value]列表中，采用最近邻替换该值
                    if config[index] not in parameter["categories"]:
                        new_x.append(min(parameter["categories"], key=lambda x: abs(x - config[index]) / max(abs(x), abs(config[index]))))
                    else:
                        new_x.append(config[index])
                else:
                    if config[index] < parameter["low"]:
                        new_x.append(parameter["low"])
                    elif config[index] > parameter["high"]:
                        new_x.append(parameter["high"])
                    else:
                        new_x.append(config[index])
        else:
            with open(os.path.join(get_project_root(), r"data/benchmarks/hpo_bench/hpo_bench.json"), 'r', encoding='utf-8') as f:
                search_spaces = json.load(f)
            search_space = search_spaces[search_space_id]
            for index, value in enumerate(search_space):
                # 如果x的值不在search_space[value]列表中，采用最近邻替换该值
                if config[index] not in search_space[value]:
                    new_x.append(min(search_space[value], key=lambda x: abs(x - config[index]) / max(x, config[index])))
                else:
                    new_x.append(config[index])

        if json.dumps(config) != json.dumps(new_x):
            logger.info(f"config {config} is out of range, new config is {new_x}")

        return new_x

    def parse_configs(self, response, topk=5):
            pattern_0 = re.compile(r"Configuration \d+: (.*)\.\n")
            suggest_configs = []
            configs = []
            groups = re.findall(pattern_0, response + "\n")
            for t in groups[:topk]:
                configs.append(t)
                kvs = t.split(". ")
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

    def random_search(self, search_space):
        if self.benchmark=="hpob":
            for i in range(10):
                parameters_name = search_space["parameters_name"]  # 获取的参数列表
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

                if self.is_verify:
                    x_new = self.verify(x_new, self.search_space_id)

                if x_new not in self.history_configs or i == 9 or self.is_deduplicate:
                    break

            return x_new

        elif self.benchmark=="nl2workflow":
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

        else:
            config = []
            for i in range(10):
                x_new = {}
                for parameter_name in search_space:
                    categories = search_space[parameter_name]
                    random_parameter = random.choice(categories)
                    x_new[parameter_name] = random_parameter

                config = [x_new[key] for key in x_new]

                if self.is_verify:
                    config = self.verify(config, self.search_space_id)

                if config not in self.history_configs or i == 9 or self.is_deduplicate:
                    break
            return config

    def mutate(self, individual, search_space):
        for i in range(10):
            x_new = []
            search_space_new = []
            for item in search_space:
                search_space_new.append(search_space[item])

            for i in range(len(individual)):
                categories = search_space_new[i]
                if np.random.rand() < self.mutation_rate:
                    random_parameter = random.choice([cat for cat in categories if cat != individual[i]])
                    x_new.append(random_parameter)
                else:
                    x_new.append(individual[i])

            if self.is_verify:
                x_new = self.verify(x_new, self.search_space_id)

            if x_new not in self.history_configs or i == 9 or self.is_deduplicate:
                break

        return creator.Individual(x_new)

    def mutate_nl2workflow(self, individual, search_space):
        x_new = []
        parameters_name = [key for key in search_space]
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_rate:
                if search_space[parameters_name[i]]['_type'] == "uniform":
                    max_times = 10
                    while max_times > 0:
                        max_times -= 1
                        new_value = individual[i] + random.gauss(0, 1)
                        if new_value >= search_space[parameters_name[i]]['_value'][0] and new_value <= search_space[parameters_name[i]]['_value'][1]:
                            break
                    if new_value < search_space[parameters_name[i]]['_value'][0]:
                        new_value = search_space[parameters_name[i]]['_value'][0]
                    elif new_value > search_space[parameters_name[i]]['_value'][1]:
                        new_value = search_space[parameters_name[i]]['_value'][1]
                    x_new.append(new_value)
                elif search_space[parameters_name[i]]['_type'] == "randint":
                    x_new.append(random.randint(search_space[parameters_name[i]]['_value'][0], search_space[parameters_name[i]]['_value'][1]))
                elif search_space[parameters_name[i]]['_type'] == "choice":
                    categories = search_space[parameters_name[i]]['_value']
                    max_times = 10
                    while max_times > 0:
                        random_parameter = random.choice(categories)
                        if random_parameter != individual[i]:
                            x_new.append(random_parameter)
                            break
                        elif max_times == 1:
                            x_new.append(individual[i])
                            break
                        max_times -= 1

                elif search_space[parameters_name[i]]['_type'] == "loguniform":
                    low = search_space[parameters_name[i]]['_value'][0]
                    high = search_space[parameters_name[i]]['_value'][1]
                    x_new.append(random.uniform(low, high))
            else:
                x_new.append(individual[i])
        return creator.Individual(x_new)

    def get_context_cases(self, history_x, fewshot=False):
        """
        选择2个，进化3个(根据不同的交叉概率)
        """
        if fewshot:
            cases = tools.selBest(history_x, k=self.top_K)
            return cases, []
        else:
            if self.evolute_type=="evo":
                ind1, ind2 = self.toolbox.select(history_x)
                child1, child2 = self.toolbox.clone(ind1), self.toolbox.clone(ind2)
                self.toolbox.mate(child1, child2)
                if self.benchmark=="hpob":
                    self.toolbox.mutate(child1)
                    self.toolbox.mutate(child2)
                elif self.benchmark=="nl2workflow":
                    child1 = self.mutate_nl2workflow(child1, self.search_space)
                    child2 = self.mutate_nl2workflow(child2, self.search_space)
                else:
                    child1 = self.mutate(child1, self.search_space)
                    child2 = self.mutate(child2, self.search_space)
            else:
                raise ValueError("evolute_type must be 'evo'")

            child1.fitness.values = self.evaluate(child1)
            child2.fitness.values = self.evaluate(child2)

            logger.info(f"原本的最佳{self.top_K}:")
            for item in tools.selBest(history_x, k=self.top_K):
                logger.info(f"fitness: {item.fitness.values[0]}, case: {item}")
            history_x.append(child1)
            history_x.append(child2)
            logger.info(f"进化后最佳{self.top_K}:")
            for item in tools.selBest(history_x, k=self.top_K):
                logger.info(f"fitness: {item.fitness.values[0]}, case: {item}")
            if self.type == 0:
                cases = tools.selBest(history_x, k=len(history_x))
            elif self.type == 1:
                cases = tools.selBest(history_x, k=self.top_K)
            else:
                cases = [child1, child2]
        if self.sort == 0:
            res = sorted(cases, key=lambda x: x.fitness.values[0], reverse=True)
        elif self.sort == 1:
            res = sorted(cases, key=lambda x: x.fitness.values[0], reverse=False)
        else:
            random.shuffle(cases)
            res = cases
        return res, [child1, child2]

if __name__ == '__main__':
    m = Evo_Context()
    m.run_nl2workflow_experiment()