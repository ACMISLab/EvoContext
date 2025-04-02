import numpy as np
import random
random.seed(42)
import openml
import xgboost as xgb
from baselines.baseline import Baseline
from llm.get_llm import get_llm
from prompts.EA_llm_prompt import EVOLVE_PROMPT_CONTEXT
import json
from deap import base, creator, tools
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

class EvoPrompt(Baseline):

    def __init__(self):
        super().__init__()
        self.name = f"EvoPrompt({MODEL_NAME_MAP[MODEL_NAME]})"
        self.llm_method = True
        self.llm = get_llm()
        self.population_size = 5
        self.toolbox = self.initialize_toolbox()

    def run_method(self, seed_id, search_space_id, dataset_id):
        from data.benchmarks.hpob.search_spaces_info import SEARCH_SPACE_INFO
        from data.benchmarks.hpob.datasets_info import DATASETS_INFO
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

        for i in range(self.population_size):
            x = self.random_search(search_space)
            individual = creator.Individual(x)
            individual.fitness.values = self.evaluate(x)
            y_observed.append(individual.fitness.values[0])
            x_observed.append(individual)
            # 记录当前最优的分数
            max_accuracy_history.append(max(y_observed))

        for i in range(self.n_trials):
            if i < self.population_size:
                continue
            ind1, ind2 = self.toolbox.select(x_observed)
            child = self.evolve(ind1, ind2)
            new_x = child
            new_x.fitness.values = self.evaluate(new_x)
            y_observed.append(new_x.fitness.values[0])
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
        from utils.utils import get_project_root
        import os
        import json
        dataset = DATASET_MAP[dataset_name][0]
        benchmark = HPOExpRunner(model, dataset, seed_id)
        with open(os.path.join(get_project_root(), r"data/benchmarks/hpo_bench/hpo_bench.json"), 'r') as f:
            search_space = json.load(f)[model]
        x_observed = []
        y_observed = []
        max_accuracy_history = []

        for i in range(self.population_size):
            config = self.random_search_hpo_bench(search_space)
            x = [config[key] for key in config]
            individual = creator.Individual(x)
            individual.fitness.values = benchmark.evaluate_point(config),
            y_observed.append(individual.fitness.values[0])
            x_observed.append(individual)
            # 记录当前最优的分数
            max_accuracy_history.append(max(y_observed))

        for i in range(self.n_trials):
            if i < self.population_size:
                continue
            ind1, ind2 = self.toolbox.select(x_observed)
            child, evaluate_point = self.evolve_hpo_bench(ind1, ind2, model, dataset_name)
            new_x = child
            new_x.fitness.values = benchmark.evaluate_point(evaluate_point),
            y_observed.append(new_x.fitness.values[0])
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

        for i in range(self.population_size):
            x = self.random_search_nl2workflow(self.search_space)
            individual = creator.Individual(x)
            individual.fitness.values = evaluate(x, model, dataset_name),
            y_observed.append(individual.fitness.values[0])
            x_observed.append(individual)
            max_accuracy_history.append(max(y_observed))

        for i in range(self.n_trials):
            if i < self.population_size:
                continue
            ind1, ind2 = self.toolbox.select(x_observed)
            try:
                new_x = self.evolve_nl2workflow(ind1, ind2, model, dataset_name)
            except:
                continue
            new_x.fitness.values = evaluate(new_x, model, dataset_name),
            y_observed.append(new_x.fitness.values[0])
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

    def initialize_toolbox(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        toolbox.register("select", tools.selRoulette, k=2, fit_attr='fitness')

        return toolbox

    def evaluate(self, new_x):
        new_x_arry = np.array(new_x)
        x_q = xgb.DMatrix(new_x_arry.reshape(-1, self.dim))
        new_y_arry = self.bst_surrogate.predict(x_q)
        new_y = float(new_y_arry[0])
        return new_y,

    def evolve(self, parent1, parent2):
        search_space = self.search_space
        dataset = self.dataset
        configuration1 = {}
        configuration2 = {}
        for i in range(len(search_space["parameters_name"])):
            configuration1[search_space["parameters_name"][i]] = parent1[i]
            configuration2[search_space["parameters_name"][i]] = parent2[i]

        target_characteristics = ".".join(
            [f'The "{i}" class size is {dataset["target_characteristics"][i][1]}' for i in
             dataset["target_characteristics"]])
        parameters = [search_space[p] for p in search_space["parameters_name"]]
        prompt = EVOLVE_PROMPT_CONTEXT.format(
            dataset_name=dataset["name"],
            classes_number=len(dataset["target_characteristics"]),
            num_samples=dataset["num_samples"],
            features_number=dataset["numeric_features"] + dataset["categorical_features"],
            numeric_features_number=dataset["numeric_features"],
            categorical_features_number=dataset["categorical_features"],
            target_characteristics=target_characteristics,
            ml_model=search_space["model_name"],
            parameters=json.dumps(parameters, ensure_ascii=False),
            configuration1=json.dumps(configuration1, ensure_ascii=False),
            configuration2=json.dumps(configuration2, ensure_ascii=False)
        )
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        max_try = 5
        for _ in range(max_try):
            content, message, input_tokens, output_tokens, price = self.llm.predict(messages=messages)
            messages.append(message)
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens

            tag_begin = "<configuration>"
            tag_end = "</configuration>"
            try:
                from utils.utils import load_json
                begin = content.find(tag_begin)
                end = content.find(tag_end)
                if begin != -1 and end != -1:
                    content = content[begin + len(tag_begin):end]
                    child = load_json(content)
                    x_new = []
                    tag = True
                    for parameter_name in search_space["parameters_name"]:
                        if parameter_name in child:
                            x_new.append(child[parameter_name])
                        else:
                            tag = False
                            break
                    if tag:
                        return creator.Individual(x_new)
                    messages.append({
                        "role": "user",
                        "content": f"Please generate a valid child configuration. missing parameters, the configured parameters need to contain {search_space['parameters_name']}"
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": "The final parameter configuration must be enclosed in <configuration> and </configuration>."
                    })
            except:
                messages.append({
                    "role": "user",
                    "content": "Please ensure the final parameter configuration must be enclosed in <configuration> and </configuration>."
                })
        raise Exception("Failed to generate a valid child configuration after {} attempts.".format(max_try))

    def evolve_nl2workflow(self, parent1, parent2, model, dataset_name):
        parameters = [key for key in self.search_space]
        configuration1 = {}
        configuration2 = {}
        for i in range(len(parameters)):
            configuration1[parameters[i]] = parent1[i]
            configuration2[parameters[i]] = parent2[i]

        dataset_info_path = os.path.join(get_project_root(),
                                         r"data/benchmarks/nl2workflow/datasets/datasets_info.json")
        with open(dataset_info_path, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)[dataset_name]

        prompt = EVOLVE_PROMPT_CONTEXT.format(
            dataset_name=dataset_name,
            classes_number=dataset_info["classes_number"],
            num_samples=dataset_info["num_samples"],
            features_number=dataset_info["features_number"],
            numeric_features_number=dataset_info["numeric_features_number"],
            categorical_features_number=dataset_info["categorical_features_number"],
            target_characteristics=dataset_info["target_characteristics"],
            ml_model=model,
            parameters=json.dumps(parameters, ensure_ascii=False),
            configuration1=json.dumps(configuration1, ensure_ascii=False),
            configuration2=json.dumps(configuration2, ensure_ascii=False)
        )
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        max_try = 5
        for _ in range(max_try):
            content, message, input_tokens, output_tokens, price = self.llm.predict(messages=messages)
            messages.append(message)
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens

            tag_begin = "<configuration>"
            tag_end = "</configuration>"
            try:
                from utils.utils import load_json
                begin = content.find(tag_begin)
                end = content.find(tag_end)
                if begin != -1 and end != -1:
                    content = content[begin + len(tag_begin):end]
                    child = load_json(content)
                    x_new = []
                    tag = True
                    for parameter_name in parameters:
                        if parameter_name in child:
                            x_new.append(child[parameter_name])
                        else:
                            tag = False
                            break
                    if tag:
                        return creator.Individual(x_new)
                    messages.append({
                        "role": "user",
                        "content": f"Please generate a valid child configuration. missing parameters, the configured parameters need to contain {search_space['parameters_name']}"
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": "The final parameter configuration must be enclosed in <configuration> and </configuration>."
                    })
            except:
                messages.append({
                    "role": "user",
                    "content": "Please ensure the final parameter configuration must be enclosed in <configuration> and </configuration>."
                })
        raise Exception("Failed to generate a valid child configuration after {} attempts.".format(max_try))

    def evolve_hpo_bench(self, parent1, parent2, model, dataset_name):
        with open(os.path.join(get_project_root(), r"data/benchmarks/hpo_bench/hpo_bench.json"), 'r') as f:
            hyperparameter_constraints = json.load(f)[model]
        configuration1 = {}
        configuration2 = {}
        parameters_names = [parameters_name for parameters_name in hyperparameter_constraints]
        for i in range(len(parameters_names)):
            configuration1[parameters_names[i]] = parent1[i]
            configuration2[parameters_names[i]] = parent2[i]

        task = openml.tasks.get_task(DATASET_MAP[dataset_name][1])
        dataset_ = task.get_dataset()
        X, y, categorical_mask, _ = dataset_.get_data(target=dataset_.default_target_attribute)
        hyperparameter_constraints_str = ""
        for key, value in hyperparameter_constraints.items():
            hyperparameter_constraints_str += f"The range of values of '{key}' is: {value}\n"

        prompt = EVOLVE_PROMPT_CONTEXT.format(
            dataset_name=dataset_name,
            classes_number=len(np.unique(y)),
            num_samples=X.shape[0],
            features_number=X.shape[1],
            numeric_features_number=X.shape[1] - len(categorical_mask),
            categorical_features_number=len(categorical_mask),
            target_characteristics="",
            ml_model=MODEL_MAP[model],
            parameters=hyperparameter_constraints_str,
            configuration1=json.dumps(configuration1, ensure_ascii=False),
            configuration2=json.dumps(configuration2, ensure_ascii=False)
        )
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        max_try = 5
        for _ in range(max_try):
            content, message, input_tokens, output_tokens, price = self.llm.predict(messages=messages)
            messages.append(message)
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens

            tag_begin = "<configuration>"
            tag_end = "</configuration>"
            try:
                from utils.utils import load_json
                begin = content.find(tag_begin)
                end = content.find(tag_end)
                if begin != -1 and end != -1:
                    content = content[begin + len(tag_begin):end]
                    child = load_json(content)
                    x_new = []
                    tag = True
                    for parameter_name in parameters_names:
                        if parameter_name in child:
                            x_new.append(child[parameter_name])
                        else:
                            tag = False
                            break
                    if tag:
                        return creator.Individual(x_new), child
                    messages.append({
                        "role": "user",
                        "content": f"Please generate a valid child configuration. missing parameters, the configured parameters need to contain {parameters_names}"
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": "The final parameter configuration must be enclosed in <configuration> and </configuration>."
                    })
            except:
                messages.append({
                    "role": "user",
                    "content": "Please ensure the final parameter configuration must be enclosed in <configuration> and </configuration>."
                })
        raise Exception("Failed to generate a valid child configuration after {} attempts.".format(max_try))

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

    def random_search_hpo_bench(self, search_space):
        x_new = {}
        for parameter_name in search_space:
            categories = search_space[parameter_name]
            random_parameter = random.choice(categories)
            x_new[parameter_name] = random_parameter
        return x_new

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
    evo_prompt = EvoPrompt()
    evo_prompt.run_nl2workflow_experiment()