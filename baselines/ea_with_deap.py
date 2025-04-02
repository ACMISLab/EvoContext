from deap import base, creator, tools
import numpy as np
import random
random.seed(42)
import xgboost as xgb
from data.benchmarks.hpob.search_spaces_info import SEARCH_SPACE_INFO
from baselines.baseline import Baseline
import os
import json
from utils.utils import get_project_root

class GeneticAlgorithm(Baseline):

    def __init__(self, population_size=5, mutation_rate=0.8, cross_rate=0.2):
        super().__init__()
        self.population_size = population_size # 种群的大小
        self.mutation_rate = mutation_rate # 变异率
        self.cross_rate = cross_rate   # 交叉概率
        self.name = f"DEAP"  # 算法名称
        self.toolbox = self.initialize_toolbox()

    def run_method(self, seed_id, search_space_id, dataset_id):
        # 加载评估代理模型
        surrogate_name = 'surrogate-' + search_space_id + '-' + dataset_id
        bst_surrogate = xgb.Booster()
        bst_surrogate.load_model(self.surrogates_dir + surrogate_name + '.json')
        self.bst_surrogate = bst_surrogate

        search_space = SEARCH_SPACE_INFO[search_space_id]
        self.search_space = search_space
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
            child1, child2 = self.toolbox.clone(ind1), self.toolbox.clone(ind2)
            self.toolbox.mate(child1, child2)
            self.toolbox.mutate(child1)
            new_x = child1
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
        from data.benchmarks.hpo_bench.tabular_benchmarks import DATASET_MAP, HPOExpRunner
        dataset = DATASET_MAP[dataset_name][0]
        benchmark = HPOExpRunner(model, dataset, seed_id)
        with open(os.path.join(get_project_root(), r"data/benchmarks/hpo_bench/hpo_bench.json"), 'r') as f:
            search_space = json.load(f)[model]
        parameters_name = [key for key in search_space.keys()]
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
            child1, child2 = self.toolbox.clone(ind1), self.toolbox.clone(ind2)
            self.toolbox.mate(child1, child2)
            child1 = self.mutate(child1, search_space)
            new_x = child1
            new_x_pro = {parameters_name[i]: new_x[i]  for i in range(len(new_x))}
            new_x.fitness.values = benchmark.evaluate_point(new_x_pro),

            y_observed.append(new_x.fitness.values[0])
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

        for i in range(self.population_size):
            new_x = self.random_search_nl2workflow(search_space)
            individual = creator.Individual(new_x)
            individual.fitness.values = evaluate(new_x, model_name, dataset_name),
            y_observed.append(individual.fitness.values[0])
            x_observed.append(individual)
            # 记录当前最优的分数
            max_accuracy_history.append(max(y_observed))

        for i in range(self.n_trials):
            if i < self.population_size:
                continue
            ind1, ind2 = self.toolbox.select(x_observed)
            child1, child2 = self.toolbox.clone(ind1), self.toolbox.clone(ind2)
            self.toolbox.mate(child1, child2)
            child1 = self.mutate_nl2workflow(child1, search_space)
            new_x = child1
            new_x.fitness.values = evaluate(new_x, model_name, dataset_name),

            # 记录观察到的超参数和分数
            y_observed.append(new_x.fitness.values[0])
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

    def initialize_toolbox(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        toolbox.register("mate", tools.cxUniform, indpb=self.cross_rate)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=self.mutation_rate)
        toolbox.register("select", tools.selRoulette, k=2, fit_attr='fitness')

        return toolbox

    def mutate(self, individual, search_space):
        x_new = []
        search_space_new = []
        for item in search_space:
            search_space_new.append(search_space[item])

        for i in range(len(individual)):
            categories = search_space_new[i]
            if np.random.rand() < self.mutation_rate:
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
            else:
                x_new.append(individual[i])
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

    def evaluate(self, new_x):
        new_x_arry = np.array(new_x)
        x_q = xgb.DMatrix(new_x_arry.reshape(-1, self.dim))
        new_y_arry = self.bst_surrogate.predict(x_q)
        new_y = float(new_y_arry[0])
        return new_y,

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

    # 测试代码
    ga = GeneticAlgorithm()
    # ga.run_nl2workflow_experiment()
    ga.run_hpo_bench_experiment()
    # ga.run_hpob_experiment()
