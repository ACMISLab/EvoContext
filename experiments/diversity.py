import json
import os
import numpy as np
import matplotlib

matplotlib.use('agg')  # 使用Agg后端
import matplotlib.pyplot as plt

from utils.utils import get_project_root

n_trials = 15

METHODS_MAP = {
    "llama3-8b": {
        "对比试验": {
            "Tradition": {
                "methods": [
                    "evo_context_t0_type1_starting0_evo_scoreTrue_sort0(llama3-8b)",
                    "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(llama3-8b)",
                    "FLAML", "NNI_TPE", "NNI_Evolution", "Random", "DEAP", "SMAC3",
                ],
                "methods_name_map": {
                    "evo_context_t0_type1_starting0_evo_scoreTrue_sort0(llama3-8b)": "EvoContext(Coldstarting)",
                    "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(llama3-8b)": "EvoContext(Warmstarting)",
                    "NNI_TPE": "NNI(TPE)",
                    "NNI_Evolution": "NNI(Evolution)",
                }
            },
            "LLM": {
                "methods": [
                    "evo_context_t0_type1_starting0_evo_scoreTrue_sort0(llama3-8b)",
                    "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(llama3-8b)",
                    "LLM_Zero_Shot_temperature1(llama3-8b)", "LLM_Few_Shot_temperature1(llama3-8b)",
                    "LLAMBO(llama3-8b)", "EvoPrompt(llama3-8b)", "MLCopilot(llama3-8b)",
                ],
                "methods_name_map": {
                    "evo_context_t0_type1_starting0_evo_scoreTrue_sort0(llama3-8b)": "EvoContext(Coldstarting)",
                    "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(llama3-8b)": "EvoContext(Warmstarting)",
                    "LLAMBO(llama3-8b)": "LLAMBO",
                    "EvoPrompt(llama3-8b)": "EvoPrompt",
                    "MLCopilot(llama3-8b)": "MLCopilot",
                    "LLM_Zero_Shot_temperature1(llama3-8b)": "Zero Shot",
                    "LLM_Few_Shot_temperature1(llama3-8b)": "Few Shot",
                }
        }
        },
        "模块消融": {
            "methods": [
                "llm_optimization_best(llama3-8b)",
                    "DEAP",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(llama3-8b)",
            ],
            "methods_name_map": {
                "llm_optimization_best(llama3-8b)": "w/o evolution",
                 "DEAP": "w/o LLM",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(llama3-8b)": "EvoContext",
            }
        },
        "示例数据文本化": {
            "methods": [
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0wotextualize(llama3-8b)",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(llama3-8b)",
            ],
            "methods_name_map": {
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0wotextualize(llama3-8b)": "w/o Textualize",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(llama3-8b)": "Textualize(default)",
            }
        },
        "迭代顺序": {
            "methods": [
                "evo_context_t0_type1_starting1_evo_scoreTrue_sort0(llama3-8b)",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(llama3-8b)",
            ],
            "methods_name_map": {
                "evo_context_t0_type1_starting1_evo_scoreTrue_sort0(llama3-8b)": "LLM->Evolution",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(llama3-8b)": "Evolution->LLM(default)",
            }
        },
        "变异率和交叉率": {
            "methods": [
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0_mutation2cross2(llama3-8b)",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0_mutation2cross8(llama3-8b)",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0_mutation8cross8(llama3-8b)",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(llama3-8b)",
            ],
            "methods_name_map": {
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(llama3-8b)": "mutation(0.8) crossover(0.2) Default",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0_mutation2cross2(llama3-8b)": "mutation(0.2) crossover(0.2)",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0_mutation2cross8(llama3-8b)": "mutation(0.2) crossover(0.8)",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0_mutation8cross8(llama3-8b)": "mutation(0.8) crossover(0.8)",
            }
        },
        "参考的示例": {
            "methods": [
                "evo_context_t0_type0_starting2_evo_scoreTrue_sort0(llama3-8b)",
                "evo_context_t0_type2_starting2_evo_scoreTrue_sort0(llama3-8b)",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(llama3-8b)",
            ],
            "methods_name_map": {
                "evo_context_t0_type0_starting2_evo_scoreTrue_sort0(llama3-8b)": "ALL",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(llama3-8b)": "Top5(default)",
                "evo_context_t0_type2_starting2_evo_scoreTrue_sort0(llama3-8b)": "Evolution",
            }
        },
        "提示词中是否给分数": {
            "methods": [
                "evo_context_t0_type1_starting2_evo_scoreFalse_sort0(llama3-8b)",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(llama3-8b)",
            ],
            "methods_name_map": {
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(llama3-8b)": "score(default)",
                "evo_context_t0_type1_starting2_evo_scoreFalse_sort0(llama3-8b)": "w/o score",
            }
        },
        "提示词中示例的排序方式": {
            "methods": [
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort1(llama3-8b)",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort2(llama3-8b)",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(llama3-8b)",
            ],
            "methods_name_map": {
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(llama3-8b)": "Descend(default)",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort1(llama3-8b)": "Ascend",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort2(llama3-8b)": "Random",
            }
        },
    },
    "qwen2_5-14b": {
        "对比试验": {
            "Tradition": {
                "methods": [
                    "evo_context_t0_type1_starting0_evo_scoreTrue_sort0(qwen2_5-14b)",
                    "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(qwen2_5-14b)",
                    "FLAML", "NNI_TPE", "NNI_Evolution", "Random", "DEAP", "SMAC3",
                ],
                "methods_name_map": {
                    "evo_context_t0_type1_starting0_evo_scoreTrue_sort0(qwen2_5-14b)": "EvoContext(Coldstarting)",
                    "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(qwen2_5-14b)": "EvoContext(Warmstarting)",
                    "NNI_TPE": "NNI(TPE)",
                    "NNI_Evolution": "NNI(Evolution)",
                }
            },
            "LLM": {
                "methods": [
                    "evo_context_t0_type1_starting0_evo_scoreTrue_sort0(qwen2_5-14b)",
                    "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(qwen2_5-14b)",
                    "LLM_Zero_Shot_temperature1(qwen2_5-14b)", "LLM_Few_Shot_temperature1(qwen2_5-14b)",
                    "LLAMBO(qwen2_5-14b)",
                    "EvoPrompt(qwen2_5-14b)",
                    "MLCopilot(qwen2_5-14b)",
                ],
                "methods_name_map": {
                    "evo_context_t0_type1_starting0_evo_scoreTrue_sort0(qwen2_5-14b)": "EvoContext(Coldstarting)",
                    "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(qwen2_5-14b)": "EvoContext(Warmstarting)",
                    "LLAMBO(qwen2_5-14b)": "LLAMBO",
                    "EvoPrompt(qwen2_5-14b)": "EvoPrompt",
                    "MLCopilot(qwen2_5-14b)": "MLCopilot",
                    "LLM_Zero_Shot_temperature1(qwen2_5-14b)": "Zero Shot",
                    "LLM_Few_Shot_temperature1(qwen2_5-14b)": "Few Shot",
                }
            }
        },
        "模块消融": {
            "methods": [
                "llm_optimization_best(qwen2_5-14b)",
                    "DEAP",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(qwen2_5-14b)",
            ],
            "methods_name_map": {
                "llm_optimization_best(qwen2_5-14b)": "w/o evolution",
                 "DEAP": "w/o LLM",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(qwen2_5-14b)": "EvoContext",
            }
        },
        "示例数据文本化": {
            "methods": [
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0wotextualize(qwen2_5-14b)",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(qwen2_5-14b)",
            ],
            "methods_name_map": {
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0wotextualize(qwen2_5-14b)": "w/o Textualize",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(qwen2_5-14b)": "EvoContext",
            }
        },
        "迭代顺序": {
            "methods": [
                "evo_context_t0_type1_starting1_evo_scoreTrue_sort0(qwen2_5-14b)",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(qwen2_5-14b)",
            ],
            "methods_name_map": {
                "evo_context_t0_type1_starting1_evo_scoreTrue_sort0(qwen2_5-14b)": "LLM->Evolution",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(qwen2_5-14b)": "Evolution->LLM(default)",
            }
        },
        "变异率和交叉率": {
            "methods": [
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0_mutation2cross2(qwen2_5-14b)",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0_mutation2cross8(qwen2_5-14b)",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0_mutation8cross8(qwen2_5-14b)",
"evo_context_t0_type1_starting2_evo_scoreTrue_sort0(qwen2_5-14b)",
            ],
            "methods_name_map": {
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(qwen2_5-14b)": "mutation(0.8) crossover(0.2) Default",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0_mutation2cross2(qwen2_5-14b)": "mutation(0.2) crossover(0.2)",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0_mutation2cross8(qwen2_5-14b)": "mutation(0.2) crossover(0.8)",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0_mutation8cross8(qwen2_5-14b)": "mutation(0.8) crossover(0.8)",
            }
        },
        "参考的示例": {
            "methods": [
                "evo_context_t0_type0_starting2_evo_scoreTrue_sort0(qwen2_5-14b)",
                "evo_context_t0_type2_starting2_evo_scoreTrue_sort0(qwen2_5-14b)",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(qwen2_5-14b)",
            ],
            "methods_name_map": {
                "evo_context_t0_type0_starting2_evo_scoreTrue_sort0(qwen2_5-14b)": "ALL",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(qwen2_5-14b)": "Top5(default)",
                "evo_context_t0_type2_starting2_evo_scoreTrue_sort0(qwen2_5-14b)": "Evolution",
            }
        },
        "提示词中是否给分数": {
            "methods": [
                "evo_context_t0_type1_starting2_evo_scoreFalse_sort0(qwen2_5-14b)",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(qwen2_5-14b)",
            ],
            "methods_name_map": {
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(qwen2_5-14b)": "score(default)",
                "evo_context_t0_type1_starting2_evo_scoreFalse_sort0(qwen2_5-14b)": "w/o score",
            }
        },
        "提示词中示例的排序方式": {
            "methods": [
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort1(qwen2_5-14b)",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort2(qwen2_5-14b)",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(qwen2_5-14b)",
            ],
            "methods_name_map": {
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(qwen2_5-14b)": "Descend(default)",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort1(qwen2_5-14b)": "Ascend",
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort2(qwen2_5-14b)": "Random",
            }
        },
    }
}

LLM_NAME_MAP = {
    "llama3-8b": "Llama3-8B",
    "qwen2_5-14b": "Qwen2.5-14B",
}

def get_serch_space_info(model):
    results = {
        "type": [],
        "value": []
    }
    if model in ["Logistic Regression", "SVM", "XGBoost", "Random Forest", "KNN", "LightGBM", "AdaBoost"]:
        data_path = os.path.join(get_project_root(), r"data/benchmarks/nl2workflow/search_space.json")
        with open(data_path, 'r', encoding='utf-8') as f:
            search_spaces = json.load(f)
        for search_space in search_spaces:
            if search_space["algorithm"] == model:
                for key, value in search_space["search_space"].items():
                    if value["_type"] == "choice":
                        results["type"].append("categorical")
                    else:
                        results["type"].append("continuous")
                    results["value"].append(value["_value"])
    elif model in ["lr", "svm", "rf", "xgb", "nn"]:
        data_path = os.path.join(get_project_root(), r"data/benchmarks/hpo_bench/hpo_bench.json")
        with open(data_path, 'r', encoding='utf-8') as f:
            search_spaces = json.load(f)
        for item in search_spaces:
            if item == model:
                for key, value in search_spaces[item].items():
                    results["type"].append("categorical")
                    results["value"].append(value)
    elif model in ["4796", "5527", "5636", "5859", "5860", "5891", "5906", "5965", "5970", "5971", "6766", "6767", "6794", "7607", "7609", "5889"]:
        from data.benchmarks.hpob.search_spaces_info import SEARCH_SPACE_INFO
        parameters_name = SEARCH_SPACE_INFO[model]["parameters_name"]
        for parameter in parameters_name:
            if SEARCH_SPACE_INFO[model][parameter]["type"] == "categorical":
                results["type"].append("categorical")
                results["value"].append(SEARCH_SPACE_INFO[model][parameter]["categories"])
            else:
                results["type"].append("continuous")
                results["value"].append([SEARCH_SPACE_INFO[model][parameter]["low"], SEARCH_SPACE_INFO[model][parameter]["high"]])
    else:
        raise ValueError(f"Unknown model: {model}")
    return results

def diversity(configs, scores, model):
    serch_space_info = get_serch_space_info(model)
    diversity = 0
    N = len(configs)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            distance = 0
            for k in range(len(serch_space_info["type"])):
                if serch_space_info["type"][k] == "categorical":
                    distance += 1 - (configs[i][k] == configs[j][k])
                else:
                    try:
                        distance += abs((int(configs[i][k]) - int(configs[j][k])) / (
                                    serch_space_info["value"][k][1] - serch_space_info["value"][k][0]))
                    except:
                        distance += abs((float(configs[i][k]) - float(configs[j][k])) / (
                                    serch_space_info["value"][k][1] - serch_space_info["value"][k][0]))
            diversity += (distance/len(serch_space_info["type"]) + abs(scores[i] - scores[j]))
    return diversity / (2 * N * N)

def get_plot_data(benchmark, experiment, llm=None,  baseline=None):
    if llm is None:
        item = METHODS_MAP[experiment]
    else:
        if baseline is None:
            item = METHODS_MAP[llm][experiment]
        else:
            item = METHODS_MAP[llm][experiment][baseline]
    results = {}
    for method in item["methods"]:
        data_path = os.path.join(get_project_root(), r"experiments", f"{benchmark}/{method}.json")
        diversitys = []
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if benchmark == "Public":
                assert len(data) == 410 + 200
            else:
                assert len(data) == 55
            for d in data:
                if len(d["x_observed"]) < n_trials:
                    d["x_observed"] += [d["x_observed"][-1]] * (n_trials - len(d["x_observed"]))
                if len(d["y_observed"]) < n_trials:
                    d["y_observed"] += [d["y_observed"][-1]] * (n_trials - len(d["y_observed"]))
                diversitys.append(diversity(d["x_observed"][:n_trials], d["y_observed"][:n_trials], d["search_space"]))
            data_np = np.array(diversitys)
            if method in item["methods_name_map"]:
                results[item["methods_name_map"][method]] = round(data_np.mean(), 3)
            else:
                results[method] = round(data_np.mean(), 3)
        else:
            print(f"{data_path} not exist")
    return results

benchmarks_names = {
    "Public": "HPOB+HPOBench",
    "Private": "Uniplore-HPO"
}
def _plot(experiment):
    """
    :param llm: llam3-8b, qwen2.5-14b
    :param experiment: 对比试验，模块消融，示例数据文本化
    :return:
    """
    benchmarks = ["Public", "Private"]
    # 设置全局的字体大小
    plt.rcParams.update({'font.size': 24})
    colors = ["#33B4DC", "#7ECEB8", "#F2CE7E", "#D19391", "#E5BBD2", "#A2A2A2", "#EE895F", "#84ADD0"]
    if experiment == "对比试验":
        methods = ["Tradition", "LLM"]
        for m in methods:
            fig, axs = plt.subplots(2, 2, figsize=(20, 10))
            llms = ["llama3-8b", "qwen2_5-14b"]
            labels = []
            labels_colors = []
            for i in range(len(llms)):
                for j in range(len(benchmarks)):
                    row = i
                    col = j
                    ax = axs[row, col]
                    results = get_plot_data(llm=llms[i], benchmark=benchmarks[j], experiment=experiment, baseline=m)
                    methods_ = [method for method in results]
                    values = [results[method] for method in results]
                    bars = ax.bar(methods_, values, color=colors[:len(methods_)])
                    if labels == []:
                        labels = methods_
                        labels_colors = bars
                    # 在柱子上显示值
                    for index, value in enumerate(values):
                        ax.text(index, value, str(value), ha='center', va='bottom')
                    ax.set_title(f'{benchmarks_names[benchmarks[j]]}, LLM: {LLM_NAME_MAP[llms[i]]}', pad=30, fontsize=23)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    # 取消横坐标上的方法值
                    ax.set_xticks([])

            fig.legend(labels_colors, labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=4)
            fig.supylabel('Average Diversity')  # 设置总图的纵坐标标签
            plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust the padding between and around subplots
            save_path = os.path.join(get_project_root(), r"experiments/plots/diversity", f"{experiment}_{m}.pdf")
            plt.savefig(save_path)
    else:
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        llms = ["llama3-8b", "qwen2_5-14b"]
        labels = []
        labels_colors = []
        for i in range(len(llms)):
            for j in range(len(benchmarks)):
                row = i
                col = j
                ax = axs[row, col]
                results = get_plot_data(llm=llms[i], benchmark=benchmarks[j], experiment=experiment)
                methods = list(results.keys())
                values = list(results.values())

                # 为每个方法分配不同的颜色
                bars = ax.bar(methods, values, color=colors[:len(methods)])
                if labels == []:
                    labels = methods
                    labels_colors = bars

                # 在柱子上显示值
                for index, value in enumerate(values):
                    ax.text(index, value, str(value), ha='center', va='bottom')
                ax.set_title(f'{benchmarks_names[benchmarks[j]]}, LLM: {LLM_NAME_MAP[llms[i]]}', pad=30, fontsize=23)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                # 取消横坐标上的方法值
                ax.set_xticks([])

        fig.legend(labels_colors, labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=4)
        fig.supylabel('Average Diversity')  # 设置总图的纵坐标标签
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the padding between and around subplots
        save_path = os.path.join(get_project_root(), r"experiments/plots/diversity", f"{experiment}.pdf")
        plt.savefig(save_path)
_plot("对比试验")