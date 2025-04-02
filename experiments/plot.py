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
                "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(qwen2_5-14b)": "Textualize(default)",
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
    },
    "基座LLM": {
        "methods": [
            "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(llama3-8b)",
            "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(llama3-70b)",
            "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(qwen2_5-14b)",
            "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(qwen2_5-72b)",
            "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(glm4-9b)",
            "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(deepseek-v3)",
            "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(gpt-4o)",
        ],
        "methods_name_map": {
            "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(llama3-8b)": "Llama3-8B",
            "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(llama3-70b)": "Llama3-70B",
            "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(qwen2_5-14b)": "Qwen2.5-14b",
            "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(qwen2_5-72b)": "Qwen2.5-72b",
            "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(glm4-9b)": "GLM4-9b",
            "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(deepseek-v3)": "DeepSeek-v3",
            "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(gpt-4o)": "GPT4o",
        }
    }
}

LLM_NAME_MAP = {
    "llama3-8b": "Llama3-8B",
    "qwen2_5-14b": "Qwen2.5-14B",
}
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
        max_accuracy_historys = []
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if benchmark == "Public":
                assert len(data) == 410 + 200
            else:
                assert len(data) == 55
            for d in data:
                target = "max_accuracy_history"
                if len(d[target]) < n_trials:
                    d[target] += [d[target][-1]] * (n_trials - len(d[target]))
                # best = 1, worst = 0, 计算normalized regret
                best = 1
                worst = 0
                normalized_regrets = [(item-best) / (worst-best) for item in d[target]]
                max_accuracy_historys.append(normalized_regrets[:n_trials])
            data_np = np.array(max_accuracy_historys)
            if method in item["methods_name_map"]:
                results[item["methods_name_map"][method]] = data_np
            else:
                results[method] = data_np
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
    markers = ["o", "v", "D", "s"]
    if experiment == "对比试验":
        methods = ["Tradition", "LLM"]
        for m in methods:
            lines = []
            fig, axs = plt.subplots(2, 2, figsize=(16, 12))
            llms = ["llama3-8b", "qwen2_5-14b"]
            labels = []
            for i in range(len(llms)):
                for j in range(len(benchmarks)):
                    row = i
                    col = j
                    ax = axs[row, col]
                    x = range(1, n_trials + 1)
                    results = get_plot_data(llm=llms[i], benchmark=benchmarks[j], experiment=experiment,
                                            baseline=m)
                    t = 0
                    for method in results:

                        data = results[method]

                        # 检查所有元素的长度是否一致
                        if all(len(item) == len(data[0]) for item in data):
                            # 如果长度一致，尝试再次转换
                            data_np = np.array(data)
                        else:
                            # 如果长度不一致，处理数据，例如通过截断或填充
                            max_length = max(len(item) for item in data)
                            data_uniform = [item + [item[-1]] * (max_length - len(item)) for item in data]
                            data_np = np.array(data_uniform)

                        column_means = np.mean(data_np, axis=0)
                        if not np.isnan(column_means).all():
                            if t<4:
                                line, = ax.plot(x, column_means[:n_trials], marker=markers[t % 4], color=colors[t])
                            else:
                                line, = ax.plot(x, column_means[:n_trials], marker=markers[t % 4], linestyle='--',color=colors[t])


                            lines.append(line)
                            if method not in labels:
                                labels.append(method)

                        t += 1
                    ax.set_title(f'{benchmarks_names[benchmarks[j]]}, LLM: {LLM_NAME_MAP[llms[i]]}', pad=30, fontsize=23)
                    if i == 1:
                        ax.set_xlabel('Number of trials')
                    if j == 0:
                        ax.set_ylabel('Average nRegret')
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
            fig.legend(labels=labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=4)
            plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust the padding between and around subplots
            save_path = os.path.join(get_project_root(), r"experiments/plots", f"{experiment}_{m}.pdf")
            plt.savefig(save_path)
    elif experiment == "基座LLM":
        lines = []
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        for j in range(len(benchmarks)):
            col = j
            ax = axs[col]
            x = range(1, n_trials + 1)
            results = get_plot_data(benchmark=benchmarks[j], experiment=experiment)
            labels = []
            t = 0
            for method in results:
                data = results[method]

                # 检查所有元素的长度是否一致
                if all(len(item) == len(data[0]) for item in data):
                    # 如果长度一致，尝试再次转换
                    data_np = np.array(data)
                else:
                    # 如果长度不一致，处理数据，例如通过截断或填充
                    max_length = max(len(item) for item in data)
                    data_uniform = [item + [item[-1]] * (max_length - len(item)) for item in data]
                    data_np = np.array(data_uniform)

                column_means = np.mean(data_np, axis=0)
                if not np.isnan(column_means).all():
                    if t < 4:
                        line, = ax.plot(x, column_means[:n_trials], marker=markers[t % 4], color=colors[t])
                    else:
                        line, = ax.plot(x, column_means[:n_trials], marker=markers[t % 4], linestyle='--',
                                        color=colors[t])
                    lines.append(line)
                    if method not in labels:
                        labels.append(method)
                    t += 1
            ax.set_title(f'{benchmarks_names[benchmarks[j]]}')
            if j == 0:
                ax.set_ylabel('Average nRegret')
            ax.set_xlabel('Number of trials')

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        fig.legend(labels=labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=4)
        plt.tight_layout(rect=[0, 0, 1, 0.88])  # Adjust the padding between and around subplots

        save_path = os.path.join(get_project_root(), r"experiments/plots", f"{experiment}.pdf")
        plt.savefig(save_path)
    else:
        lines = []
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        llms = ["llama3-8b", "qwen2_5-14b"]
        labels = []
        for i in range(len(llms)):
            for j in range(len(benchmarks)):
                row = i
                col = j
                ax = axs[row, col]
                x = range(1, n_trials + 1)
                results = get_plot_data(llm=llms[i], benchmark=benchmarks[j], experiment=experiment)
                t = 0
                for method in results:
                    data = results[method]

                    # 检查所有元素的长度是否一致
                    if all(len(item) == len(data[0]) for item in data):
                        # 如果长度一致，尝试再次转换
                        data_np = np.array(data)
                    else:
                        # 如果长度不一致，处理数据，例如通过截断或填充
                        max_length = max(len(item) for item in data)
                        data_uniform = [item + [item[-1]] * (max_length - len(item)) for item in data]
                        data_np = np.array(data_uniform)

                    column_means = np.mean(data_np, axis=0)
                    if not np.isnan(column_means).all():
                        if t < 4:
                            line, = ax.plot(x, column_means[:n_trials], marker=markers[t % 4], color=colors[t])
                        else:
                            line, = ax.plot(x, column_means[:n_trials], marker=markers[t % 4], linestyle='--',
                                            color=colors[t])
                        lines.append(line)
                        if method not in labels:
                            labels.append(method)
                        t += 1
                    ax.set_title(f'{benchmarks_names[benchmarks[j]]}, LLM: {LLM_NAME_MAP[llms[i]]}', pad=30, fontsize=23)
                    if i == 1:
                        ax.set_xlabel('Number of trials')
                    if j == 0:
                        ax.set_ylabel('Average nRegret')
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
        fig.legend(labels=labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=2)
        plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust the padding between and around subplots

        save_path = os.path.join(get_project_root(), r"experiments/plots", f"{experiment}.pdf")
        plt.savefig(save_path)
_plot("基座LLM")