import json
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from utils.utils import get_project_root

n_trials = 15

METHODS_MAP = {
    "llama3-8b": {
        "Few-Shot": [
                    "LLM_Few_Shot_temperature0(llama3-8b)",
                    "LLM_Few_Shot_temperature1(llama3-8b)",
                    "LLM_Few_Shot_temperature2(llama3-8b)",
                    "LLM_Few_Shot_temperature3(llama3-8b)",
                    "LLM_Few_Shot_temperature4(llama3-8b)",
                    "LLM_Few_Shot_temperature5(llama3-8b)",
                    "LLM_Few_Shot_temperature6(llama3-8b)",
                    "LLM_Few_Shot_temperature7(llama3-8b)",
                    "LLM_Few_Shot_temperature8(llama3-8b)",
                    "LLM_Few_Shot_temperature9(llama3-8b)",
                    "LLM_Few_Shot_temperature10(llama3-8b)",
            "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(llama3-8b)",
            "evo_context_t1_type1_starting2_evo_scoreTrue_sort0(llama3-8b)",
            "evo_context_t2_type1_starting2_evo_scoreTrue_sort0(llama3-8b)",
            "evo_context_t3_type1_starting2_evo_scoreTrue_sort0(llama3-8b)",
            "evo_context_t4_type1_starting2_evo_scoreTrue_sort0(llama3-8b)",
            "evo_context_t5_type1_starting2_evo_scoreTrue_sort0(llama3-8b)",
            "evo_context_t6_type1_starting2_evo_scoreTrue_sort0(llama3-8b)",
            "evo_context_t7_type1_starting2_evo_scoreTrue_sort0(llama3-8b)",
            "evo_context_t8_type1_starting2_evo_scoreTrue_sort0(llama3-8b)",
            "evo_context_t9_type1_starting2_evo_scoreTrue_sort0(llama3-8b)",
            "evo_context_t10_type1_starting2_evo_scoreTrue_sort0(llama3-8b)",
        ],
    },
    "qwen2_5-14b": {
        "Few-Shot": [
                    "LLM_Few_Shot_temperature0(qwen2_5-14b)",
                    "LLM_Few_Shot_temperature1(qwen2_5-14b)",
                    "LLM_Few_Shot_temperature2(qwen2_5-14b)",
                    "LLM_Few_Shot_temperature3(qwen2_5-14b)",
                    "LLM_Few_Shot_temperature4(qwen2_5-14b)",
                    "LLM_Few_Shot_temperature5(qwen2_5-14b)",
                    "LLM_Few_Shot_temperature6(qwen2_5-14b)",
                    "LLM_Few_Shot_temperature7(qwen2_5-14b)",
                    "LLM_Few_Shot_temperature8(qwen2_5-14b)",
                    "LLM_Few_Shot_temperature9(qwen2_5-14b)",
                    "LLM_Few_Shot_temperature10(qwen2_5-14b)",
            "evo_context_t0_type1_starting2_evo_scoreTrue_sort0(qwen2_5-14b)",
            "evo_context_t1_type1_starting2_evo_scoreTrue_sort0(qwen2_5-14b)",
            "evo_context_t2_type1_starting2_evo_scoreTrue_sort0(qwen2_5-14b)",
            "evo_context_t3_type1_starting2_evo_scoreTrue_sort0(qwen2_5-14b)",
            "evo_context_t4_type1_starting2_evo_scoreTrue_sort0(qwen2_5-14b)",
            "evo_context_t5_type1_starting2_evo_scoreTrue_sort0(qwen2_5-14b)",
            "evo_context_t6_type1_starting2_evo_scoreTrue_sort0(qwen2_5-14b)",
            "evo_context_t7_type1_starting2_evo_scoreTrue_sort0(qwen2_5-14b)",
            "evo_context_t8_type1_starting2_evo_scoreTrue_sort0(qwen2_5-14b)",
            "evo_context_t9_type1_starting2_evo_scoreTrue_sort0(qwen2_5-14b)",
            "evo_context_t10_type1_starting2_evo_scoreTrue_sort0(qwen2_5-14b)",
        ],
    }
}

def get_plot_data(benchmark, llm, experiment):
    item = METHODS_MAP[llm][experiment]
    results = []
    for method in item:
        data_path = os.path.join(get_project_root(), r"experiments", f"{benchmark}/{method}.json")
        max_accuracy_historys = []
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if benchmark == "Public":
                print(len(data))
                print(f"{benchmark}/{method}")
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
            results.append(data_np)
        else:
            print(f"{data_path} not exist")
    return results

LLM_NAME_MAP = {
    "llama3-8b": "Llama3-8B",
    "qwen2_5-14b": "Qwen2.5-14B",
}
benchmarks_names = {
    "Public": "HPOB+HPOBench",
    "Private": "Uniplore-HPO"
}

def _plot(experiment):
    benchmarks = ["Public", "Private"]
    lines = []
    plt.rcParams.update({'font.size': 24})
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    llms = ["llama3-8b", "qwen2_5-14b"]
    num_colors = 11
    colors = ['#33B4DC'] * num_colors + ['#FFCE65'] * num_colors
    for i in range(len(llms)):
        for j in range(len(benchmarks)):
            row = i
            col = j
            ax = axs[row, col]
            x = range(1, n_trials + 1)
            results = get_plot_data(llm=llms[i], benchmark=benchmarks[j], experiment=experiment)
            for k in range(len(results)):
                data = results[k]
                if all(len(item) == len(data[0]) for item in data):
                    data_np = np.array(data)
                else:
                    max_length = max(len(item) for item in data)
                    data_uniform = [item + [item[-1]] * (max_length - len(item)) for item in data]
                    data_np = np.array(data_uniform)

                column_means = np.mean(data_np, axis=0)
                if not np.isnan(column_means).all():
                    line, = ax.plot(x, column_means[:n_trials], marker='o', color=colors[k])
                    lines.append(line)
            ax.set_title(f'{benchmarks_names[benchmarks[j]]}, LLM: {LLM_NAME_MAP[llms[i]]}', pad=30, fontsize=23)
            legend_labels = [experiment, 'EvoContext']
            legend_lines = [lines[0], lines[-1]]
            if i == 1:
                ax.set_xlabel('Number of trials')
            if j == 0:
                ax.set_ylabel('Average nRegret')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
    fig.legend(legend_lines, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=4)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    save_path = os.path.join(get_project_root(), r"experiments/plots", f"LLM温度参数的影响.pdf")
    plt.savefig(save_path)

_plot("Few-Shot")