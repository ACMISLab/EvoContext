import xgboost as xgb
import os
from utils.utils import get_project_root
from data.benchmarks.nl2workflow.data_processing import MODELS_MAP, convert_configs
import numpy as np

def evaluate(config, model, dataset):
    config = convert_configs(model_name=model, config=config)
    new_x_arry = np.array(config)
    x_q = xgb.DMatrix(new_x_arry.reshape(-1, len(config)))
    bst_surrogate = xgb.Booster()
    surrogates_dir = os.path.join(get_project_root(), r"data/benchmarks/nl2workflow/saved-surrogates")
    model_path = os.path.join(surrogates_dir, f'surrogate-{MODELS_MAP[model]}-{dataset}.json')
    # 判断模型地址是否存在
    if not os.path.exists(model_path):
        raise ValueError("模型文件不存在")
    bst_surrogate.load_model(model_path)
    new_y_arry = bst_surrogate.predict(x_q)
    new_y = float(new_y_arry[0])
    return new_y