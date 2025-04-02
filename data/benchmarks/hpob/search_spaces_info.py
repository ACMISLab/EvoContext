"""
Your goal is optimize the hyperparameters of {ml_model}.
The {ml_model} has hyperparameters: {parameters_name}.
Here are the details of each hyperparameter:
- {parameter1}
- {parameter2}
- {parameter3}

Now, please optimize the hyperparameters of {ml_model} to maximize the model performance on the {dataset}.
"""
import json

"""
search_space_id: {
    "ml_model": "machine learning model name",  # e.g., "XGBoost"
    "parameters_name": ["parameter1", "parameter2", "parameter3"],
    "parameter1": {
        "description": "超参数1的描述",
        "type": "float",
        "low": 0.0001,
        "high": 0.1
    },
    "parameter2": {
        "description": "超参数2的描述",
        "type": "int",
        "low": 1,
        "high": 10
    },
    "parameter3": {
        "description": "超参数3的描述",
        "type": "categorical",
        "categories": ["category1", "category2", "category3"]
    }
}
"""

SEARCH_SPACE_INFO = {
    "4796": {
        "model_name": "rpart.preproc",
        "parameters_name": [
            "cp",
            "minbucket",
            "minsplit"
        ],
        "quantile_info": {
            "cp": [
                0.06542494148015976,
                0.24426785111427307,
                0.4488784670829773,
                0.6678040623664856,
                0.8920339345932007
            ],
            "minbucket": [
                0.0,
                0.3333231806755066,
                0.4999907910823822,
                0.6666598916053772,
                1.0
            ],
            "minsplit": [
                0.0,
                0.3333282470703125,
                0.4999954104423523,
                0.6666632890701294,
                1.0
            ]
        },
        "cp": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "剪枝复杂度参数，控制模型的复杂度"
        },
        "minbucket": {
            "type": "categorical",
            "categories": [0.16665859, 0.6666599, 0.33332319, 0.4999908, 0.83332976, 0.0, 1.0],
            "description": "每个叶子节点的最小样本数"
        },
        "minsplit": {
            "type": "categorical",
            "categories": [0.33332826, 0.16666263, 0.66666329, 0.0, 1.0, 0.83333155, 0.4999954],
            "description": "划分节点所需的最小样本数"
        }
    },
    "5527": {
        "model_name": "svm",
        "parameters_name": [
            "cost",
            "gamma",
            "degree"
        ],
        "quantile_info": {
            "cost": [
                0.0933031290769577,
                0.28888046741485596,
                0.4859902560710907,
                0.6847033500671387,
                0.8893853425979614
            ],
            "gamma": [
                0.0,
                0.0,
                0.0,
                0.20384544134140015,
                0.7276173830032349
            ],
            "degree": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.800000011920929
            ]
        },
        "cost": {
            "type": "float64",
            "low": 1.0724331833633832e-06,
            "high": 0.9999989042752158,
            "description": "惩罚参数，控制对错误分类的惩罚力度"
        },
        "gamma": {
            "type": "float64",
            "low": 0.0,
            "high": 0.9999844117884508,
            "description": "核函数的参数，影响决策边界的形状"
        },
        "degree": {
            "type": "categorical",
            "categories": [
                0.0,
                0.4,
                0.8,
                1.0
            ],
            "description": "多项式核的度数"
        }
    },
    "5636": {
        "model_name": "rpart",
        "parameters_name": [
            "cp",
            "maxdepth",
            "minbucket",
            "minsplit"
        ],
        "quantile_info": {
            "cp": [
                0.7498170137405396,
                0.8690887689590454,
                0.9245840907096863,
                0.9612493515014648,
                0.9885063171386719
            ],
            "maxdepth": [
                0.1034482792019844,
                0.3103448152542114,
                0.517241358757019,
                0.7241379022598267,
                0.931034505367279
            ],
            "minbucket": [
                0.10169491171836853,
                0.2881355881690979,
                0.49152541160583496,
                0.7118644118309021,
                0.8983050584793091
            ],
            "minsplit": [
                0.10000000149011612,
                0.28333333134651184,
                0.5166666507720947,
                0.699999988079071,
                0.9166666865348816
            ]
        },
        "cp": {
            "type": "float64",
            "low": 0.0017861445988133,
            "high": 1.0,
            "description": "剪枝复杂度参数，控制模型的复杂度"
        },
        "maxdepth": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "树的最大深度"
        },
        "minbucket": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "每个叶子节点的最小样本数"
        },
        "minsplit": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "划分节点所需的最小样本数"
        }
    },
    "5859": {
        "model_name": "rpart",
        "parameters_name": [
            "cp",
            "maxdepth",
            "minbucket",
            "minsplit"
        ],
        "quantile_info": {
            "cp": [
                0.7487955689430237,
                0.8686177134513855,
                0.9245337843894958,
                0.9616148471832275,
                0.9886093735694885
            ],
            "maxdepth": [
                0.06896551698446274,
                0.27586206793785095,
                0.48275861144065857,
                0.6896551847457886,
                0.931034505367279
            ],
            "minbucket": [
                0.10169491171836853,
                0.2881355881690979,
                0.49152541160583496,
                0.694915235042572,
                0.8983050584793091
            ],
            "minsplit": [
                0.0833333358168602,
                0.28333333134651184,
                0.5166666507720947,
                0.699999988079071,
                0.8999999761581421
            ]
        },
        "cp": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "剪枝复杂度参数，控制模型的复杂度"
        },
        "maxdepth": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "树的最大深度"
        },
        "minbucket": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "每个叶子节点的最小样本数"
        },
        "minsplit": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "划分节点所需的最小样本数"
        }
    },
    "5860": {
        "model_name": "glmnet",
        "parameters_name": [
            "alpha",
            "lambda"
        ],
        "quantile_info": {
            "alpha": [
                0.6830036640167236,
                0.8396069407463074,
                0.907231867313385,
                0.9524593949317932,
                0.9840936660766602
            ],
            "lambda": [
                0.10185094177722931,
                0.2959875464439392,
                0.49748632311820984,
                0.6897231340408325,
                0.8943955898284912
            ]
        },
        "alpha": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "控制L1和L2正则化的比例"
        },
        "lambda": {
            "type": "float64",
            "low": 0.0001217023571036,
            "high": 1.0,
            "description": "正则化参数，控制模型的复杂度"
        }
    },
    "5891": {
        "model_name": "svm",
        "parameters_name": [
            "cost",
            "gamma",
            "degree"
        ],
        "quantile_info": {
            "cost": [
                2.764857754300465e-06,
                5.414249244495295e-05,
                0.0008402120438404381,
                0.01334882527589798,
                0.23920153081417084
            ],
            "gamma": [
                0.0,
                0.0,
                0.0,
                2.961662403322407e-06,
                0.012318101711571217
            ],
            "degree": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.800000011920929
            ]
        },
        "cost": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "惩罚参数，控制对错误分类的惩罚力度"
        },
        "gamma": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "核函数的参数，影响决策边界的形状"
        },
        "degree": {
            "type": "categorical",
            "categories": [
                0.0,
                0.8,
                1.0,
                0.4
            ],
            "description": "多项式核的度数"
        }
    },
    "5906": {
        "model_name": "xgboost",
        "parameters_name": [
            "alpha",
            "colsample_bylevel",
            "colsample_bytree",
            "eta",
            "lambda",
            "max_depth",
            "min_child_weight",
            "nrounds",
            "subsample"
        ],
        "quantile_info": {
            "alpha": [
                0.09515941143035889,
                0.299391508102417,
                0.5008200407028198,
                0.702559769153595,
                0.8986281752586365
            ],
            "colsample_bylevel": [
                0.0,
                0.0,
                0.0,
                0.38948240876197815,
                0.812872588634491
            ],
            "colsample_bytree": [
                0.0,
                0.0,
                0.0,
                0.907547116279602,
                0.9783622026443481
            ],
            "eta": [
                0.0994350016117096,
                0.2897390127182007,
                0.47953805327415466,
                0.6717483401298523,
                0.8802154660224915
            ],
            "lambda": [
                0.10090772807598114,
                0.2882172465324402,
                0.4963969588279724,
                0.6983296275138855,
                0.902617871761322
            ],
            "max_depth": [
                0.0,
                0.0,
                0.0,
                0.2666666805744171,
                0.800000011920929
            ],
            "min_child_weight": [
                0.0,
                0.0,
                0.0,
                0.7525123357772827,
                0.9213484525680542
            ],
            "nrounds": [
                0.8741395473480225,
                0.9317451119422913,
                0.9606738686561584,
                0.9800571799278259,
                0.9934771060943604
            ],
            "subsample": [
                0.1005484014749527,
                0.3024177849292755,
                0.5121214985847473,
                0.7042391300201416,
                0.9002918004989624
            ]
        },
        "alpha": {
            "type": "float64",
            "low": 2.92e-05,
            "high": 0.999910278,
            "description": "L1正则化参数"
        },
        "colsample_bylevel": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "每一层的特征采样比例"
        },
        "colsample_bytree": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "构建每棵树时使用的特征比例"
        },
        "eta": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "学习率，控制每次迭代的步长"
        },
        "lambda": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "L2正则化参数"
        },
        "max_depth": {
            "type": "categorical",
            "categories": [0.8, 0.0, 0.06666667, 0.73333333, 0.2, 0.66666667, 0.26666667, 0.13333333, 0.53333333, 0.86666667, 0.46666667, 1.0, 0.33333333, 0.93333333, 0.6],
            "description": "树的最大深度"
        },
        "min_child_weight": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "子节点中最小的样本权重和"
        },
        "nrounds": {
            "type": "float64",
            "low": 0.558652872,
            "high": 1.0,
            "description": "迭代次数"
        },
        "subsample": {
            "type": "float64",
            "low": 4.72e-05,
            "high": 0.999830676,
            "description": "用于训练模型的样本比例"
        }
    },
    "5965": {
        "model_name": "ranger",
        "parameters_name": [
            "min.node.size",
            "mtry",
            "num.trees",
            "sample.fraction"
        ],
        "quantile_info": {
            "min.node.size": [
                0.46210187673568726,
                0.5664262175559998,
                0.6640549898147583,
                0.7485977411270142,
                0.8476542234420776
            ],
            "mtry": [
                0.09263505041599274,
                0.1852734386920929,
                0.2936554551124573,
                0.40037819743156433,
                0.6270030736923218
            ],
            "num.trees": [
                0.8515899777412415,
                0.9206660985946655,
                0.9536146521568298,
                0.975911021232605,
                0.9927330017089844
            ],
            "sample.fraction": [
                0.08993007987737656,
                0.27936723828315735,
                0.47541359066963196,
                0.6813692450523376,
                0.8926385045051575
            ]
        },
        "min.node.size": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "叶子节点的最小样本数"
        },
        "mtry": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "每棵树随机选择的特征数量"
        },
        "num.trees": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "森林中树的数量"
        },
        "sample.fraction": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "用于构建树的样本比例"
        }
    },
    "5970": {
        "model_name": "ranger",
        "parameters_name": [
            "alpha",
            "lambda"
        ],
        "quantile_info": {
            "alpha": [
                0.09921348839998245,
                0.3036178648471832,
                0.5029023885726929,
                0.7025471329689026,
                0.9019204378128052
            ],
            "lambda": [
                2.924562977568712e-06,
                5.955695451120846e-05,
                0.0009623399237170815,
                0.01604960486292839,
                0.2584221065044403
            ]
        },
        "alpha": {
            "type": "float64",
            "low": 1.3486429008569235e-05,
            "high": 0.999993216854508,
            "description": "控制L1正则化"
        },
        "lambda": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "控制L2正则化"
        }
    },
    "5971": {
        "model_name": "xgboost",
        "parameters_name": [
            "alpha",
            "colsample_bylevel",
            "colsample_bytree",
            "eta",
            "lambda",
            "max_depth",
            "min_child_weight",
            "nrounds",
            "subsample"
        ],
        "quantile_info": {
            "alpha": [
                0.09481142461299896,
                0.29385891556739807,
                0.49647200107574463,
                0.6990981101989746,
                0.8990970253944397
            ],
            "colsample_bylevel": [
                0.0,
                0.0,
                0.0,
                0.37460485100746155,
                0.789810299873352
            ],
            "colsample_bytree": [
                0.0,
                0.0,
                0.0,
                0.3983893394470215,
                0.8080970644950867
            ],
            "eta": [
                0.0936686098575592,
                0.290618896484375,
                0.49267420172691345,
                0.6937156319618225,
                0.8991912603378296
            ],
            "lambda": [
                0.09659961611032486,
                0.2959820330142975,
                0.49755313992500305,
                0.6990941166877747,
                0.9013055562973022
            ],
            "max_depth": [
                0.0,
                0.0,
                0.0,
                0.2666666805744171,
                0.800000011920929
            ],
            "min_child_weight": [
                0.0,
                0.0,
                0.0,
                0.7558436393737793,
                0.9190552234649658
            ],
            "nrounds": [
                0.10019999742507935,
                0.30000001192092896,
                0.49639999866485596,
                0.6948000192642212,
                0.8981999754905701
            ],
            "subsample": [
                0.09769324213266373,
                0.29670873284339905,
                0.49858415126800537,
                0.6987059712409973,
                0.8988183736801147
            ]
        },
        "alpha": {
            "type": "float64",
            "low": 3.526446474087557e-06,
            "high": 1.0,
            "description": "L1正则化参数"
        },
        "colsample_bylevel": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "每一层的特征采样比例"
        },
        "colsample_bytree": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "构建每棵树时使用的特征比例"
        },
        "eta": {
            "type": "float64",
            "low": 1.0128443936672635e-05,
            "high": 1.0,
            "description": "学习率，控制每次迭代的步长"
        },
        "lambda": {
            "type": "float64",
            "low": 2.441609784760115e-05,
            "high": 1.0,
            "description": "L2正则化参数"
        },
        "max_depth": {
            "type": "categorical",
            "categories": [0.0, 0.46666667, 0.53333333, 1.0, 0.13333333, 0.2, 0.06666667, 0.26666667, 0.33333333, 0.86666667, 0.73333333, 0.66666667, 0.6, 0.93333333, 0.8],
            "description": "树的最大深度"
        },
        "min_child_weight": {
            "type": "float64",
            "low": 0.0,
            "high": 0.9999944553298932,
            "description": "子节点中最小的样本权重和"
        },
        "nrounds": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "迭代次数"
        },
        "subsample": {
            "type": "float64",
            "low": 4.676697646707956e-05,
            "high": 1.0,
            "description": "用于训练模型的样本比例"
        }
    },
    "6766": {
        "model_name": "glmnet",
        "parameters_name": [
            "alpha",
            "lambda"
        ],
        "quantile_info": {
            "alpha": [
                0.1001073494553566,
                0.29937177896499634,
                0.4996643364429474,
                0.6992779970169067,
                0.8997756838798523
            ],
            "lambda": [
                2.890365067287348e-06,
                6.0076035879319534e-05,
                0.000984818208962679,
                0.01589113287627697,
                0.24908830225467682
            ]
        },
        "alpha": {
            "type": "float64",
            "low": 0.0,
            "high": 0.9999992982479988,
            "description": "控制L1和L2正则化的比例"
        },
        "lambda": {
            "type": "float64",
            "low": 3.13363541299707e-12,
            "high": 1.0,
            "description": "正则化参数，控制模型的复杂度"
        }
    },
    "6767": {
        "model_name": "xgboost",
        "parameters_name": [
            "alpha",
            "colsample_bylevel",
            "colsample_bytree",
            "eta",
            "lambda",
            "max_depth",
            "min_child_weight",
            "nrounds",
            "nthread",
            "subsample"
        ],
        "quantile_info": {
            "alpha": [
                2.869840272978763e-06,
                6.20772989350371e-05,
                0.0010229060426354408,
                0.016160352155566216,
                0.2581975758075714
            ],
            "colsample_bylevel": [
                0.0,
                0.0,
                0.0,
                0.3565407693386078,
                0.7874588370323181
            ],
            "colsample_bytree": [
                0.0,
                0.0,
                0.0,
                0.3777301609516144,
                0.8001556396484375
            ],
            "eta": [
                0.0009901922894641757,
                0.006909035611897707,
                0.030453940853476524,
                0.12475746870040894,
                0.5012167096138
            ],
            "lambda": [
                2.870916887331987e-06,
                6.0366957768565044e-05,
                0.000976294744759798,
                0.01562914252281189,
                0.250921368598938
            ],
            "max_depth": [
                0.0,
                0.0,
                0.0,
                0.20000000298023224,
                0.800000011920929
            ],
            "min_child_weight": [
                0.0,
                0.0,
                0.0,
                0.029564645141363144,
                0.31596511602401733
            ],
            "nrounds": [
                0.09160000085830688,
                0.28139999508857727,
                0.4781999886035919,
                0.6827999949455261,
                0.8930000066757202
            ],
            "nthread": [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0
            ],
            "subsample": [
                0.09939602762460709,
                0.2979719340801239,
                0.4973636269569397,
                0.6984626054763794,
                0.8995145559310913
            ]
        },
        "alpha": {
            "type": "float64",
            "low": 6.82e-12,
            "high": 1.0,
            "description": "L1正则化参数"
        },
        "colsample_bylevel": {
            "type": "float64",
            "low": 0.0,
            "high": 0.999988003,
            "description": "每一层的特征采样比例"
        },
        "colsample_bytree": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "构建每棵树时使用的特征比例"
        },
        "eta": {
            "type": "float64",
            "low": 1.45e-08,
            "high": 0.999996288,
            "description": "学习率，控制每次迭代的步长"
        },
        "lambda": {
            "type": "float64",
            "low": 0.0,
            "high": 0.999968404,
            "description": "L2正则化参数"
        },
        "max_depth": {
            "type": "categorical",
            "categories": [0.0, 1.0, 0.53333333, 0.33333333, 0.8, 0.86666667, 0.06666667, 0.73333333, 0.2, 0.93333333, 0.6, 0.46666667, 0.66666667, 0.26666667, 0.13333333],
            "description": "树的最大深度"
        },
        "min_child_weight": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "子节点中最小的样本权重和"
        },
        "nrounds": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "迭代次数"
        },
        "nthread": {
            "type": "categorical",
            "categories": [
                0,
                1
            ],
            "description": "线程数，控制并行计算"
        },
        "subsample": {
            "type": "float64",
            "low": 0.0,
            "high": 0.99999371,
            "description": "用于训练模型的样本比例"
        }
    },
    "6794": {
        "model_name": "ranger",
        "parameters_name": [
            "min.node.size",
            "mtry",
            "num.trees",
            "sample.fraction"
        ],
        "quantile_info": {
            "min.node.size": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.013771518133580685
            ],
            "mtry": [
                0.0006422607693821192,
                0.0019267823081463575,
                0.004495825152844191,
                0.008991650305688381,
                0.023121386766433716
            ],
            "num.trees": [
                0.08900000154972076,
                0.2775000035762787,
                0.47600001096725464,
                0.6804999709129333,
                0.8930000066757202
            ],
            "sample.fraction": [
                0.09363872557878494,
                0.28603243827819824,
                0.48501116037368774,
                0.6883272528648376,
                0.8946738839149475
            ]
        },
        "min.node.size": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "叶子节点的最小样本数"
        },
        "mtry": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "每棵树随机选择的特征数量"
        },
        "num.trees": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "森林中树的数量"
        },
        "sample.fraction": {
            "type": "float64",
            "low": 8.48e-07,
            "high": 1.0,
            "description": "用于构建树的样本比例"
        }
    },
    "7607": {
        "model_name": "ranger",
        "parameters_name": [
            "min.node.size",
            "mtry",
            "num.trees",
            "sample.fraction"
        ],
        "quantile_info": {
            "min.node.size": [
                0.0647146999835968,
                0.2238837480545044,
                0.3689197301864624,
                0.5144040584564209,
                0.6960402131080627
            ],
            "mtry": [
                0.0005636978312395513,
                0.0016910935519263148,
                0.0033821871038526297,
                0.010146561078727245,
                0.08680947124958038
            ],
            "num.trees": [
                0.10249999910593033,
                0.29649999737739563,
                0.4950000047683716,
                0.6980000138282776,
                0.9020000100135803
            ],
            "sample.fraction": [
                0.09809006750583649,
                0.2988605797290802,
                0.5018405914306641,
                0.7029315233230591,
                0.9013432264328003
            ]
        },
        "min.node.size": {
            "type": "float64",
            "low": 0.0,
            "high": 0.9998894939362448,
            "description": "叶子节点的最小样本数"
        },
        "mtry": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "每棵树随机选择的特征数量"
        },
        "num.trees": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "森林中树的数量"
        },
        "sample.fraction": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "用于构建树的样本比例"
        }
    },
    "7609": {
        "model_name": "ranger",
        "parameters_name": [
            "min.node.size",
            "mtry",
            "num.trees",
            "sample.fraction"
        ],
        "quantile_info": {
            "min.node.size": [
                0.06465144455432892,
                0.21477456390857697,
                0.3591289818286896,
                0.5031031966209412,
                0.6673663854598999
            ],
            "mtry": [
                0.0005633803084492683,
                0.0016901408089324832,
                0.004507042467594147,
                0.013521126471459866,
                0.20619718730449677
            ],
            "num.trees": [
                0.10249999910593033,
                0.30649998784065247,
                0.5019999742507935,
                0.6984999775886536,
                0.9010000228881836
            ],
            "sample.fraction": [
                0.10183405131101608,
                0.29806625843048096,
                0.49913233518600464,
                0.6995525360107422,
                0.8986156582832336
            ]
        },
        "min.node.size": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "每个叶子节点的最小样本数"
        },
        "mtry": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "每棵树分裂时随机选择的特征数量"
        },
        "num.trees": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "森林中树的数量"
        },
        "sample.fraction": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "训练每棵树时所用样本的比例"
        }
    },
    "5889": {
        "model_name": "ranger",
        "parameters_name": [
            "mtry",
            "num.trees",
            "sample.fraction"
        ],
        "quantile_info": {
            "mtry": [
                0.09642001241445541,
                0.192843496799469,
                0.3056538701057434,
                0.40208005905151367,
                0.5672317147254944
            ],
            "num.trees": [
                0.8321493268013,
                0.9060224890708923,
                0.9482557773590088,
                0.9735459089279175,
                0.9920570254325867
            ],
            "sample.fraction": [
                0.09004493802785873,
                0.2657637596130371,
                0.45976343750953674,
                0.6665132641792297,
                0.884817898273468
            ]
        },
        "mtry": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "每棵树分裂时随机选择的特征数量"
        },
        "num.trees": {
            "type": "float64",
            "low": 0.0,
            "high": 1.0,
            "description": "森林中树的数量"
        },
        "sample.fraction": {
            "type": "float64",
            "low": 0.0,
            "high": 0.999419732,
            "description": "训练每棵树时所用样本的比例"
        }
    }
}
def duplicate_SEARCH_SPACE_INFO():
    search = {}
    for s in SEARCH_SPACE_INFO:
        if SEARCH_SPACE_INFO[s]["model_name"] not in search.keys():
            search[SEARCH_SPACE_INFO[s]["model_name"]] = [SEARCH_SPACE_INFO[s]["parameters_name"]]
        else:
            search[SEARCH_SPACE_INFO[s]["model_name"]].append(SEARCH_SPACE_INFO[s]["parameters_name"])

    print(json.dumps(search, indent=4, ensure_ascii=False))
    print(len(search))


# SEARCH_SPACE_INFO = {
#     "4796": {
#         "ml_model": "rpart.preproc",
#         "parameters_name": [
#             "minsplit",
#             "minbucket",
#             "cp"
#         ],
#         "minsplit": {
#             "type": "categorical",
#             "categories": [
#                 0.333328261,
#                 0.166662628,
#                 0.666663285,
#                 0.0,
#                 1.0,
#                 0.833331549,
#                 0.499995398
#             ],
#             "description": "划分节点所需的最小样本数"
#         },
#         "minbucket": {
#             "type": "categorical",
#             "categories": [
#                 0.16665859,
#                 0.666659904,
#                 0.33332319,
#                 0.499990796,
#                 0.833329764,
#                 0.0,
#                 1.0
#             ],
#             "description": "每个叶子节点的最小样本数"
#         },
#         "cp": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "剪枝复杂度参数，控制模型的复杂度"
#         }
#     },
#     "5527": {
#         "ml_model": "svm",
#         "parameters_name": [
#             "cost",
#             "gamma",
#             "gamma.na",
#             "degree",
#             "degree.na",
#             "kernel.ohe.na",
#             "kernel.ohe.linear",
#             "kernel.ohe.polynomial"
#         ],
#         "cost": {
#             "type": "float64",
#             "low": 1.0724331833633832e-06,
#             "high": 0.9999989042752158,
#             "description": "惩罚参数，控制对错误分类的惩罚力度"
#         },
#         "gamma": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 0.9999844117884508,
#             "description": "核函数的参数，影响决策边界的形状"
#         },
#         "gamma.na": {
#             "type": "categorical",
#             "categories": [
#                 0.0,
#                 1.0
#             ],
#             "description": "对于缺失值的处理方法"
#         },
#         "degree": {
#             "type": "categorical",
#             "categories": [
#                 0.0,
#                 0.4,
#                 0.8,
#                 1.0
#             ],
#             "description": "多项式核的度数"
#         },
#         "degree.na": {
#             "type": "categorical",
#             "categories": [
#                 1.0,
#                 0.0
#             ],
#             "description": "对于缺失值的处理方法"
#         },
#         "kernel.ohe.na": {
#             "type": "categorical",
#             "categories": [
#                 1.0,
#                 0.0
#             ],
#             "description": "处理核类型缺失值的方法"
#         },
#         "kernel.ohe.linear": {
#             "type": "categorical",
#             "categories": [
#                 0.0,
#                 1.0
#             ],
#             "description": "使用线性核"
#         },
#         "kernel.ohe.polynomial": {
#             "type": "categorical",
#             "categories": [
#                 0.0,
#                 1.0
#             ],
#             "description": "使用多项式核"
#         }
#     },
#     "5636": {
#         "ml_model": "rpart",
#         "parameters_name": [
#             "minsplit",
#             "minsplit.na",
#             "minbucket",
#             "cp",
#             "maxdepth",
#             "maxdepth.na"
#         ],
#         "minsplit": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "划分节点所需的最小样本数"
#         },
#         "minsplit.na": {
#             "type": "categorical",
#             "categories": [
#                 0.0,
#                 1.0
#             ],
#             "description": "对于缺失值的处理方法"
#         },
#         "minbucket": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "每个叶子节点的最小样本数"
#         },
#         "cp": {
#             "type": "float64",
#             "low": 0.0017861445988133,
#             "high": 1.0,
#             "description": "剪枝复杂度参数，控制模型的复杂度"
#         },
#         "maxdepth": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "树的最大深度"
#         },
#         "maxdepth.na": {
#             "type": "categorical",
#             "categories": [
#                 0.0,
#                 1.0
#             ],
#             "description": "对于缺失值的处理方法"
#         }
#     },
#     "5859": {
#         "ml_model": "rpart",
#         "parameters_name": [
#             "minsplit",
#             "minsplit.na",
#             "minbucket",
#             "cp",
#             "maxdepth",
#             "maxdepth.na"
#         ],
#         "minsplit": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "划分节点所需的最小样本数"
#         },
#         "minsplit.na": {
#             "type": "categorical",
#             "categories": [
#                 0.0,
#                 1.0
#             ],
#             "description": "对于缺失值的处理方法"
#         },
#         "minbucket": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "每个叶子节点的最小样本数"
#         },
#         "cp": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "剪枝复杂度参数，控制模型的复杂度"
#         },
#         "maxdepth": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "树的最大深度"
#         },
#         "maxdepth.na": {
#             "type": "categorical",
#             "categories": [
#                 0.0,
#                 1.0
#             ],
#             "description": "对于缺失值的处理方法"
#         }
#     },
#     "5860": {
#         "ml_model": "glmnet",
#         "parameters_name": [
#             "alpha",
#             "lambda"
#         ],
#         "alpha": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "控制L1和L2正则化的比例"
#         },
#         "lambda": {
#             "type": "float64",
#             "low": 0.0001217023571036,
#             "high": 1.0,
#             "description": "正则化参数，控制模型的复杂度"
#         }
#     },
#     "5891": {
#         "ml_model": "svm",
#         "parameters_name": [
#             "cost",
#             "gamma",
#             "gamma.na",
#             "degree",
#             "degree.na",
#             "kernel.ohe.na",
#             "kernel.ohe.linear",
#             "kernel.ohe.polynomial"
#         ],
#         "cost": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "惩罚参数，控制对错误分类的惩罚力度"
#         },
#         "gamma": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "核函数的参数，影响决策边界的形状"
#         },
#         "gamma.na": {
#             "type": "categorical",
#             "categories": [
#                 1.0,
#                 0.0
#             ],
#             "description": "对于缺失值的处理方法"
#         },
#         "degree": {
#             "type": "categorical",
#             "categories": [
#                 0.0,
#                 0.8,
#                 1.0,
#                 0.4
#             ],
#             "description": "多项式核的度数"
#         },
#         "degree.na": {
#             "type": "categorical",
#             "categories": [
#                 1.0,
#                 0.0
#             ],
#             "description": "对于缺失值的处理方法"
#         },
#         "kernel.ohe.na": {
#             "type": "categorical",
#             "categories": [
#                 0.0,
#                 1.0
#             ],
#             "description": "处理核类型缺失值的方法"
#         },
#         "kernel.ohe.linear": {
#             "type": "categorical",
#             "categories": [
#                 1.0,
#                 0.0
#             ],
#             "description": "使用线性核"
#         },
#         "kernel.ohe.polynomial": {
#             "type": "categorical",
#             "categories": [
#                 0.0,
#                 1.0
#             ],
#             "description": "使用多项式核"
#         }
#     },
#     "5906": {
#         "ml_model": "xgboost",
#         "parameters_name": [
#             "eta",
#             "max_depth",
#             "max_depth.na",
#             "min_child_weight",
#             "min_child_weight.na",
#             "subsample",
#             "colsample_bytree",
#             "colsample_bytree.na",
#             "colsample_bylevel",
#             "colsample_bylevel.na",
#             "lambda",
#             "alpha",
#             "nrounds",
#             "nrounds.na",
#             "booster.ohe.na",
#             "booster.ohe.gblinear"
#         ],
#         "eta": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "学习率，控制每次迭代的步长"
#         },
#         "max_depth": {
#             "type": "categorical",
#             "categories": [
#                 0.8,
#                 0.0,
#                 0.066666667,
#                 0.733333333,
#                 0.2,
#                 0.666666667,
#                 0.266666667,
#                 0.133333333,
#                 0.533333333,
#                 0.866666667,
#                 0.466666667,
#                 1.0,
#                 0.333333333,
#                 0.933333333,
#                 0.6
#             ],
#             "description": "树的最大深度"
#         },
#         "max_depth.na": {
#             "type": "categorical",
#             "categories": [
#                 0,
#                 1
#             ],
#             "description": "对于缺失值的处理方法"
#         },
#         "min_child_weight": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "子节点中最小的样本权重和"
#         },
#         "min_child_weight.na": {
#             "type": "categorical",
#             "categories": [
#                 0,
#                 1
#             ],
#             "description": "对于缺失值的处理方法"
#         },
#         "subsample": {
#             "type": "float64",
#             "low": 4.72e-05,
#             "high": 0.999830676,
#             "description": "用于训练模型的样本比例"
#         },
#         "colsample_bytree": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "构建每棵树时使用的特征比例"
#         },
#         "colsample_bytree.na": {
#             "type": "categorical",
#             "categories": [
#                 0,
#                 1
#             ],
#             "description": "对于缺失值的处理方法"
#         },
#         "colsample_bylevel": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "每一层的特征采样比例"
#         },
#         "colsample_bylevel.na": {
#             "type": "categorical",
#             "categories": [
#                 0,
#                 1
#             ],
#             "description": "对于缺失值的处理方法"
#         },
#         "lambda": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "L2正则化参数"
#         },
#         "alpha": {
#             "type": "float64",
#             "low": 2.92e-05,
#             "high": 0.999910278,
#             "description": "L1正则化参数"
#         },
#         "nrounds": {
#             "type": "float64",
#             "low": 0.558652872,
#             "high": 1.0,
#             "description": "迭代次数"
#         },
#         "nrounds.na": {
#             "type": "categorical",
#             "categories": [
#                 0
#             ],
#             "description": "对于缺失值的处理方法"
#         },
#         "booster.ohe.na": {
#             "type": "categorical",
#             "categories": [
#                 1,
#                 0
#             ],
#             "description": "处理提升器缺失值的方法"
#         },
#         "booster.ohe.gblinear": {
#             "type": "categorical",
#             "categories": [
#                 0,
#                 1
#             ],
#             "description": "使用线性提升器"
#         }
#     },
#     "5965": {
#         "ml_model": "ranger",
#         "parameters_name": [
#             "num.trees",
#             "num.trees.na",
#             "mtry",
#             "sample.fraction",
#             "min.node.size",
#             "min.node.size.na",
#             "replace.ohe.FALSE",
#             "replace.ohe.na",
#             "respect.unordered.factors.ohe.INVALID",
#             "respect.unordered.factors.ohe.TRUE"
#         ],
#         "num.trees": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "森林中树的数量"
#         },
#         "num.trees.na": {
#             "type": "categorical",
#             "categories": [
#                 0.0,
#                 1.0
#             ],
#             "description": "对于缺失值的处理方法"
#         },
#         "mtry": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "每棵树随机选择的特征数量"
#         },
#         "sample.fraction": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "用于构建树的样本比例"
#         },
#         "min.node.size": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "叶子节点的最小样本数"
#         },
#         "min.node.size.na": {
#             "type": "categorical",
#             "categories": [
#                 1.0,
#                 0.0
#             ],
#             "description": "对于缺失值的处理方法"
#         },
#         "replace.ohe.FALSE": {
#             "type": "categorical",
#             "categories": [
#                 0.0,
#                 1.0
#             ],
#             "description": "是否允许替换样本"
#         },
#         "replace.ohe.na": {
#             "type": "categorical",
#             "categories": [
#                 1.0,
#                 0.0
#             ],
#             "description": "对于缺失值的处理方法"
#         },
#         "respect.unordered.factors.ohe.INVALID": {
#             "type": "categorical",
#             "categories": [
#                 1.0,
#                 0.0
#             ],
#             "description": "处理无序因子的方式"
#         },
#         "respect.unordered.factors.ohe.TRUE": {
#             "type": "categorical",
#             "categories": [
#                 0.0,
#                 1.0
#             ],
#             "description": "处理无序因子的方式"
#         }
#     },
#     "5970": {
#         "ml_model": "ranger",
#         "parameters_name": [
#             "alpha",
#             "lambda"
#         ],
#         "alpha": {
#             "type": "float64",
#             "low": 1.3486429008569235e-05,
#             "high": 0.999993216854508,
#             "description": "控制L1正则化"
#         },
#         "lambda": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "控制L2正则化"
#         }
#     },
#     "5971": {
#         "ml_model": "xgboost",
#         "parameters_name": [
#             "eta",
#             "max_depth",
#             "max_depth.na",
#             "min_child_weight",
#             "min_child_weight.na",
#             "subsample",
#             "colsample_bytree",
#             "colsample_bytree.na",
#             "colsample_bylevel",
#             "colsample_bylevel.na",
#             "lambda",
#             "alpha",
#             "nrounds",
#             "nrounds.na",
#             "booster.ohe.na",
#             "booster.ohe.gblinear"
#         ],
#         "eta": {
#             "type": "float64",
#             "low": 1.0128443936672635e-05,
#             "high": 1.0,
#             "description": "学习率，控制每次迭代的步长"
#         },
#         "max_depth": {
#             "type": "categorical",
#             "categories": [
#                 0.0,
#                 0.466666667,
#                 0.533333333,
#                 1.0,
#                 0.133333333,
#                 0.2,
#                 0.066666667,
#                 0.266666667,
#                 0.333333333,
#                 0.866666667,
#                 0.733333333,
#                 0.666666667,
#                 0.6,
#                 0.933333333,
#                 0.8
#             ],
#             "description": "树的最大深度"
#         },
#         "max_depth.na": {
#             "type": "categorical",
#             "categories": [
#                 1.0,
#                 0.0
#             ],
#             "description": "对于缺失值的处理方法"
#         },
#         "min_child_weight": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 0.9999944553298932,
#             "description": "子节点中最小的样本权重和"
#         },
#         "min_child_weight.na": {
#             "type": "categorical",
#             "categories": [
#                 1.0,
#                 0.0
#             ],
#             "description": "对于缺失值的处理方法"
#         },
#         "subsample": {
#             "type": "float64",
#             "low": 4.676697646707956e-05,
#             "high": 1.0,
#             "description": "用于训练模型的样本比例"
#         },
#         "colsample_bytree": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "构建每棵树时使用的特征比例"
#         },
#         "colsample_bytree.na": {
#             "type": "categorical",
#             "categories": [
#                 1.0,
#                 0.0
#             ],
#             "description": "对于缺失值的处理方法"
#         },
#         "colsample_bylevel": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "每一层的特征采样比例"
#         },
#         "colsample_bylevel.na": {
#             "type": "categorical",
#             "categories": [
#                 1.0,
#                 0.0
#             ],
#             "description": "对于缺失值的处理方法"
#         },
#         "lambda": {
#             "type": "float64",
#             "low": 2.441609784760115e-05,
#             "high": 1.0,
#             "description": "L2正则化参数"
#         },
#         "alpha": {
#             "type": "float64",
#             "low": 3.526446474087557e-06,
#             "high": 1.0,
#             "description": "L1正则化参数"
#         },
#         "nrounds": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "迭代次数"
#         },
#         "nrounds.na": {
#             "type": "categorical",
#             "categories": [
#                 0.0,
#                 1.0
#             ],
#             "description": "对于缺失值的处理方法"
#         },
#         "booster.ohe.na": {
#             "type": "categorical",
#             "categories": [
#                 0.0,
#                 1.0
#             ],
#             "description": "处理提升器缺失值的方法"
#         },
#         "booster.ohe.gblinear": {
#             "type": "categorical",
#             "categories": [
#                 1.0,
#                 0.0
#             ],
#             "description": "使用线性提升器"
#         }
#     },
#     "6766": {
#         "ml_model": "glmnet",
#         "parameters_name": [
#             "alpha",
#             "lambda"
#         ],
#         "alpha": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 0.9999992982479988,
#             "description": "控制L1和L2正则化的比例"
#         },
#         "lambda": {
#             "type": "float64",
#             "low": 3.13363541299707e-12,
#             "high": 1.0,
#             "description": "正则化参数，控制模型的复杂度"
#         }
#     },
#     "6767": {
#         "ml_model": "xgboost",
#         "parameters_name": [
#             "eta",
#             "subsample",
#             "lambda",
#             "alpha",
#             "nthread",
#             "nthread.na",
#             "nrounds",
#             "nrounds.na",
#             "max_depth",
#             "max_depth.na",
#             "min_child_weight",
#             "min_child_weight.na",
#             "colsample_bytree",
#             "colsample_bytree.na",
#             "colsample_bylevel",
#             "colsample_bylevel.na",
#             "booster.ohe.na",
#             "booster.ohe.gblinear"
#         ],
#         "eta": {
#             "type": "float64",
#             "low": 1.45e-08,
#             "high": 0.999996288,
#             "description": "学习率，控制每次迭代的步长"
#         },
#         "subsample": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 0.99999371,
#             "description": "用于训练模型的样本比例"
#         },
#         "lambda": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 0.999968404,
#             "description": "L2正则化参数"
#         },
#         "alpha": {
#             "type": "float64",
#             "low": 6.82e-12,
#             "high": 1.0,
#             "description": "L1正则化参数"
#         },
#         "nthread": {
#             "type": "categorical",
#             "categories": [
#                 0,
#                 1
#             ],
#             "description": "线程数，控制并行计算"
#         },
#         "nthread.na": {
#             "type": "categorical",
#             "categories": [
#                 1,
#                 0
#             ],
#             "description": "对于缺失值的处理方法"
#         },
#         "nrounds": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "迭代次数"
#         },
#         "nrounds.na": {
#             "type": "categorical",
#             "categories": [
#                 0,
#                 1
#             ],
#             "description": "对于缺失值的处理方法"
#         },
#         "max_depth": {
#             "type": "categorical",
#             "categories": [
#                 0.0,
#                 1.0,
#                 0.533333333,
#                 0.333333333,
#                 0.8,
#                 0.866666667,
#                 0.066666667,
#                 0.733333333,
#                 0.2,
#                 0.933333333,
#                 0.6,
#                 0.466666667,
#                 0.666666667,
#                 0.266666667,
#                 0.133333333
#             ],
#             "description": "树的最大深度"
#         },
#         "max_depth.na": {
#             "type": "categorical",
#             "categories": [
#                 1,
#                 0
#             ],
#             "description": "对于缺失值的处理方法"
#         },
#         "min_child_weight": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "子节点中最小的样本权重和"
#         },
#         "min_child_weight.na": {
#             "type": "categorical",
#             "categories": [
#                 1,
#                 0
#             ],
#             "description": "对于缺失值的处理方法"
#         },
#         "colsample_bytree": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "构建每棵树时使用的特征比例"
#         },
#         "colsample_bytree.na": {
#             "type": "categorical",
#             "categories": [
#                 1,
#                 0
#             ],
#             "description": "对于缺失值的处理方法"
#         },
#         "colsample_bylevel": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 0.999988003,
#             "description": "每一层的特征采样比例"
#         },
#         "colsample_bylevel.na": {
#             "type": "categorical",
#             "categories": [
#                 1,
#                 0
#             ],
#             "description": "对于缺失值的处理方法"
#         },
#         "booster.ohe.na": {
#             "type": "categorical",
#             "categories": [
#                 0,
#                 1
#             ],
#             "description": "处理提升器缺失值的方法"
#         },
#         "booster.ohe.gblinear": {
#             "type": "categorical",
#             "categories": [
#                 1,
#                 0
#             ],
#             "description": "使用线性提升器"
#         }
#     },
#     "6794": {
#         "ml_model": "ranger",
#         "parameters_name": [
#             "num.trees",
#             "num.trees.na",
#             "mtry",
#             "sample.fraction",
#             "min.node.size",
#             "min.node.size.na",
#             "replace.ohe.FALSE",
#             "replace.ohe.na",
#             "respect.unordered.factors.ohe.na",
#             "respect.unordered.factors.ohe.TRUE"
#         ],
#         "num.trees": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "森林中树的数量"
#         },
#         "num.trees.na": {
#             "type": "categorical",
#             "categories": [
#                 0,
#                 1
#             ],
#             "description": "对于缺失值的处理方法"
#         },
#         "mtry": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "每棵树随机选择的特征数量"
#         },
#         "sample.fraction": {
#             "type": "float64",
#             "low": 8.48e-07,
#             "high": 1.0,
#             "description": "用于构建树的样本比例"
#         },
#         "min.node.size": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "叶子节点的最小样本数"
#         },
#         "min.node.size.na": {
#             "type": "categorical",
#             "categories": [
#                 0,
#                 1
#             ],
#             "description": "对于缺失值的处理方法"
#         },
#         "replace.ohe.FALSE": {
#             "type": "categorical",
#             "categories": [
#                 0,
#                 1
#             ],
#             "description": "是否允许替换样本"
#         },
#         "replace.ohe.na": {
#             "type": "categorical",
#             "categories": [
#                 1,
#                 0
#             ],
#             "description": "对于缺失值的处理方法"
#         },
#         "respect.unordered.factors.ohe.na": {
#             "type": "categorical",
#             "categories": [
#                 1,
#                 0
#             ],
#             "description": "处理无序因子的方式"
#         },
#         "respect.unordered.factors.ohe.TRUE": {
#             "type": "categorical",
#             "categories": [
#                 0,
#                 1
#             ],
#             "description": "处理无序因子的方式"
#         }
#     },
#     "7607": {
#         "ml_model": "ranger",
#         "parameters_name": [
#             "num.trees",
#             "num.trees.na",
#             "mtry",
#             "min.node.size",
#             "sample.fraction",
#             "respect.unordered.factors.ohe.na",
#             "respect.unordered.factors.ohe.TRUE",
#             "replace.ohe.FALSE",
#             "replace.ohe.na"
#         ],
#         "num.trees": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "森林中树的数量"
#         },
#         "num.trees.na": {
#             "type": "categorical",
#             "categories": [
#                 0.0,
#                 1.0
#             ],
#             "description": "对于缺失值的处理方法"
#         },
#         "mtry": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "每棵树随机选择的特征数量"
#         },
#         "min.node.size": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 0.9998894939362448,
#             "description": "叶子节点的最小样本数"
#         },
#         "sample.fraction": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "用于构建树的样本比例"
#         },
#         "respect.unordered.factors.ohe.na": {
#             "type": "categorical",
#             "categories": [
#                 0.0,
#                 1.0
#             ],
#             "description": "处理无序因子的方式"
#         },
#         "respect.unordered.factors.ohe.TRUE": {
#             "type": "categorical",
#             "categories": [
#                 1.0,
#                 0.0
#             ],
#             "description": "处理无序因子的方式"
#         },
#         "replace.ohe.FALSE": {
#             "type": "categorical",
#             "categories": [
#                 0.0,
#                 1.0
#             ],
#             "description": "是否允许替换样本"
#         },
#         "replace.ohe.na": {
#             "type": "categorical",
#             "categories": [
#                 1.0,
#                 0.0
#             ],
#             "description": "对于缺失值的处理方法"
#         }
#     },
#     "7609": {
#         "ml_model": "ranger",
#         "parameters_name": [
#             "num.trees",
#             "num.trees.na",
#             "mtry",
#             "min.node.size",
#             "sample.fraction",
#             "respect.unordered.factors.ohe.na",
#             "respect.unordered.factors.ohe.TRUE",
#             "replace.ohe.FALSE",
#             "replace.ohe.na"
#         ],
#         "num.trees": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "森林中树的数量"
#         },
#         "num.trees.na": {
#             "type": "categorical",
#             "categories": [
#                 0.0,
#                 1.0
#             ],
#             "description": "处理缺失值的树的数量"
#         },
#         "mtry": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "每棵树分裂时随机选择的特征数量"
#         },
#         "min.node.size": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "每个叶子节点的最小样本数"
#         },
#         "sample.fraction": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "训练每棵树时所用样本的比例"
#         },
#         "respect.unordered.factors.ohe.na": {
#             "type": "categorical",
#             "categories": [
#                 0.0,
#                 1.0
#             ],
#             "description": "处理无序因子的策略"
#         },
#         "respect.unordered.factors.ohe.TRUE": {
#             "type": "categorical",
#             "categories": [
#                 1.0,
#                 0.0
#             ],
#             "description": "处理无序因子时的另一种策略"
#         },
#         "replace.ohe.FALSE": {
#             "type": "categorical",
#             "categories": [
#                 1.0,
#                 0.0
#             ],
#             "description": "是否允许重复抽样"
#         },
#         "replace.ohe.na": {
#             "type": "categorical",
#             "categories": [
#                 0.0,
#                 1.0
#             ],
#             "description": "处理缺失值的抽样方法"
#         }
#     },
#     "5889": {
#         "ml_model": "ranger",
#         "parameters_name": [
#             "num.trees",
#             "num.trees.na",
#             "mtry",
#             "sample.fraction",
#             "replace.ohe.FALSE",
#             "replace.ohe.na"
#         ],
#         "num.trees": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "森林中树的数量"
#         },
#         "num.trees.na": {
#             "type": "categorical",
#             "categories": [
#                 0,
#                 1
#             ],
#             "description": "处理缺失值的树的数量"
#         },
#         "mtry": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 1.0,
#             "description": "每棵树分裂时随机选择的特征数量"
#         },
#         "sample.fraction": {
#             "type": "float64",
#             "low": 0.0,
#             "high": 0.999419732,
#             "description": "训练每棵树时所用样本的比例"
#         },
#         "replace.ohe.FALSE": {
#             "type": "categorical",
#             "categories": [
#                 0,
#                 1
#             ],
#             "description": "是否允许重复抽样"
#         },
#         "replace.ohe.na": {
#             "type": "categorical",
#             "categories": [
#                 1,
#                 0
#             ],
#             "description": "处理缺失值的抽样方法"
#         }
#     }
# }
#
# import json
#
# import pandas as pd
#
# data = pd.read_csv(r"D:\LLM-Driven_AI-Studio\HPO-KDEA\data\benchmarks\hpob\data\mlcopilot_space.csv")
# space_id = data[:16]["space_id"].values
# quantile_info = data[:16]["quantile_info"].values
#
# results = {
#
# }
# for i in range(len(space_id)):
#
#     print(space_id[i], quantile_info[i])
#     quantile_info_json = json.loads(quantile_info[i])
#     parameters_name = [col for col in quantile_info_json]
#     results[str(space_id[i])] = {
#         "model_name": SEARCH_SPACE_INFO[str(space_id[i])]["ml_model"],
#         "parameters_name": parameters_name,
#         "quantile_info": quantile_info_json
#     }
#     for p in parameters_name:
#         results[str(space_id[i])][p] = SEARCH_SPACE_INFO[str(space_id[i])][p]
#     with open("results.json", "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=4, ensure_ascii=False)