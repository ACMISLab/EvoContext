
"""
The dataset name is {name}, and it is a {type} task.
The dataset has {num_samples} samples and {len(features)} features.
It uses {features} as features to predict {target}. 
The target variable has the following characteristics: {target_characteristics}.
或者
The dataset name is "{name}".
It contains 2 classes, 6598 instances, 168 features, 166 numeric features, 2 categorical features.
The majority class size is 5581 and the minority class size is 1017.
"""
"""
"dataset_id": {
    "name": "dataset name",
    "type": "task type",  # classification or regression
    "num_samples": 1000,  # number of samples
    "features": ["feature1", "feature2", "feature3"],  # list of features
    "target": "target_variable",  # the variable to be predicted
    "target_characteristics": "target_characteristics"
    # classification: list of classes(classe1有n个，classe2有m个，...), regression: range of values（四分位数：min，25%，50%，75%，max）
}
"""

DATASETS_INFO = {
    "2": {
        "name": "anneal",
        "type": "classification",
        "num_samples": 898,
        "numeric_features": 6,
        "categorical_features": 33,
        "target": "class",
        "target_characteristics": {
            "1": [
                0,
                8
            ],
            "2": [
                1,
                99
            ],
            "3": [
                2,
                684
            ],
            "5": [
                4,
                67
            ],
            "U": [
                5,
                40
            ]
        }
    },
    "3": {
        "name": "kr-vs-kp",
        "type": "classification",
        "num_samples": 3196,
        "numeric_features": 0,
        "categorical_features": 37,
        "target": "class",
        "target_characteristics": {
            "won": [
                0,
                1669
            ],
            "nowin": [
                1,
                1527
            ]
        }
    },
    "11": {
        "name": "balance-scale",
        "type": "classification",
        "num_samples": 625,
        "numeric_features": 4,
        "categorical_features": 1,
        "target": "class",
        "target_characteristics": {
            "L": [
                2,
                288
            ],
            "R": [
                2,
                288
            ],
            "B": [
                1,
                49
            ]
        }
    },
    "15": {
        "name": "breast-w",
        "type": "classification",
        "num_samples": 699,
        "numeric_features": 9,
        "categorical_features": 1,
        "target": "Class",
        "target_characteristics": {
            "benign": [
                0,
                458
            ],
            "malignant": [
                1,
                241
            ]
        }
    },
    "18": {
        "name": "mfeat-morphological",
        "type": "classification",
        "num_samples": 2000,
        "numeric_features": 6,
        "categorical_features": 1,
        "target": "class",
        "target_characteristics": {
            "1": [
                9,
                200
            ],
            "2": [
                9,
                200
            ],
            "3": [
                9,
                200
            ],
            "4": [
                9,
                200
            ],
            "5": [
                9,
                200
            ],
            "6": [
                9,
                200
            ],
            "7": [
                9,
                200
            ],
            "8": [
                9,
                200
            ],
            "9": [
                9,
                200
            ],
            "10": [
                9,
                200
            ]
        }
    },
    "21": {
        "name": "car",
        "type": "classification",
        "num_samples": 1728,
        "numeric_features": 0,
        "categorical_features": 7,
        "target": "class",
        "target_characteristics": {
            "acc": [
                0,
                384
            ],
            "good": [
                1,
                69
            ],
            "unacc": [
                2,
                1210
            ],
            "vgood": [
                3,
                65
            ]
        }
    },
    "23": {
        "name": "cmc",
        "type": "classification",
        "num_samples": 1473,
        "numeric_features": 2,
        "categorical_features": 8,
        "target": "Contraceptive_method_used",
        "target_characteristics": {
            "1": [
                0,
                629
            ],
            "2": [
                1,
                333
            ],
            "3": [
                2,
                511
            ]
        }
    },
    "29": {
        "name": "credit-approval",
        "type": "classification",
        "num_samples": 690,
        "numeric_features": 6,
        "categorical_features": 10,
        "target": "class",
        "target_characteristics": {
            "+": [
                0,
                307
            ],
            "-": [
                1,
                383
            ]
        }
    },
    "31": {
        "name": "credit-g",
        "type": "classification",
        "num_samples": 1000,
        "numeric_features": 7,
        "categorical_features": 14,
        "target": "class",
        "target_characteristics": {
            "good": [
                0,
                700
            ],
            "bad": [
                1,
                300
            ]
        }
    },
    "36": {
        "name": "segment",
        "type": "classification",
        "num_samples": 2310,
        "numeric_features": 19,
        "categorical_features": 1,
        "target": "class",
        "target_characteristics": {
            "brickface": [
                6,
                330
            ],
            "sky": [
                6,
                330
            ],
            "foliage": [
                6,
                330
            ],
            "cement": [
                6,
                330
            ],
            "window": [
                6,
                330
            ],
            "path": [
                6,
                330
            ],
            "grass": [
                6,
                330
            ]
        }
    },
    "37": {
        "name": "diabetes",
        "type": "classification",
        "num_samples": 768,
        "numeric_features": 8,
        "categorical_features": 1,
        "target": "class",
        "target_characteristics": {
            "tested_negative": [
                0,
                500
            ],
            "tested_positive": [
                1,
                268
            ]
        }
    },
    "41": {
        "name": "soybean",
        "type": "classification",
        "num_samples": 683,
        "numeric_features": 0,
        "categorical_features": 36,
        "target": "class",
        "target_characteristics": {
            "diaporthe-stem-canker": [
                12,
                20
            ],
            "charcoal-rot": [
                12,
                20
            ],
            "rhizoctonia-root-rot": [
                12,
                20
            ],
            "powdery-mildew": [
                12,
                20
            ],
            "downy-mildew": [
                12,
                20
            ],
            "bacterial-blight": [
                12,
                20
            ],
            "bacterial-pustule": [
                12,
                20
            ],
            "purple-seed-stain": [
                12,
                20
            ],
            "phyllosticta-leaf-spot": [
                12,
                20
            ],
            "phytophthora-rot": [
                3,
                88
            ],
            "brown-stem-rot": [
                11,
                44
            ],
            "anthracnose": [
                11,
                44
            ],
            "brown-spot": [
                7,
                92
            ],
            "alternarialeaf-spot": [
                14,
                91
            ],
            "frog-eye-leaf-spot": [
                14,
                91
            ],
            "diaporthe-pod-&-stem-blight": [
                15,
                15
            ],
            "cyst-nematode": [
                16,
                14
            ],
            "2-4-d-injury": [
                17,
                16
            ],
            "herbicide-injury": [
                18,
                8
            ]
        }
    },
    "43": {
        "name": "spambase",
        "type": "classification",
        "num_samples": 4601,
        "numeric_features": 57,
        "categorical_features": 1,
        "target": "class",
        "target_characteristics": {
            "0": [
                0,
                2788
            ],
            "1": [
                1,
                1813
            ]
        }
    },
    "49": {
        "name": "tic-tac-toe",
        "type": "classification",
        "num_samples": 958,
        "numeric_features": 0,
        "categorical_features": 10,
        "target": "Class",
        "target_characteristics": {
            "negative": [
                0,
                332
            ],
            "positive": [
                1,
                626
            ]
        }
    },
    "53": {
        "name": "vehicle",
        "type": "classification",
        "num_samples": 846,
        "numeric_features": 18,
        "categorical_features": 1,
        "target": "Class",
        "target_characteristics": {
            "opel": [
                0,
                212
            ],
            "saab": [
                1,
                217
            ],
            "bus": [
                2,
                218
            ],
            "van": [
                3,
                199
            ]
        }
    },
    "219": {
        "name": "electricity",
        "type": "classification",
        "num_samples": 45312,
        "numeric_features": 7,
        "categorical_features": 2,
        "target": "class",
        "target_characteristics": {
            "UP": [
                0,
                19237
            ],
            "DOWN": [
                1,
                26075
            ]
        }
    },
    "272": {
        "name": "haberman",
        "type": "classification",
        "num_samples": 306,
        "numeric_features": 2,
        "categorical_features": 2,
        "target": "Survival_status",
        "target_characteristics": {
            "1": [
                0,
                225
            ],
            "2": [
                1,
                81
            ]
        }
    },
    "282": {
        "name": "heart-statlog",
        "type": "classification",
        "num_samples": 270,
        "numeric_features": 13,
        "categorical_features": 1,
        "target": "class",
        "target_characteristics": {
            "absent": [
                0,
                150
            ],
            "present": [
                1,
                120
            ]
        }
    },
    "2079": {
        "name": "eucalyptus",
        "type": "classification",
        "num_samples": 736,
        "numeric_features": 14,
        "categorical_features": 6,
        "target": "Utility",
        "target_characteristics": {
            "none": [
                0,
                180
            ],
            "low": [
                1,
                107
            ],
            "average": [
                2,
                130
            ],
            "good": [
                3,
                214
            ],
            "best": [
                4,
                105
            ]
        }
    },
    "3022": {
        "name": "vowel",
        "type": "classification",
        "num_samples": 990,
        "numeric_features": 10,
        "categorical_features": 3,
        "target": "Class",
        "target_characteristics": {
            "hid": [
                10,
                90
            ],
            "hId": [
                10,
                90
            ],
            "hEd": [
                10,
                90
            ],
            "hAd": [
                10,
                90
            ],
            "hYd": [
                10,
                90
            ],
            "had": [
                10,
                90
            ],
            "hOd": [
                10,
                90
            ],
            "hod": [
                10,
                90
            ],
            "hUd": [
                10,
                90
            ],
            "hud": [
                10,
                90
            ],
            "hed": [
                10,
                90
            ]
        }
    },
    "3485": {
        "name": "scene",
        "type": "classification",
        "num_samples": 2407,
        "numeric_features": 294,
        "categorical_features": 6,
        "target": "Urban",
        "target_characteristics": {
            "0": [
                0,
                1976
            ],
            "1": [
                1,
                431
            ]
        }
    },
    "3492": {
        "name": "monks-problems-1",
        "type": "classification",
        "num_samples": 556,
        "numeric_features": 0,
        "categorical_features": 7,
        "target": "class",
        "target_characteristics": {
            "0": [
                1,
                278
            ],
            "1": [
                1,
                278
            ]
        }
    },
    "3493": {
        "name": "monks-problems-2",
        "type": "classification",
        "num_samples": 601,
        "numeric_features": 0,
        "categorical_features": 7,
        "target": "class",
        "target_characteristics": {
            "0": [
                0,
                395
            ],
            "1": [
                1,
                206
            ]
        }
    },
    "3494": {
        "name": "monks-problems-3",
        "type": "classification",
        "num_samples": 554,
        "numeric_features": 0,
        "categorical_features": 7,
        "target": "class",
        "target_characteristics": {
            "0": [
                0,
                266
            ],
            "1": [
                1,
                288
            ]
        }
    },
    "3512": {
        "name": "synthetic_control",
        "type": "classification",
        "num_samples": 600,
        "numeric_features": 60,
        "categorical_features": 2,
        "target": "class",
        "target_characteristics": {
            "Cyclic": [
                5,
                100
            ],
            "Decreasing_trend": [
                5,
                100
            ],
            "Downward_shift": [
                5,
                100
            ],
            "Increasing_trend": [
                5,
                100
            ],
            "Normal": [
                5,
                100
            ],
            "Upward_shift": [
                5,
                100
            ]
        }
    },
    "3543": {
        "name": "irish",
        "type": "classification",
        "num_samples": 500,
        "numeric_features": 2,
        "categorical_features": 4,
        "target": "Leaving_Certificate",
        "target_characteristics": {
            "not_taken": [
                0,
                278
            ],
            "taken": [
                1,
                222
            ]
        }
    },
    "3549": {
        "name": "analcatdata_authorship",
        "type": "classification",
        "num_samples": 841,
        "numeric_features": 70,
        "categorical_features": 1,
        "target": "Author",
        "target_characteristics": {
            "Austen": [
                0,
                317
            ],
            "London": [
                1,
                296
            ],
            "Milton": [
                2,
                55
            ],
            "Shakespeare": [
                3,
                173
            ]
        }
    },
    "3560": {
        "name": "analcatdata_dmft",
        "type": "classification",
        "num_samples": 797,
        "numeric_features": 0,
        "categorical_features": 5,
        "target": "Prevention",
        "target_characteristics": {
            "All_methods": [
                0,
                127
            ],
            "Diet_enrichment": [
                1,
                132
            ],
            "Health_education": [
                2,
                124
            ],
            "Mouthwash": [
                3,
                155
            ],
            "None": [
                4,
                136
            ],
            "Oral_hygiene": [
                5,
                123
            ]
        }
    },
    "3561": {
        "name": "profb",
        "type": "classification",
        "num_samples": 672,
        "numeric_features": 5,
        "categorical_features": 5,
        "target": "Home/Away",
        "target_characteristics": {
            "at_home": [
                0,
                448
            ],
            "away": [
                1,
                224
            ]
        }
    },
    "3567": {
        "name": "collins",
        "type": "classification",
        "num_samples": 500,
        "numeric_features": 20,
        "categorical_features": 4,
        "target": "Corp.Genre",
        "target_characteristics": {
            "101": [
                0,
                44
            ],
            "102": [
                1,
                27
            ],
            "103": [
                3,
                17
            ],
            "104": [
                3,
                17
            ],
            "105": [
                4,
                36
            ],
            "106": [
                5,
                48
            ],
            "107": [
                6,
                75
            ],
            "108": [
                7,
                30
            ],
            "109": [
                8,
                80
            ],
            "110": [
                13,
                29
            ],
            "113": [
                13,
                29
            ],
            "114": [
                13,
                29
            ],
            "111": [
                10,
                24
            ],
            "112": [
                11,
                6
            ],
            "115": [
                14,
                9
            ]
        }
    },
    "3889": {
        "name": "sylva_agnostic",
        "type": "classification",
        "num_samples": 14395,
        "numeric_features": 216,
        "categorical_features": 1,
        "target": "label",
        "target_characteristics": {
            "-1": [
                0,
                13509
            ],
            "1": [
                1,
                886
            ]
        }
    },
    "3891": {
        "name": "gina_agnostic",
        "type": "classification",
        "num_samples": 3468,
        "numeric_features": 970,
        "categorical_features": 1,
        "target": "label",
        "target_characteristics": {
            "-1": [
                0,
                1763
            ],
            "1": [
                1,
                1705
            ]
        }
    },
    "3896": {
        "name": "ada_agnostic",
        "type": "classification",
        "num_samples": 4562,
        "numeric_features": 48,
        "categorical_features": 1,
        "target": "label",
        "target_characteristics": {
            "-1": [
                0,
                3430
            ],
            "1": [
                1,
                1132
            ]
        }
    },
    "3899": {
        "name": "mozilla4",
        "type": "classification",
        "num_samples": 15545,
        "numeric_features": 5,
        "categorical_features": 1,
        "target": "state",
        "target_characteristics": {
            "0": [
                0,
                5108
            ],
            "1": [
                1,
                10437
            ]
        }
    },
    "3902": {
        "name": "pc4",
        "type": "classification",
        "num_samples": 1458,
        "numeric_features": 37,
        "categorical_features": 1,
        "target": "c",
        "target_characteristics": {
            "false": [
                0.0,
                1280
            ],
            "true": [
                1.0,
                178
            ]
        }
    },
    "3903": {
        "name": "pc3",
        "type": "classification",
        "num_samples": 1563,
        "numeric_features": 37,
        "categorical_features": 1,
        "target": "c",
        "target_characteristics": {
            "false": [
                0.0,
                1403
            ],
            "true": [
                1.0,
                160
            ]
        }
    },
    "3913": {
        "name": "kc2",
        "type": "classification",
        "num_samples": 522,
        "numeric_features": 21,
        "categorical_features": 1,
        "target": "problems",
        "target_characteristics": {
            "no": [
                0,
                415
            ],
            "yes": [
                1,
                107
            ]
        }
    },
    "3917": {
        "name": "kc1",
        "type": "classification",
        "num_samples": 2109,
        "numeric_features": 21,
        "categorical_features": 1,
        "target": "defects",
        "target_characteristics": {
            "false": [
                0.0,
                1783
            ],
            "true": [
                1.0,
                326
            ]
        }
    },
    "3918": {
        "name": "pc1",
        "type": "classification",
        "num_samples": 1109,
        "numeric_features": 21,
        "categorical_features": 1,
        "target": "defects",
        "target_characteristics": {
            "false": [
                0.0,
                1032
            ],
            "true": [
                1.0,
                77
            ]
        }
    },
    "3950": {
        "name": "musk",
        "type": "classification",
        "num_samples": 6598,
        "numeric_features": 167,
        "categorical_features": 3,
        "target": "class",
        "target_characteristics": {
            "0": [
                0,
                5581
            ],
            "1": [
                1,
                1017
            ]
        }
    },
    "3954": {
        "name": "MagicTelescope",
        "type": "classification",
        "num_samples": 19020,
        "numeric_features": 11,
        "categorical_features": 1,
        "target": "class:",
        "target_characteristics": {
            "g": [
                0,
                12332
            ],
            "h": [
                1,
                6688
            ]
        }
    },
    "7295": {
        "name": "Click_prediction_small",
        "type": 0,
        "num_samples": 39948,
        "numeric_features": 11,
        "categorical_features": 1,
        "target": "click",
        "target_characteristics": {
            "regression_value_range": [
                0.0,
                0.0,
                0.0,
                0.0,
                1.0
            ]
        }
    },
    "9889": {
        "name": "wilt",
        "type": "classification",
        "num_samples": 4839,
        "numeric_features": 5,
        "categorical_features": 1,
        "target": "Class",
        "target_characteristics": {
            "1": [
                0,
                4578
            ],
            "2": [
                1,
                261
            ]
        }
    },
    "9903": {
        "name": "autoUniv-au7-500",
        "type": "classification",
        "num_samples": 500,
        "numeric_features": 8,
        "categorical_features": 5,
        "target": "Class",
        "target_characteristics": {
            "class1": [
                0,
                43
            ],
            "class2": [
                1,
                71
            ],
            "class3": [
                2,
                85
            ],
            "class4": [
                3,
                109
            ],
            "class5": [
                4,
                192
            ]
        }
    },
    "9905": {
        "name": "autoUniv-au7-700",
        "type": "classification",
        "num_samples": 700,
        "numeric_features": 8,
        "categorical_features": 5,
        "target": "Class",
        "target_characteristics": {
            "class1": [
                0,
                214
            ],
            "class2": [
                1,
                245
            ],
            "class3": [
                2,
                241
            ]
        }
    },
    "9906": {
        "name": "autoUniv-au7-1100",
        "type": "classification",
        "num_samples": 1100,
        "numeric_features": 8,
        "categorical_features": 5,
        "target": "Class",
        "target_characteristics": {
            "class1": [
                0,
                153
            ],
            "class2": [
                1,
                246
            ],
            "class3": [
                2,
                216
            ],
            "class4": [
                3,
                180
            ],
            "class5": [
                4,
                305
            ]
        }
    },
    "9910": {
        "name": "Bioresponse",
        "type": "classification",
        "num_samples": 3751,
        "numeric_features": 1776,
        "categorical_features": 1,
        "target": "target",
        "target_characteristics": {
            "0": [
                0,
                1717
            ],
            "1": [
                1,
                2034
            ]
        }
    },
    "9911": {
        "name": "Amazon_employee_access",
        "type": "classification",
        "num_samples": 32769,
        "numeric_features": 0,
        "categorical_features": 10,
        "target": "target",
        "target_characteristics": {
            "0": [
                0,
                1897
            ],
            "1": [
                1,
                30872
            ]
        }
    },
    "9914": {
        "name": "wilt",
        "type": "classification",
        "num_samples": 4839,
        "numeric_features": 5,
        "categorical_features": 1,
        "target": "Class",
        "target_characteristics": {
            "1": [
                0,
                4578
            ],
            "2": [
                1,
                261
            ]
        }
    },
    "9946": {
        "name": "wdbc",
        "type": "classification",
        "num_samples": 569,
        "numeric_features": 30,
        "categorical_features": 1,
        "target": "Class",
        "target_characteristics": {
            "1": [
                0,
                357
            ],
            "2": [
                1,
                212
            ]
        }
    },
    "9952": {
        "name": "phoneme",
        "type": "classification",
        "num_samples": 5404,
        "numeric_features": 5,
        "categorical_features": 1,
        "target": "Class",
        "target_characteristics": {
            "1": [
                0,
                3818
            ],
            "2": [
                1,
                1586
            ]
        }
    },
    "9957": {
        "name": "qsar-biodeg",
        "type": "classification",
        "num_samples": 1055,
        "numeric_features": 41,
        "categorical_features": 1,
        "target": "Class",
        "target_characteristics": {
            "1": [
                0,
                699
            ],
            "2": [
                1,
                356
            ]
        }
    },
    "9967": {
        "name": "steel-plates-fault",
        "type": "classification",
        "num_samples": 1941,
        "numeric_features": 33,
        "categorical_features": 1,
        "target": "Class",
        "target_characteristics": {
            "1": [
                0,
                1268
            ],
            "2": [
                1,
                673
            ]
        }
    },
    "9970": {
        "name": "hill-valley",
        "type": "classification",
        "num_samples": 1212,
        "numeric_features": 100,
        "categorical_features": 1,
        "target": "Class",
        "target_characteristics": {
            "0": [
                1,
                606
            ],
            "1": [
                1,
                606
            ]
        }
    },
    "9971": {
        "name": "ilpd",
        "type": "classification",
        "num_samples": 583,
        "numeric_features": 9,
        "categorical_features": 2,
        "target": "Class",
        "target_characteristics": {
            "1": [
                0,
                416
            ],
            "2": [
                1,
                167
            ]
        }
    },
    "9976": {
        "name": "madelon",
        "type": "classification",
        "num_samples": 2600,
        "numeric_features": 500,
        "categorical_features": 1,
        "target": "Class",
        "target_characteristics": {
            "1": [
                1,
                1300
            ],
            "2": [
                1,
                1300
            ]
        }
    },
    "9977": {
        "name": "nomao",
        "type": "classification",
        "num_samples": 34465,
        "numeric_features": 89,
        "categorical_features": 30,
        "target": "Class",
        "target_characteristics": {
            "1": [
                0,
                9844
            ],
            "2": [
                1,
                24621
            ]
        }
    },
    "9978": {
        "name": "ozone-level-8hr",
        "type": "classification",
        "num_samples": 2534,
        "numeric_features": 72,
        "categorical_features": 1,
        "target": "Class",
        "target_characteristics": {
            "1": [
                0,
                2374
            ],
            "2": [
                1,
                160
            ]
        }
    },
    "9979": {
        "name": "cardiotocography",
        "type": "classification",
        "num_samples": 2126,
        "numeric_features": 35,
        "categorical_features": 1,
        "target": "Class",
        "target_characteristics": {
            "1": [
                0,
                384
            ],
            "2": [
                1,
                579
            ],
            "3": [
                2,
                53
            ],
            "4": [
                3,
                81
            ],
            "5": [
                4,
                72
            ],
            "6": [
                5,
                332
            ],
            "7": [
                6,
                252
            ],
            "8": [
                7,
                107
            ],
            "9": [
                8,
                69
            ],
            "10": [
                9,
                197
            ]
        }
    },
    "9980": {
        "name": "climate-model-simulation-crashes",
        "type": "classification",
        "num_samples": 540,
        "numeric_features": 20,
        "categorical_features": 1,
        "target": "Class",
        "target_characteristics": {
            "1": [
                0,
                46
            ],
            "2": [
                1,
                494
            ]
        }
    },
    "9983": {
        "name": "eeg-eye-state",
        "type": "classification",
        "num_samples": 14980,
        "numeric_features": 14,
        "categorical_features": 1,
        "target": "Class",
        "target_characteristics": {
            "1": [
                0,
                8257
            ],
            "2": [
                1,
                6723
            ]
        }
    },
    "10093": {
        "name": "banknote-authentication",
        "type": "classification",
        "num_samples": 1372,
        "numeric_features": 4,
        "categorical_features": 1,
        "target": "Class",
        "target_characteristics": {
            "1": [
                0,
                762
            ],
            "2": [
                1,
                610
            ]
        }
    },
    "10101": {
        "name": "blood-transfusion-service-center",
        "type": "classification",
        "num_samples": 748,
        "numeric_features": 4,
        "categorical_features": 1,
        "target": "Class",
        "target_characteristics": {
            "1": [
                0,
                570
            ],
            "2": [
                1,
                178
            ]
        }
    },
    "14951": {
        "name": "eeg-eye-state",
        "type": "classification",
        "num_samples": 14980,
        "numeric_features": 14,
        "categorical_features": 1,
        "target": "Class",
        "target_characteristics": {
            "1": [
                0,
                8257
            ],
            "2": [
                1,
                6723
            ]
        }
    },
    "14952": {
        "name": "PhishingWebsites",
        "type": "classification",
        "num_samples": 11055,
        "numeric_features": 0,
        "categorical_features": 31,
        "target": "Result",
        "target_characteristics": {
            "-1": [
                0,
                4898
            ],
            "1": [
                1,
                6157
            ]
        }
    },
    "14964": {
        "name": "artificial-characters",
        "type": "classification",
        "num_samples": 10218,
        "numeric_features": 7,
        "categorical_features": 1,
        "target": "Class",
        "target_characteristics": {
            "1": [
                0,
                1196
            ],
            "2": [
                1,
                1192
            ],
            "3": [
                2,
                1416
            ],
            "4": [
                3,
                808
            ],
            "5": [
                4,
                1008
            ],
            "6": [
                8,
                1000
            ],
            "9": [
                8,
                1000
            ],
            "7": [
                6,
                800
            ],
            "8": [
                7,
                1198
            ],
            "10": [
                9,
                600
            ]
        }
    },
    "14965": {
        "name": "bank-marketing",
        "type": "classification",
        "num_samples": 45211,
        "numeric_features": 7,
        "categorical_features": 10,
        "target": "Class",
        "target_characteristics": {
            "1": [
                0,
                39922
            ],
            "2": [
                1,
                5289
            ]
        }
    },
    "14966": {
        "name": "Bioresponse",
        "type": "classification",
        "num_samples": 3751,
        "numeric_features": 1776,
        "categorical_features": 1,
        "target": "target",
        "target_characteristics": {
            "0": [
                0,
                1717
            ],
            "1": [
                1,
                2034
            ]
        }
    },
    "14968": {
        "name": "cylinder-bands",
        "type": "classification",
        "num_samples": 540,
        "numeric_features": 18,
        "categorical_features": 22,
        "target": "band_type",
        "target_characteristics": {
            "band": [
                0,
                228
            ],
            "noband": [
                1,
                312
            ]
        }
    },
    "14971": {
        "name": "Click_prediction_small",
        "type": 0,
        "num_samples": 39948,
        "numeric_features": 11,
        "categorical_features": 1,
        "target": "click",
        "target_characteristics": {
            "regression_value_range": [
                0.0,
                0.0,
                0.0,
                0.0,
                1.0
            ]
        }
    },
    "34537": {
        "name": "PhishingWebsites",
        "type": "classification",
        "num_samples": 11055,
        "numeric_features": 0,
        "categorical_features": 31,
        "target": "Result",
        "target_characteristics": {
            "-1": [
                0,
                4898
            ],
            "1": [
                1,
                6157
            ]
        }
    },
    "34539": {
        "name": "Amazon_employee_access",
        "type": "classification",
        "num_samples": 32769,
        "numeric_features": 0,
        "categorical_features": 10,
        "target": "target",
        "target_characteristics": {
            "0": [
                0,
                1897
            ],
            "1": [
                1,
                30872
            ]
        }
    },
    "125923": {
        "name": "Australian",
        "type": 0,
        "num_samples": 690,
        "numeric_features": 14,
        "categorical_features": 1,
        "target": "Y",
        "target_characteristics": {
            "regression_value_range": [
                0.0,
                0.0,
                0.0,
                1.0,
                1.0
            ]
        }
    },
    "145677": {
        "name": "Bioresponse",
        "type": "classification",
        "num_samples": 3751,
        "numeric_features": 1776,
        "categorical_features": 1,
        "target": "target",
        "target_characteristics": {
            "0": [
                0,
                1717
            ],
            "1": [
                1,
                2034
            ]
        }
    },
    "145804": {
        "name": "tic-tac-toe",
        "type": "classification",
        "num_samples": 958,
        "numeric_features": 0,
        "categorical_features": 10,
        "target": "Class",
        "target_characteristics": {
            "negative": [
                0,
                332
            ],
            "positive": [
                1,
                626
            ]
        }
    },
    "145833": {
        "name": "bank-marketing",
        "type": "classification",
        "num_samples": 45211,
        "numeric_features": 7,
        "categorical_features": 10,
        "target": "Class",
        "target_characteristics": {
            "1": [
                0,
                39922
            ],
            "2": [
                1,
                5289
            ]
        }
    },
    "145834": {
        "name": "banknote-authentication",
        "type": "classification",
        "num_samples": 1372,
        "numeric_features": 4,
        "categorical_features": 1,
        "target": "Class",
        "target_characteristics": {
            "1": [
                0,
                762
            ],
            "2": [
                1,
                610
            ]
        }
    },
    "145836": {
        "name": "blood-transfusion-service-center",
        "type": "classification",
        "num_samples": 748,
        "numeric_features": 4,
        "categorical_features": 1,
        "target": "Class",
        "target_characteristics": {
            "1": [
                0,
                570
            ],
            "2": [
                1,
                178
            ]
        }
    },
    "145839": {
        "name": "climate-model-simulation-crashes",
        "type": "classification",
        "num_samples": 540,
        "numeric_features": 20,
        "categorical_features": 1,
        "target": "Class",
        "target_characteristics": {
            "1": [
                0,
                46
            ],
            "2": [
                1,
                494
            ]
        }
    },
    "145847": {
        "name": "hill-valley",
        "type": "classification",
        "num_samples": 1212,
        "numeric_features": 100,
        "categorical_features": 1,
        "target": "Class",
        "target_characteristics": {
            "0": [
                1,
                606
            ],
            "1": [
                1,
                606
            ]
        }
    },
    "145848": {
        "name": "ilpd",
        "type": "classification",
        "num_samples": 583,
        "numeric_features": 9,
        "categorical_features": 2,
        "target": "Class",
        "target_characteristics": {
            "1": [
                0,
                416
            ],
            "2": [
                1,
                167
            ]
        }
    },
    "145853": {
        "name": "madelon",
        "type": "classification",
        "num_samples": 2600,
        "numeric_features": 500,
        "categorical_features": 1,
        "target": "Class",
        "target_characteristics": {
            "1": [
                1,
                1300
            ],
            "2": [
                1,
                1300
            ]
        }
    },
    "145854": {
        "name": "nomao",
        "type": "classification",
        "num_samples": 34465,
        "numeric_features": 89,
        "categorical_features": 30,
        "target": "Class",
        "target_characteristics": {
            "1": [
                0,
                9844
            ],
            "2": [
                1,
                24621
            ]
        }
    },
    "145855": {
        "name": "ozone-level-8hr",
        "type": "classification",
        "num_samples": 2534,
        "numeric_features": 72,
        "categorical_features": 1,
        "target": "Class",
        "target_characteristics": {
            "1": [
                0,
                2374
            ],
            "2": [
                1,
                160
            ]
        }
    },
    "145857": {
        "name": "phoneme",
        "type": "classification",
        "num_samples": 5404,
        "numeric_features": 5,
        "categorical_features": 1,
        "target": "Class",
        "target_characteristics": {
            "1": [
                0,
                3818
            ],
            "2": [
                1,
                1586
            ]
        }
    },
    "145862": {
        "name": "qsar-biodeg",
        "type": "classification",
        "num_samples": 1055,
        "numeric_features": 41,
        "categorical_features": 1,
        "target": "Class",
        "target_characteristics": {
            "1": [
                0,
                699
            ],
            "2": [
                1,
                356
            ]
        }
    },
    "145872": {
        "name": "steel-plates-fault",
        "type": "classification",
        "num_samples": 1941,
        "numeric_features": 33,
        "categorical_features": 1,
        "target": "Class",
        "target_characteristics": {
            "1": [
                0,
                1268
            ],
            "2": [
                1,
                673
            ]
        }
    },
    "145878": {
        "name": "wdbc",
        "type": "classification",
        "num_samples": 569,
        "numeric_features": 30,
        "categorical_features": 1,
        "target": "Class",
        "target_characteristics": {
            "1": [
                0,
                357
            ],
            "2": [
                1,
                212
            ]
        }
    },
    "145953": {
        "name": "kr-vs-kp",
        "type": "classification",
        "num_samples": 3196,
        "numeric_features": 0,
        "categorical_features": 37,
        "target": "class",
        "target_characteristics": {
            "won": [
                0,
                1669
            ],
            "nowin": [
                1,
                1527
            ]
        }
    },
    "145972": {
        "name": "credit-g",
        "type": "classification",
        "num_samples": 1000,
        "numeric_features": 7,
        "categorical_features": 14,
        "target": "class",
        "target_characteristics": {
            "good": [
                0,
                700
            ],
            "bad": [
                1,
                300
            ]
        }
    },
    "145976": {
        "name": "diabetes",
        "type": "classification",
        "num_samples": 768,
        "numeric_features": 8,
        "categorical_features": 1,
        "target": "class",
        "target_characteristics": {
            "tested_negative": [
                0,
                500
            ],
            "tested_positive": [
                1,
                268
            ]
        }
    },
    "145979": {
        "name": "spambase",
        "type": "classification",
        "num_samples": 4601,
        "numeric_features": 57,
        "categorical_features": 1,
        "target": "class",
        "target_characteristics": {
            "0": [
                0,
                2788
            ],
            "1": [
                1,
                1813
            ]
        }
    },
    "146012": {
        "name": "electricity",
        "type": "classification",
        "num_samples": 45312,
        "numeric_features": 7,
        "categorical_features": 2,
        "target": "class",
        "target_characteristics": {
            "UP": [
                0,
                19237
            ],
            "DOWN": [
                1,
                26075
            ]
        }
    },
    "146064": {
        "name": "monks-problems-1",
        "type": "classification",
        "num_samples": 556,
        "numeric_features": 0,
        "categorical_features": 7,
        "target": "class",
        "target_characteristics": {
            "0": [
                1,
                278
            ],
            "1": [
                1,
                278
            ]
        }
    },
    "146065": {
        "name": "monks-problems-2",
        "type": "classification",
        "num_samples": 601,
        "numeric_features": 0,
        "categorical_features": 7,
        "target": "class",
        "target_characteristics": {
            "0": [
                0,
                395
            ],
            "1": [
                1,
                206
            ]
        }
    },
    "146066": {
        "name": "monks-problems-3",
        "type": "classification",
        "num_samples": 554,
        "numeric_features": 0,
        "categorical_features": 7,
        "target": "class",
        "target_characteristics": {
            "0": [
                0,
                266
            ],
            "1": [
                1,
                288
            ]
        }
    },
    "146082": {
        "name": "musk",
        "type": "classification",
        "num_samples": 6598,
        "numeric_features": 167,
        "categorical_features": 3,
        "target": "class",
        "target_characteristics": {
            "0": [
                0,
                5581
            ],
            "1": [
                1,
                1017
            ]
        }
    },
    "146803": {
        "name": "credit-g",
        "type": "classification",
        "num_samples": 1000,
        "numeric_features": 7,
        "categorical_features": 14,
        "target": "class",
        "target_characteristics": {
            "good": [
                0,
                700
            ],
            "bad": [
                1,
                300
            ]
        }
    }
}
