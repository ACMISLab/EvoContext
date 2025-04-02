CANDIDATE_SAMPLER_PROMPT = """
The following are examples of the performance of a {model} measured in {metric} and the corresponding model hyperparameter configurations.
The model is evaluated on a tabular {task} task containing {classes_number} classes.
The tabular dataset contains {samples_number} samples and {features_number} features ({categorical_features_number} categorical, {continuous_features_number} numerical).
The allowable ranges for the hyperparameters are: {hyperparameters_info}.
Recommend a configuration that can achieve the target performance of {target_score}.
Do not recommend values at the minimum or maximum of allowable range, do not recommend rounded values.
Recommend values with the highest possible precision, as requested by the allowed ranges.
Your response must only contain the predicted configuration, in the format ## configuration ##.
Here are some examples:
{cases}
You can't copy the example hyperparameter configurations given. You must generate a new one.
Performance: {target_score}
Hyperparameter configuration:
"""
SURROGATE_MODEL_PROMPT = """The following are examples of the performance of a {model} measured in {metric} and the corresponding model hyperparameter configurations.
The model is evaluated on a tabular {task} task containing {classes_number} classes.
The tabular dataset contains {samples_number} samples and {features_number} features ({categorical_features_number} categorical, {continuous_features_number} numerical).
Your response should only contain the predicted accuracy in the format ## performance ##.
{cases}
Hyperparameter configuration: {evaluated_configuration}
Performance:
"""



