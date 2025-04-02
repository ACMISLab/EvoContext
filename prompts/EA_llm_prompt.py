"""
Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts and generate a new prompt:
Prompt 1: Rewrite the input text into simpler text.
Prompt 2: Rewrite my complex sentence in simpler terms, but keep the meaning.
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1. Crossover Prompt: Rewrite the complex text into simpler text while keeping its meaning.
2. <prompt>Transform the provided text into simpler language, maintaining its essence.</prompt>

Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts and generate a new prompt:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.
"""

EVOLVE_PROMPT = """Please follow the instruction step-by-step to generate a better machine learning model parameter configuration.
1. Crossover the following parameter configurations and generate a new parameter configuration:
Parameter configuration 1: [0.5,0.7,0.2]
Parameter configuration 2: [0.4,0.7,0.3]
2. Mutate the parameter configuration generated in Step 1 and generate a final parameter configuration bracketed with <configuration> and </configuration>.

1. Crossover Configuration: [0.5,0.7,0.3]
2. <configuration>[0.3,0.7,0.3]</configuration>

Please follow the instruction step-by-step to generate a better machine learning model parameter configuration.
1. Crossover the following parameter configurations and generate a new parameter configuration:
Parameter configuration 1: {configuration1}
Parameter configuration 2: {configuration2}
2. Mutate the parameter configuration generated in Step 1 and generate a final parameter configuration bracketed with <configuration> and </configuration>."""

EVOLVE_PROMPT_CONTEXT = """# role
you are a machine learning model parameter configuration optimizer. 

# task
You are given a dataset and a search space. Your task is to generate a better machine learning model parameter configuration by evolving the given parameter configurations.

# dataset
The dataset name is "{dataset_name}".
The dataset contains {classes_number} classes, {num_samples} instances, {features_number} features, {numeric_features_number} numeric features, {categorical_features_number} categorical features.
The target variable has the following characteristics: 
{target_characteristics}.

# search space
The "{ml_model}" has parameters: 
{parameters}.

# workflow
Please follow the instruction step-by-step to generate a better machine learning model parameter configuration.
1. Crossover the following parameter configurations and generate a new parameter configuration:
Parameter configuration 1: {configuration1}
Parameter configuration 2: {configuration2}
Remember: For each parameter value, it must come from either Configuration 1 or Configuration 2.
2. Mutate the parameter configuration generated in Step 1 and generate a final parameter configuration bracketed with <configuration> and </configuration>.
Remember: Randomly select at least one parameter to choose a new parameter value at random from its range of values.

# Remember
Parameter configuration must be a JSON: {{param1: value1, param2: value2, ...}}.
Don't give the process of crossover and mutation.

# output format
Your output can only contain the following two lines:
1. Crossover Configuration: <the parameter configuration in the first step>
2. <configuration><the final parameter configuration></configuration>"""
