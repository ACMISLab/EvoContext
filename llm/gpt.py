# -*- coding: utf-8 -*-
import asyncio
import json
import os
import time

import openai
from llm.base_model import Base_Model
from utils.logs import logger

openai.api_key = ""
class GPT_Model(Base_Model):
    # model = ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125", "gpt-4", "gpt-4-32k", "gpt-4-0613", "gpt-4-32k-0613"]

    def predict(self, messages, tools=None, temperature=0, top_p=1, tool_choice=None):

        logger.info("llm input:\n" + json.dumps(messages, ensure_ascii=False, indent=4))

        retry_time = 1
        max_time = 3

        for i in range(max_time):
            try:
                rep = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=temperature,
                    top_p=top_p
                )
                message = rep.choices[0].message
                input_tokens = rep.usage.prompt_tokens
                output_tokens = rep.usage.completion_tokens

                price = self.costs(input_tokens, output_tokens)
                if message["content"] is not None and message["content"] != "":
                    logger.info("llm output:\n" + message["content"])

                logger.info(f"input_tokens: {input_tokens}, output_tokens: {output_tokens}, price: {price}")

                return message["content"], message, input_tokens, output_tokens, price

            except Exception as e:
                logger.info(f"llm调用失败，重试第{retry_time}次，错误信息：{e}")
                time.sleep(5)
                retry_time += 1
        raise Exception("llm调用失败")

    def costs(self, input_tokens, output_tokens):
        price = 0
        if self.model.rfind("ft:") != -1:
            input_price = 3
            output_price = 6
        elif self.model.rfind("gpt-4") != -1:
            input_price = 30
            output_price = 60
        elif self.model.rfind("gpt-3.5") != -1:
            input_price = 0.5
            output_price = 1.5

        price += (input_tokens * input_price + output_tokens * output_price) / 1000000

        return round(price, 4)

    async def async_predict(self, messages, tools=None, temperature=0, top_p=1, tool_choice=None, max_tokens=4096):
        from aiohttp import ClientSession
        '''Generate a response from the LLM async.'''

        MAX_RETRIES = 3

        async with ClientSession(trust_env=True) as session:
            openai.aiosession.set(session)

            resp = None
            for retry in range(MAX_RETRIES):
                try:
                    resp = await openai.ChatCompletion.acreate(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        n=5,
                        request_timeout=15
                    )
                    break
                except Exception as e:
                    print(f'[AF] RETRYING LLM REQUEST {retry+1}/{MAX_RETRIES}...')
                    print(resp)
                    print(e)

        await openai.aiosession.get().close()

        if resp is None:
            return None
        return resp['choices'][0]['message']['content'], resp['usage']['prompt_tokens'], resp['usage']['completion_tokens']

async def main():
    model = GPT_Model(model="gpt-4o-mini")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "# role\nyou are a machine learning model parameter configuration optimizer. \n\n# task\nYou are given a dataset and a search space. Your task is to generate a better machine learning model parameter configuration by evolving the given parameter configurations.\n\n# dataset\nThe dataset name is \"blood-transfusion-service-center\".\nThe dataset contains 2 classes, 748 instances, 5 features, 4 numeric features, 1 categorical features.\nThe target variable has the following characteristics: \nThe \"1\" class size is 570.The \"2\" class size is 178.\n\n# search space\nThe \"rpart\" has parameters: \n[{\"type\": \"float64\", \"low\": 0.0017861445988133, \"high\": 1.0, \"description\": \"剪枝复杂度参数，控制模型的复杂度\"}, {\"type\": \"float64\", \"low\": 0.0, \"high\": 1.0, \"description\": \"树的最大深度\"}, {\"type\": \"float64\", \"low\": 0.0, \"high\": 1.0, \"description\": \"每个叶子节点的最小样本数\"}, {\"type\": \"float64\", \"low\": 0.0, \"high\": 1.0, \"description\": \"划分节点所需的最小样本数\"}].\n\n# workflow\nPlease follow the instruction step-by-step to generate a better machine learning model parameter configuration.\n1. Crossover the following parameter configurations and generate a new parameter configuration:\nParameter configuration 1: {\"cp\": 0.5237, \"maxdepth\": 0.4865, \"minbucket\": 0.2986, \"minsplit\": 0.4736}\nParameter configuration 2: {\"cp\": 0.5238, \"maxdepth\": 0.4865, \"minbucket\": 0.2986, \"minsplit\": 0.4736}\n2. Mutate the parameter configuration generated in Step 1 and generate a final parameter configuration bracketed with <configuration> and </configuration>.\n\n# Remember\nParameter configuration must be a JSON: {param1: value1, param2: value2, ...}\nDon't give the process of crossover and mutation\n\n# output format\nYour output can only contain the following two lines:\n1. Crossover Configuration: <the parameter configuration in the first step>\n2. <configuration><the final parameter configuration></configuration>"
        }
    ]

    # content, message, input_tokens, output_tokens, price = model.predict(messages=messages)
    #
    # print(json.dumps(message, ensure_ascii=False, indent=4))

    coroutines = []
    coroutines.append(model.async_predict(messages))
    coroutines.append(model.async_predict(messages))

    tasks = [asyncio.create_task(c) for c in coroutines]
    results = []
    llm_response = await asyncio.gather(*tasks)
    for idx, response in enumerate(llm_response):
        if response is not None:
            content = response
            results.append(content)

    print(results)

if __name__ == '__main__':
    asyncio.run(main())
    #from config import MODEL_NAME
    # model = GPT_Model(MODEL_NAME)
    # messages = [
    #     {"role": "user", "content": "你好，你是谁？"},
    # ]
    # content, message, input_tokens, output_tokens, price = model.predict(messages=messages)
