# -*- coding: utf-8 -*-
import json
import time
from llm.base_model import Base_Model
from utils.logs import logger
import openai
import asyncio

class GLM_Local_Model(Base_Model):

    def predict(self, messages, temperature=0, top_p=1, max_tokens=4096):
        from config import Model_Local_PATH
        openai.api_key = "EMPTY"
        openai.api_base = Model_Local_PATH

        logger.info("llm input:\n" + json.dumps(messages, ensure_ascii=False, indent=4))

        retry_time = 1
        max_time = 3

        for i in range(max_time):
            try:
                rep = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens
                )
                message = rep.choices[0].message
                input_tokens = rep.usage.prompt_tokens
                output_tokens = rep.usage.completion_tokens

                price = self.costs(input_tokens, output_tokens)

                logger.info(f"input_tokens: {input_tokens}, output_tokens: {output_tokens}, price: {price}")
                return message["content"], message, input_tokens, output_tokens, price

            except Exception as e:
                logger.info(f"llm调用失败，重试第{retry_time}次，错误信息：{e}")
                time.sleep(5)
                retry_time += 1
        raise Exception("llm调用失败")

    def costs(self, input_tokens, output_tokens):

        return 0


    async def async_predict(self, messages, temperature=0, top_p=1):

        from config import Model_Local_PATH
        openai.api_key = "EMPTY"
        openai.api_base = Model_Local_PATH
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
                        request_timeout=120
                    )
                    break
                except Exception as e:
                    logger.info(f'[AF] RETRYING LLM REQUEST {retry + 1}/{MAX_RETRIES}...')
                    logger.info(resp)
                    logger.info(e)

        await openai.aiosession.get().close()

        if resp is None:
            raise Exception("LLM调用失败")
        # logger.info(f"{resp['choices'][0]['message']['content']}")

        return resp['choices'][0]['message']['content'], resp['usage']['prompt_tokens'], resp['usage']['completion_tokens']

async def main():
    model = GLM_Local_Model(model="glm-4-9b-chat")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "# role\nyou are a machine learning model parameter configuration optimizer. \n\n# task\nYou are given a dataset and a search space. Your task is to generate a better machine learning model parameter configuration by evolving the given parameter configurations.\n\n# dataset\nThe dataset name is \"blood-transfusion-service-center\".\nThe dataset contains 2 classes, 748 instances, 5 features, 4 numeric features, 1 categorical features.\nThe target variable has the following characteristics: \nThe \"1\" class size is 570.The \"2\" class size is 178.\n\n# search space\nThe \"rpart\" has parameters: \n[{\"type\": \"float64\", \"low\": 0.0017861445988133, \"high\": 1.0, \"description\": \"剪枝复杂度参数，控制模型的复杂度\"}, {\"type\": \"float64\", \"low\": 0.0, \"high\": 1.0, \"description\": \"树的最大深度\"}, {\"type\": \"float64\", \"low\": 0.0, \"high\": 1.0, \"description\": \"每个叶子节点的最小样本数\"}, {\"type\": \"float64\", \"low\": 0.0, \"high\": 1.0, \"description\": \"划分节点所需的最小样本数\"}].\n\n# workflow\nPlease follow the instruction step-by-step to generate a better machine learning model parameter configuration.\n1. Crossover the following parameter configurations and generate a new parameter configuration:\nParameter configuration 1: {\"cp\": 0.5237, \"maxdepth\": 0.4865, \"minbucket\": 0.2986, \"minsplit\": 0.4736}\nParameter configuration 2: {\"cp\": 0.5238, \"maxdepth\": 0.4865, \"minbucket\": 0.2986, \"minsplit\": 0.4736}\n2. Mutate the parameter configuration generated in Step 1 and generate a final parameter configuration bracketed with <configuration> and </configuration>.\n\n# Remember\nParameter configuration must be a JSON: {param1: value1, param2: value2, ...}\nDon't give the process of crossover and mutation\n\n# output format\nYour output can only contain the following two lines:\n1. Crossover Configuration: <the parameter configuration in the first step>\n2. <configuration><the final parameter configuration></configuration>\n"
        }
    ]

    # messages = [{
    #     "role": "user", "content": "贵阳在哪里！"
    # }]

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
            print(response[0])
            print(response[1])
            results.append(response)

if __name__ == '__main__':
    asyncio.run(main())
    # model = GLM_Local_Model("glm-4-9b-chat")
    # messages = [
    #     {
    #         "role": "user",
    #         "content": "你好！"
    #     }
    # ]
    #
    # content, message, input_tokens, output_tokens, price = model.predict(messages)
    #
    # print(json.dumps(message, ensure_ascii=False, indent=4))

