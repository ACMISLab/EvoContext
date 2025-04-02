# -*- coding: utf-8 -*-
import json
from llm.base_model import Base_Model
from utils.logs import logger
import openai
import asyncio

class DeepSeek(Base_Model):

    def predict(self, messages, temperature=0, top_p=1, max_tokens=4096):
        openai.api_key = ""
        openai.api_base = "https://api.deepseek.com"

        logger.info("llm input:\n" + json.dumps(messages, ensure_ascii=False, indent=4))

        retry_time = 1
        max_time = 5

        for i in range(max_time):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens
                )
                message = response.choices[0].message
                # 处理为和GPT格式统一
                new_message = {
                    "role": message.role,
                    "content": message.content
                }
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                logger.info(f"input_tokens: {input_tokens}, output_tokens: {output_tokens}")
                return new_message["content"], new_message, input_tokens, output_tokens, 0
            except Exception as e:
                logger.info(f"llm调用失败，重试第{retry_time}次，错误信息：{e}")
                retry_time += 1
        raise Exception("llm调用失败")

    def costs(self, input_tokens, output_tokens):

        return 0

    async def async_predict(self, messages, temperature=0, top_p=1):
        openai.api_key = ""
        openai.api_base = "https://api.deepseek.com"
        from aiohttp import ClientSession
        '''Generate a response from the LLM async.'''

        MAX_RETRIES = 3
        # print(json.dumps(messages,indent=4,ensure_ascii=False))

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
                    if retry == 1:
                        temperature = 0.7
                        top_p = 1
                    if retry == 2:
                        temperature = 0.9
                        top_p = 1
                    logger.info(resp)
                    logger.info(e)

        await openai.aiosession.get().close()

        if resp is None:
            raise Exception("LLM调用失败")
        # logger.info(f"{resp['choices'][0]['message']['content']}")

        return resp['choices'][0]['message']['content'], resp['usage']['prompt_tokens'], resp['usage']['completion_tokens']

async def main():
    model = DeepSeek(model="deepseek-chat")

    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people find information."
        },
        {
            "role": "user",
            "content": "\nThe following are examples of the performance of a xgboost measured in accuracy and the corresponding model hyperparameter configurations.\nThe model is evaluated on a tabular classification task containing 2 classes.\nThe tabular dataset contains 583 samples and 11 features (2 categorical, 9 numerical).\nThe allowable ranges for the hyperparameters are: [{\"type\": \"float64\", \"low\": 2.92e-05, \"high\": 0.999910278, \"description\": \"L1正则化参数\"}, {\"type\": \"float64\", \"low\": 0.0, \"high\": 1.0, \"description\": \"每一层的特征采样比例\"}, {\"type\": \"float64\", \"low\": 0.0, \"high\": 1.0, \"description\": \"构建每棵树时使用的特征比例\"}, {\"type\": \"float64\", \"low\": 0.0, \"high\": 1.0, \"description\": \"学习率，控制每次迭代的步长\"}, {\"type\": \"float64\", \"low\": 0.0, \"high\": 1.0, \"description\": \"L2正则化参数\"}, {\"type\": \"categorical\", \"categories\": [0.8, 0.0, 0.06666667, 0.73333333, 0.2, 0.66666667, 0.26666667, 0.13333333, 0.53333333, 0.86666667, 0.46666667, 1.0, 0.33333333, 0.93333333, 0.6], \"description\": \"树的最大深度\"}, {\"type\": \"float64\", \"low\": 0.0, \"high\": 1.0, \"description\": \"子节点中最小的样本权重和\"}, {\"type\": \"float64\", \"low\": 0.558652872, \"high\": 1.0, \"description\": \"迭代次数\"}, {\"type\": \"float64\", \"low\": 4.72e-05, \"high\": 0.999830676, \"description\": \"用于训练模型的样本比例\"}].\nRecommend a configuration that can achieve the target performance of 0.7324182.\nDo not recommend values at the minimum or maximum of allowable range, do not recommend rounded values.\nRecommend values with the highest possible precision, as requested by the allowed ranges.\nYour response must only contain the predicted configuration, in the format ## configuration ##.\nHere are some examples:\n\nPerformance: 0.713551\nHyperparameter configuration: ## alpha: 0.8049736838338678, colsample_bylevel: 0.7666545853088954, colsample_bytree: 0.9517129165903844, eta: 0.7815575453585621, lambda: 0.7780057672220979, max_depth: 1.0, min_child_weight: 0.7299199073695285, nrounds: 0.9439452848788382, subsample: 0.211200679574313 ##\nPerformance: 0.713551\nHyperparameter configuration: ## alpha: 0.3703801872150644, colsample_bylevel: 0.0, colsample_bytree: 0.0, eta: 0.0199973538619943, lambda: 0.2205297068170072, max_depth: 0.0, min_child_weight: 0.0, nrounds: 0.879449078746802, subsample: 0.8083336128700723 ##\nPerformance: 0.730703\nHyperparameter configuration: ## alpha: 0.5596263400683806, colsample_bylevel: 0.0, colsample_bytree: 0.0, eta: 0.7295216647889207, lambda: 0.3634172282968541, max_depth: 0.0, min_child_weight: 0.0, nrounds: 0.9873137975877322, subsample: 0.0998757356696494 ##\nPerformance: 0.713551\nHyperparameter configuration: ## alpha: 0.340937202294592, colsample_bylevel: 0.0, colsample_bytree: 0.0, eta: 0.0274068910050737, lambda: 0.901425036995714, max_depth: 0.0, min_child_weight: 0.0, nrounds: 0.9396733974265232, subsample: 0.8896046613459456 ##\nPerformance: 0.713551\nHyperparameter configuration: ## alpha: 0.4029684228748469, colsample_bylevel: 0.3734374676279148, colsample_bytree: 0.8993407479430285, eta: 0.2903935548598776, lambda: 0.638621333651578, max_depth: 0.8666666666666667, min_child_weight: 0.9753392187987892, nrounds: 0.9340069818095804, subsample: 0.4581861441949834 ##\nYou can't copy the example hyperparameter configurations given. You must generate a new one.\nPerformance: 0.7324182\nHyperparameter configuration:\n"
        }
    ]
    coroutines = []
    coroutines.append(model.async_predict(messages))
    coroutines.append(model.async_predict(messages))

    tasks = [asyncio.create_task(c) for c in coroutines]
    results = []
    llm_response = await asyncio.gather(*tasks)
    for idx, response in enumerate(llm_response):
        if response is not None:
            print(response)
            results.append(response)

if __name__ == '__main__':
    asyncio.run(main())
    from config import MODEL_NAME
    model = DeepSeek(model=MODEL_NAME)

    messages = [
        {
            "role": "user",
            "content": "你好！",
        }
    ]
    content, message, input_tokens, output_tokens, price = model.predict(messages=messages)

    print(json.dumps(message, ensure_ascii=False, indent=4))
