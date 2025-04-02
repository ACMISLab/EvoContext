from config import MODEL_NAME

def get_llm():
    if MODEL_NAME in [
        "gpt-4o-mini",
        "gpt-4o"
    ]:
        from llm.gpt import GPT_Model
        return GPT_Model(MODEL_NAME)

    elif MODEL_NAME in [
        "glm-4"
    ]:
        from llm.glm import GLM_Model
        return GLM_Model(MODEL_NAME)

    elif MODEL_NAME in [
        "deepseek-reasoner",
        "deepseek-chat",
    ]:
        from llm.deepseek import DeepSeek
        return DeepSeek(MODEL_NAME)

    elif MODEL_NAME in [
        "glm4-9b-chat"
    ]:
        from llm.glm import GLM_Local_Model
        return GLM_Local_Model(MODEL_NAME)

    elif MODEL_NAME in [
        "qwen2-72b-instruct",
        "Qwen2.5-14B-Instruct",
        "Qwen2.5-72B-Instruct",
        "qwen2.5-72b-instruct"
    ]:
        from llm.qwen import Qwen_Model
        return Qwen_Model(MODEL_NAME)

    elif MODEL_NAME in [
        "Meta-Llama-3.1-70B-Instruct",
        "Meta-Llama-3.1-8B-Instruct",
        "Meta-Llama-3-8B-Instruct",
        "Meta-Llama-3-70B-Instruct",
        "meta/llama3-70b-instruct"
    ]:
        from llm.llama import LLama_Model
        return LLama_Model(MODEL_NAME)

    elif MODEL_NAME in [
        "yi-1_5-34b-chat",
        "yi-1_5-9b-chat"
    ]:
        from llm.yi import Yi_Model
        return Yi_Model(MODEL_NAME)
    else:
        raise ValueError(f"Unsupported model: {MODEL_NAME}")