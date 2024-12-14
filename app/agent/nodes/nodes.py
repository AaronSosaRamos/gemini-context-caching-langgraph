from app.tools.tools import tools
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from functools import lru_cache

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

@lru_cache(maxsize=4)
def _get_model(model_name: str):
    if model_name == "openai":
        model = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    elif model_name == "gemini":
        model = ChatGoogleGenerativeAI(temperature=0, model="gemini-1.5-pro")
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    model = model.bind_tools(tools)
    return model

def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    return "continue"

def call_model(state, config):
    messages = state["messages"]
    messages = [{"role": "system", "content": "Be a helpful assistant"}] + messages
    model_name = config.get("configurable", {}).get("model_name", "gemini")
    model = _get_model(model_name)
    response = model.invoke(messages)
    return {"messages": [response]}
