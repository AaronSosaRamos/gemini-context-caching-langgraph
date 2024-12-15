from app.tools.tools import tools
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from functools import lru_cache
import datetime
import time
import re
import google.generativeai as genai
from google.generativeai import caching
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

llm_for_decision_making = ChatGoogleGenerativeAI(temperature=0, model="gemini-1.5-pro")

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

def decide_tool_or_cache(state):

    if "decision" in state and "init_decision" in state and state["init_decision"] and state["decision"]:
        return state

    messages_for_genai = [
        ("system", "You are an expert router."),
        ("human", f"""
        Evaluate the appropriate action based on the given message:
        {state["messages"][-1].content}
        Choose one of the following options:
        - 'tool': If the task requires searching or accessing the internet for information.
        - 'create_cache': If the task explicitly mentions creating a new cache.
        - 'update_cache': If the task explicitly mentions updating the cache.
        - 'get_cache': If the task explicitly mentions using the cached context for retrieving the requested info.
        - 'delete_cache': If the task explicitly mentions to delete the cache.
        - 'none': If the last message is the final response and not a task. If the message refers to the cache, you must not use 'none'.

        If the request is not explicitly about cache operations but resembles an information query or task for which you have no context stored, choose 'tool'.

        Return only the name of the selected action.
        """)
    ]
    
    result = llm_for_decision_making.invoke(messages_for_genai)
    decision = re.sub(r'```', '', result.content.strip())

    print(f"Decision: {decision}")

    return {
        "decision": decision
    }


def call_model(state, config):
    messages = state["messages"]
    messages = [{"role": "system", "content": "Be a helpful assistant"}] + messages
    model_name = config.get("configurable", {}).get("model_name", "openai")
    model = _get_model(model_name)
    response = model.invoke(messages)
    return {"messages": [response]}

def create_cache_node(state) -> str:
    """Create a cache for storing context information."""
    model_name="models/gemini-1.5-flash-001"
    ttl=5

    cache = caching.CachedContent.create(
        model=model_name, 
        contents=[state["text"]], 
        display_name='Genesis Retriever',
        system_instruction=(
            'You are an expert remembering the book of Genesis from the Bible'
            'the user\'s query based on the text you have access to.'
        ),
        ttl=datetime.timedelta(minutes=ttl),
    )

    print(f"Cache Name Result: {cache.name}")

    return {
        "cache_name": cache.name
    }

def use_cache_node(state):
    """Use an existing cache."""
    cache = caching.CachedContent.get(state["cache_name"])
    if cache:
        model = genai.GenerativeModel.from_cached_content(cached_content=cache)
        response = model.generate_content([(state["query"])])
        return {
            "response": response.text
        }

def update_cache_node(state):
    """Update the time-to-live of an existing cache."""
    cache = caching.CachedContent.get(state["cache_name"])
    if cache:
        cache.update(ttl=datetime.timedelta(minutes=state["ttl"]))
        return {
            "update_or_delete": "updated"
        }

def delete_cache_node(state):
    """Delete an existing cache to manage costs."""
    cache = caching.CachedContent.get(state["cache_name"])
    if cache:
        cache.delete()
        return {
            "update_or_delete": "deleted"
        }