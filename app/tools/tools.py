import datetime
import time
from google.generativeai import caching, GenerativeModel
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

@tool
def create_cache_tool(text: str, model_name="models/gemini-1.5-flash-001", ttl:int=5) -> str:
    """Create a cache for storing context information."""
    cache = caching.CachedContent.create(
        model=model_name, 
        contents=[text], 
        display_name='Genesis Retriever',
        system_instruction=(
            'You are an expert remembering the book of Genesis from the Bible'
            'the user\'s query based on the text you have access to.'
        ),
        ttl=datetime.timedelta(minutes=ttl),
    )
    return cache.name

@tool
def use_cache_tool(cache_name: str, query: str):
    """Use an existing cache."""
    cache = caching.CachedContent.get(cache_name)
    if cache:
        model = GenerativeModel.from_cached_content(cached_content=cache)
        response = model.generate_content([(query)])
        return response.text

@tool
def update_cache_tool(cache_name: str, ttl: int):
    """Update the time-to-live of an existing cache."""
    cache = caching.CachedContent.get(cache_name)
    if cache:
        cache.update(ttl=datetime.timedelta(minutes=ttl))

@tool
def delete_cache_tool(cache_name: str):
    """Delete an existing cache to manage costs."""
    cache = caching.CachedContent.get(cache_name)
    if cache:
        cache.delete()

# List of tools
tools = [create_cache_tool, use_cache_tool, update_cache_tool, delete_cache_tool, TavilySearchResults(max_results=5)]