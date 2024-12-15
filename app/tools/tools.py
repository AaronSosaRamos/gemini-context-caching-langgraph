from langchain_community.tools.tavily_search import TavilySearchResults

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# List of tools
tools = [TavilySearchResults(max_results=5)]