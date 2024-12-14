from dotenv import load_dotenv, find_dotenv
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from app.agent.agent_state import AgentState
from app.agent.nodes.nodes import (
    call_model, 
    should_continue
)
from app.tools.tools import tools
from langgraph.prebuilt import ToolNode

load_dotenv(find_dotenv())

class GraphConfig(TypedDict):
    model_name: Literal["openai", "gemini"]

workflow = StateGraph(AgentState, config_schema=GraphConfig)
workflow.add_node("agent", call_model)
workflow.add_node("action", ToolNode(tools))
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)
workflow.add_edge("action", "agent")
graph = workflow.compile()
