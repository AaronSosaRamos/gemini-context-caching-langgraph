from dotenv import load_dotenv, find_dotenv
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from app.agent.agent_state import AgentState
from app.agent.nodes.nodes import (
    call_model,
    create_cache_node,
    decide_tool_or_cache,
    delete_cache_node, 
    update_cache_node,
    use_cache_node
)
from app.tools.tools import tools
from langgraph.prebuilt import ToolNode

load_dotenv(find_dotenv())

class GraphConfig(TypedDict):
    model_name: Literal["openai", "gemini"]

workflow = StateGraph(AgentState, config_schema=GraphConfig)

# Add initial node to call the model
workflow.add_node("agent", call_model)

# Rename the decision node to avoid conflicts
workflow.add_node("decision_node", decide_tool_or_cache)

# Add nodes for the specific actions
workflow.add_node("action", ToolNode(tools))  # For tool-related actions
workflow.add_node("create_cache", create_cache_node)  # For creating a cache
workflow.add_node("update_cache", update_cache_node)  # For updating an existing cache
workflow.add_node("get_cache", use_cache_node)  # For using a cached context
workflow.add_node("delete_cache", delete_cache_node)  # For deleting a cache

# Set entry point of the workflow
workflow.set_entry_point("agent")

# Add edge from the initial model node to the decision-making node
workflow.add_edge("agent", "decision_node")

# Add conditional routing based on the decision made
workflow.add_conditional_edges(
    "decision_node",
    lambda state: state["decision"],  # Determine the decision type
    {
        "tool": "action",  # Route to tool action if decision is 'tool'
        "create_cache": "create_cache",  # Route to create cache if decision is 'create_cache'
        "update_cache": "update_cache",  # Route to update cache if decision is 'update_cache'
        "get_cache": "get_cache",  # Route to use cache if decision is 'get_cache'
        "delete_cache": "delete_cache",  # Route to delete cache if decision is 'delete_cache'
        "none": END
    },
)

# # Add conditional flow for the 'tool' action
# # If continue, return to 'agent'; if end, terminate the workflow
# workflow.add_conditional_edges(
#     "action",
#     should_continue,
#     {
#         "continue": "agent",  # Continue workflow by returning to 'agent'
#         "end": END,  # Terminate the workflow
#     },
# )

workflow.add_edge("action", "agent")
# Ensure all cache-related nodes terminate the workflow after execution
workflow.add_edge("create_cache", END)  # End after creating a cache
workflow.add_edge("update_cache", END)  # End after updating a cache
workflow.add_edge("get_cache", END)  # End after using a cache
workflow.add_edge("delete_cache", END)  # End after deleting a cache

# Compile the workflow
graph = workflow.compile()