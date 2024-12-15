from typing import Optional, TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

class AgentState(TypedDict):
    init_decision: Optional[bool]
    decision: Optional[str]

    text: Optional[str]
    cache_name: Optional[str]
    ttl: Optional[int]
    query: Optional[str]

    messages: Annotated[Sequence[BaseMessage], add_messages]

    response: Optional[str]
    update_or_delete: Optional[str]