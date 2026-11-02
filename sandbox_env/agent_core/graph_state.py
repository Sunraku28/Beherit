from langchain_core.prompts import message
from typing import TypeDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage

class AgentState(BaseMessage):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_url: str
    html_context: str
    next_node: str


