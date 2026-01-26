from typing import Annotated, TypedDict, List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
import operator

class NegotiationState(TypedDict):
    # Trajectory (STM)
    messages: Annotated[List[BaseMessage], add_messages] 
    # Experience (LTM)
    reflections: Annotated[List[str], operator.add]

    # Initial Settings
    user_role: str
    ai_role:str
    ai_scenario: str
    user_scenario: str
    ai_priority: str
    user_priority: str
    model: str

    # In-Context
    summary: str
    
    # Evaluate
    final_result: str
    buyer_score: int
    seller_score: int
    is_finished: bool

    

    