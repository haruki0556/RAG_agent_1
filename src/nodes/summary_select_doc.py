from langgraph.types import Command

from src.state import GraphState

def summary_select_doc(state:GraphState) -> Command:

    user_input = state["user_input"]
    
    if "Tokyo_univ_brain.pdf" in user_input:
        summary_doc = "data/Tokyo_univ_brain.pdf"
    elif "dejihari.pdf" in user_input:
        summary_doc = "data/dejihari.pdf"
    elif "critical_thinking.pdf" in user_input:
        summary_doc = "data/critical_thinking.pdf"
    else:
        summary_doc = "data/Tokyo_univ_brain.pdf"

    return Command(
        update={"summary_doc":summary_doc},
        goto="summary_load_chunks"
    )


    
