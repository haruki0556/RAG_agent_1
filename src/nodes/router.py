from langgraph.types import Command
from src.state import GraphState


def router(state:GraphState) -> Command:
    """ルーターを実行する"""
    user_input = state["user_input"]
    if "要約" in user_input.lower():
        return Command(
            update={
                "task":"要約です"
            },
            goto=["summarization_retrieve"]
        )
    else:
        return Command(
            update={
                "task":"検索です"
            },
            goto=["generate_queries"]
        )
