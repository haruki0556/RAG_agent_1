from config import N
from src.state import GraphState
from langgraph.types import Command,Send



def ranking(state:GraphState) -> Command:
    """ランキングを行う"""

    ranking = state["ranking"]
    ranked = sorted(ranking,key=lambda x:x[1],reverse=False)
    top_n = ranked[:N]
    search_results = [doc for doc, _ in top_n]
    print(f"ランキングで上位{N}件に絞れているか確認: {len(search_results)}件")
    return Command(
        update={
            "search_results":search_results
        },
        goto="generate_answer"
    )