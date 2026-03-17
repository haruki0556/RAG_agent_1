from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import Command
from typing import Literal
from typing_extensions import TypedDict
from src.state import GraphState
from config import MODEL_NAME,TEMPERATURE
from config import K_SUMMARY


def summarization_retrieve(state:GraphState,vectorStore) -> Command:
    """要約用のチャンクを取得する"""

    user_input = state["user_input"]
    search_results_with_scores = vectorStore.similarity_search_with_score(user_input,k=K_SUMMARY)
    search_results = [(doc.page_content,doc.metadata["source"]) for doc, _ in search_results_with_scores]
    return Command(
        update={
            "search_results":search_results
        },
        goto="summarize"
    )
