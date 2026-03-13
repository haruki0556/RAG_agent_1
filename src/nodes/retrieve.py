from langgraph.types import Command
from langchain_core.documents import Document
from config import SEARCH_K
from src.state import GraphState


def vector_search(query:str,vectorStore) -> Command:
    """ベクトル検索を行う"""
    

    print(f"\n検索実行: {query}")

    search_results_with_scores = vectorStore.similarity_search_with_score(query,k=SEARCH_K)

    ranking = [(doc.page_content,score) for doc, score in search_results_with_scores]
    #search_results_with_scoresには、list[taple[Document,float]]が入っている。
    #doc.page_contentのみが欲しいため、_でscoreを受け取っている。

    print(f" 取得:{len(ranking)}件のチャンク")

    return Command(
        update={"ranking":ranking},
        goto="ranking"
    )