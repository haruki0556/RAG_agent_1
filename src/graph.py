from src.state import GraphState
from typing import Annotated
from typing import TypedDict
from functools import partial

from langgraph.graph import StateGraph,START,END
import operator
from langgraph.types import Command
from src.nodes.query import generate_queries
from src.nodes.retrieve import vector_search
from src.nodes.generate import generate_answer
from src.nodes.feedback import feedback




def create_rag_graph(vectorStore):
    """RAGエージェントのグラフを作成する"""

    graph = StateGraph(GraphState)

    vector_search_with_store = partial(vector_search,vectorStore=vectorStore)

    graph.add_node("generate_queries",generate_queries)
    graph.add_node("vector_search",vector_search_with_store)
    graph.add_node("generate_answer",generate_answer)
    graph.add_node("feedback",feedback)

    graph.add_edge(START,"generate_queries")
    graph.add_edge("vector_search","generate_answer")


    return graph.compile()