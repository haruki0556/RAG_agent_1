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
from src.nodes.ranking import ranking
from src.nodes.summarization_retrieve import summarization_retrieve
from src.nodes.summarize import summarize
from src.nodes.router import router
from src.nodes.summary_select_doc import summary_select_doc
from src.nodes.summary_load_chunks import summary_load_chunks
from src.nodes.summary_map import summary_map
from src.nodes.summary_reduce import summary_reduce




def create_rag_graph(vectorStore):
    """RAGエージェントのグラフを作成する"""

    graph = StateGraph(GraphState)

    vector_search_with_store = partial(vector_search,vectorStore=vectorStore)
    summarization_retrieve_with_store = partial(summarization_retrieve,vectorStore=vectorStore)

    graph.add_node("generate_queries",generate_queries)
    graph.add_node("vector_search",vector_search_with_store)
    graph.add_node("ranking",ranking)
    graph.add_node("generate_answer",generate_answer)
    graph.add_node("feedback",feedback)
    #graph.add_node("summarize",summarize)
    #graph.add_node("summarization_retrieve",summarization_retrieve_with_store)
    graph.add_node("router",router)
    graph.add_node("summary_select_doc",summary_select_doc)
    graph.add_node("summary_load_chunks",summary_load_chunks)
    graph.add_node("summary_map",summary_map)
    graph.add_node("summary_reduce",summary_reduce)

    graph.add_edge(START,"router")

    graph.add_edge("vector_search","ranking")


    return graph.compile()