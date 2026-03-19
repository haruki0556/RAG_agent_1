from langgraph.types import Command
from src.state import GraphState
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from config import CHUNK_SIZE,CHUNK_OVERLAP

def summary_load_chunks(state:GraphState) -> Command:

    summary_doc = state["summary_doc"]

    loader = PyPDFLoader(summary_doc)
    pages = loader.load()

    python_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,chunk_overlap=CHUNK_OVERLAP)
    chunks = python_splitter.split_documents(pages)#chunksはDocument型のリスト
    for chunk in chunks:
        chunk.metadata["source"] = summary_doc

    search_results = [(chunk.page_content,chunk.metadata["source"]) for chunk in chunks]#search_resultsはtuple[str,str]のリスト

    return Command(
        update={"search_results":search_results},
        goto="summary_map"
    )

 

