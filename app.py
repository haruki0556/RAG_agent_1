from dotenv import load_dotenv

load_dotenv()

from config import get_initial_state,FAISS_INDEX_PATH
from src.graph import create_rag_graph
from src.retriever.store import load_vectorStore,build_vectorStore
import os

user_input = "ああああああああああああああああああああああああああああああああああああ"

if __name__ == "__main__":
    if os.path.exists(FAISS_INDEX_PATH):
        vectorStore = load_vectorStore()
    else:
        vectorStore = build_vectorStore()
    graph = create_rag_graph(vectorStore)
    graph.invoke(get_initial_state(user_input))
