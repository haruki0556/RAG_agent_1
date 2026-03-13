from dotenv import load_dotenv

load_dotenv()

from config import get_initial_state,FAISS_INDEX_PATH
from src.graph import create_rag_graph
from src.retriever.store import load_vectorStore,build_vectorStore
import os
import chainlit as cl

@cl.on_chat_start
async def on_chat_start():
    if os.path.exists(FAISS_INDEX_PATH):
        vectorStore = load_vectorStore()
    else:
        vectorStore = build_vectorStore()
    graph = create_rag_graph(vectorStore)
    cl.user_session.set("graph",graph)
    await cl.Message(content="RAGエージェントが起動しました。").send()

@cl.on_message
async def main(message:cl.Message):
    user_input = message.content
    graph = cl.user_session.get("graph")
    state = graph.invoke(get_initial_state(user_input))
    await cl.Message(content=state.get("answer","最大反復回数に到達しました。")).send()




