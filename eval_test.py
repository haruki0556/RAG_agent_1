import os
from langsmith import Client
from dotenv import load_dotenv
from langsmith import evaluate

load_dotenv() # .env から API_KEY などを読み込み

from config import get_initial_state,FAISS_INDEX_PATH
from src.graph import create_rag_graph
from src.retriever.store import load_vectorStore,build_vectorStore

dataset_name = "Rag-agent1"

if os.path.exists(FAISS_INDEX_PATH):
    vectorStore = load_vectorStore()
else:
    vectorStore = build_vectorStore()

graph = create_rag_graph(vectorStore)

# 1. 評価したい自分のエージェント関数
def predict(inputs: dict) -> dict:
    print(inputs.keys())
    user_input = inputs["user_input"]
    state = graph.invoke(get_initial_state(user_input))

    return {"output":state.get("answer","")}

from langsmith.schemas import Run, Example

#完全一致かどうかの判定
def exact_match(run: Run, example: Example):
    pred = (run.outputs or {}).get("output", "")
    expected = (example.outputs or {}).get("output", "")
    return {"key": "exact_match", "score": (pred or "").strip() == expected.strip()}

# 2. テスト実行（これが LangSmith の Tests タブに反映されます）
result = evaluate(
    predict,
    evaluators=[exact_match],
    data=dataset_name,
    description="Rag-agent1の評価",
    max_concurrency=0,
)
