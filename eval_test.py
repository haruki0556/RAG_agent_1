import os
from langsmith import Client
from dotenv import load_dotenv
from langsmith import evaluate

load_dotenv() # .env から API_KEY などを読み込み

from config import get_initial_state,FAISS_INDEX_PATH
from src.graph import create_rag_graph
from src.retriever.store import load_vectorStore,build_vectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage,HumanMessage
from pydantic import BaseModel,Field
from config import MODEL_NAME,TEMPERATURE


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
    reference = (example.outputs or {}).get("answer", "")
    return {"key": "exact_match", "score": (pred or "").strip() == reference.strip()}

class QAEvalOutput(BaseModel):
    score:float = Field(description="参照回答と意味的に一致しているほど高い")
    reason:str = Field(description="採点理由（短く日本語で記述）")

def qa_llm_evaluator(run: Run, example: Example):
    print(run.inputs,run.outputs,example.inputs,example.outputs)
    pred = (run.outputs or {}).get("output", "")
    reference = (example.outputs or {}).get("answer", "")
    messages = [
        SystemMessage(content=
        f"""# あなたの役割
        あなたは回答の品質を評価するエージェントです。
        予測回答が参照回答と意味的にどれだけ一致しているかを0.0~1.0のスコアで採点してください。
        - 表現の違いは許容する。
        - 事実の誤り、重要な欠落、参照回答と矛盾する主張は低スコア
        - 参照回答と同等の内容が予測回答に含まれていれば高スコア
        """),
        HumanMessage(content=f"参照回答: {reference}\n予測回答: {pred}")
    ]
    model = ChatGoogleGenerativeAI(model=MODEL_NAME,temperature=TEMPERATURE)
    evaluation:QAEvalOutput = model.with_structured_output(QAEvalOutput).invoke(messages)
    return {"key": "qa_llm_evaluator", "score": evaluation.score, "comment": evaluation.reason}
# 2. テスト実行（これが LangSmith の Tests タブに反映されます）
result = evaluate(
    predict,
    evaluators=[qa_llm_evaluator],
    data=dataset_name,
    description="Rag-agent1の評価",
    max_concurrency=0,
)
