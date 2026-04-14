import os
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
    pairs = state.get("search_results",[])

    return {
        "output":state.get("answer",""),
        "context":[doc for doc, _ in pairs],
        }

from langsmith.schemas import Run, Example

#完全一致かどうかの判定
def exact_match(run: Run, example: Example):
    pred = (run.outputs or {}).get("output", "")
    reference = (example.outputs or {}).get("answer", "")
    return {"key": "exact_match", "score": (pred or "").strip() == reference.strip()}

class QAEvalOutput(BaseModel):
    faithfulness_score:float = Field(description="忠実性の採点スコア")
    relevance_score:float = Field(description="関連性の採点スコア")
    answer_correctness_score:float = Field(description="回答正確性の採点スコア")
    context_recall_score:float = Field(description="文脈再現率の採点スコア")
    faithfulness_reason:str = Field(description="忠実性の採点理由（短く日本語で記述）")
    relevance_reason:str = Field(description="関連性の採点理由（短く日本語で記述）")
    answer_correctness_reason:str = Field(description="回答正確性の採点理由（短く日本語で記述）")
    context_recall_reason:str = Field(description="文脈再現率の採点理由（短く日本語で記述）")

def qa_llm_evaluator(run: Run, example: Example):
    print(run.inputs,run.outputs,example.inputs,example.outputs)
    user_input = (run.inputs or {}).get("user_input", "")
    pred = (run.outputs or {}).get("output", "")
    context = (run.outputs or {}).get("context", [])
    reference = (example.outputs or {}).get("answer", "")
    messages = [
        SystemMessage(content=
        f"""# あなたの役割
        あなたは予測回答の品質を評価するエージェントです。
        RAGASの評価基準にしたがって、以下の4つの評価指標を使用しそれぞれの観点から採点してください。
        また、採点の理由を日本語で簡潔に(2文以下で)記述してください。
        採点は、100点満点のスコアで行い、減点方式で採点したのち、0.00~1.00のスコアに正規化してください。

        【評価指標】
        ## 1. Faithfulness(忠実性)：予測回答が、取得したコンテキストにどれだけ正しく基づいているかを評価する。
        - 予測回答を個別のステートメントに分解し、コンテキストに裏付けされていないステートメントの数×10点を減点する。
        

        ## 2. Relevance(関連性)
        予測回答が質問にどれだけ適切に答えているかを評価する。
        予測回答のみから質問が再現できるほど高スコア。
        - 質問と回答がずれていると判断された場合、50点を減点する。
        - 余計な情報と判断されるものは、1文あたり20点を減点する。


        ## 3. Answer Correctness(回答正確性)
        予測回答と参照回答が意味的にどれだけ一致しているかを評価。
        一致していれば高スコア。
        -参照回答に含まれているステートメントの内、予測回答に含まれていないステートメントの数×10点を減点する。
        -予測回答に含まれているステートメントの内、参照回答に含まれていないステートメントの数×10点を減点する。


        ## 4. Context Recall(文脈再現率)
        取得したコンテキストが、質問に必要な情報をどれだけ含んでいるかを評価。
        必要な情報が抜けているほど低スコア。必要な情報をもれなく取得できていれば高スコア。
        - 参照回答を個別のステートメントに分解し、取得コンテキストに裏付けされていないステートメントの数×10点を減点する。
        """),
        HumanMessage(content=f"元の質問: {user_input}\n参照回答: {reference}\n予測回答: {pred}\nコンテキスト: {context}")
    ]
    model = ChatGoogleGenerativeAI(model=MODEL_NAME,temperature=TEMPERATURE)
    evaluation:QAEvalOutput = model.with_structured_output(QAEvalOutput).invoke(messages)
    return [{"key": "faithfulness_score", "score": evaluation.faithfulness_score, "comment": evaluation.faithfulness_reason}
     ,{"key": "relevance_score", "score": evaluation.relevance_score, "comment": evaluation.relevance_reason}
     ,{"key": "answer_correctness_score", "score": evaluation.answer_correctness_score, "comment": evaluation.answer_correctness_reason}
     ,{"key": "context_recall_score", "score": evaluation.context_recall_score, "comment": evaluation.context_recall_reason}]
# 2. テスト実行（これが LangSmith の Tests タブに反映されます）
result = evaluate(
    predict,
    evaluators=[qa_llm_evaluator],
    data=dataset_name,
    description="Rag-agent1の評価",
    max_concurrency=0,
)
