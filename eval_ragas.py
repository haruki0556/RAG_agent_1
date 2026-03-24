"""
LangSmith の evaluate + RAGAS（レガシー API）で Faithfulness 等 4 指標を記録するスクリプト。

実行: python eval_ragas.py
前提: .env に GOOGLE_API_KEY / GEMINI_API_KEY と LangSmith 用のキー
"""

from __future__ import annotations

import math
import os
import warnings

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langsmith import evaluate
from langsmith.schemas import Example, Run

from ragas import SingleTurnSample
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from ragas.run_config import RunConfig

from config import (
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    MODEL_NAME,
    TEMPERATURE,
    get_initial_state,
)
from src.graph import create_rag_graph
from src.retriever.store import build_vectorStore, load_vectorStore

warnings.filterwarnings(
    "ignore",
    message=".*Importing .* from 'ragas.metrics' is deprecated.*",
    category=DeprecationWarning,
)

load_dotenv()

DATASET_NAME = "Rag-agent1"

# ---------------------------------------------------------------------------
# ベクトルストア & グラフ
# ---------------------------------------------------------------------------
if os.path.exists(FAISS_INDEX_PATH):
    vectorStore = load_vectorStore()
else:
    vectorStore = build_vectorStore()

graph = create_rag_graph(vectorStore)

# ---------------------------------------------------------------------------
# RAGAS 用: LangChain モデルをラップ
# ---------------------------------------------------------------------------
lc_chat = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=TEMPERATURE)
lc_embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)

evaluator_llm = LangchainLLMWrapper(lc_chat)
evaluator_embeddings = LangchainEmbeddingsWrapper(lc_embeddings)

run_config = RunConfig()

faithfulness_m = Faithfulness(llm=evaluator_llm)
context_recall_m = ContextRecall(llm=evaluator_llm)
context_precision_m = ContextPrecision(llm=evaluator_llm)
answer_relevancy_m = AnswerRelevancy(
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
)

for _metric in (
    faithfulness_m,
    context_recall_m,
    context_precision_m,
    answer_relevancy_m,
):
    _metric.init(run_config)


# ---------------------------------------------------------------------------
# predict: LangSmith が各行で呼ぶ。outputs が evaluator の run に載る。
# ---------------------------------------------------------------------------
def predict(inputs: dict) -> dict:
    user_input = inputs["user_input"]
    state = graph.invoke(get_initial_state(user_input))
    answer = state.get("answer") or ""
    search_results = state.get("search_results") or []
    contexts = [doc for doc, _src in search_results]
    return {"output": answer, "contexts": contexts}


def build_sample(run: Run, example: Example) -> SingleTurnSample:
    """質問・予測・検索チャンク・参照回答を SingleTurnSample にまとめる。"""
    inputs = run.inputs or example.inputs or {}
    user_input = inputs.get("user_input") or ""

    outs = run.outputs or {}
    response = outs.get("output") or ""
    retrieved_contexts = list(outs.get("contexts") or [])

    reference = (example.outputs or {}).get("answer") or ""

    return SingleTurnSample(
        user_input=user_input,
        response=response,
        retrieved_contexts=retrieved_contexts,
        reference=reference,
    )


def _safe_score(metric, sample: SingleTurnSample, key: str) -> dict:
    try:
        score = metric.single_turn_score(sample)
        if score is None or (isinstance(score, float) and math.isnan(score)):
            return {"key": key, "score": None, "comment": "NaN or None"}
        return {"key": key, "score": float(score)}
    except Exception as e:  # noqa: BLE001
        return {"key": key, "score": None, "comment": f"{type(e).__name__}: {e}"}


def faithfulness_evaluator(run: Run, example: Example) -> dict:
    return _safe_score(faithfulness_m, build_sample(run, example), "faithfulness")


def context_recall_evaluator(run: Run, example: Example) -> dict:
    return _safe_score(context_recall_m, build_sample(run, example), "context_recall")


def context_precision_evaluator(run: Run, example: Example) -> dict:
    return _safe_score(context_precision_m, build_sample(run, example), "context_precision")


def answer_relevancy_evaluator(run: Run, example: Example) -> dict:
    return _safe_score(answer_relevancy_m, build_sample(run, example), "answer_relevancy")


if __name__ == "__main__":
    result = evaluate(
        predict,
        evaluators=[
            faithfulness_evaluator,
            context_recall_evaluator,
            context_precision_evaluator,
            answer_relevancy_evaluator,
        ],
        data=DATASET_NAME,
        description="RAGAS: faithfulness, context_recall, context_precision, answer_relevancy",
        max_concurrency=0,
    )
    print(result)
