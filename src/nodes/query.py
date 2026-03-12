from pydantic import BaseModel,Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage,HumanMessage
from langgraph.types import Command,Send
from typing import Literal
from typing_extensions import TypedDict
from src.state import GraphState
from config import MODEL_NAME,TEMPERATURE


def generate_queries(state:GraphState)->Command:
    """ユーザーの入力から3つのクエリを生成する"""

    class QueryGenarationOutput(BaseModel):
        queries:list[str] = Field(description="3つの異なるクエリ")

    user_input = state["user_input"]
    feedback = state.get("feedback","")
    previous_queries = state.get("queries",[])
    previous_search_results = state.get("search_results",[])

    if feedback and previous_queries:
        previous_context = "\n\n".join([f"文書{i+1}:\n{doc}" for i,doc in enumerate(previous_search_results)])
        """
        文書1:○○
        （空行）
        文書2:○○
        (略)、、、みたいになる
        """
        user_message = f""" # タスク
        前回の検索クエリで得られた検索結果が不十分でした、
        フィードバックを踏まえて、より効果的な検索クエリを生成してください。

        ユーザーの質問: {user_input}
        前回の検索クエリ: {','.join(previous_queries)}
        前回の検索結果: {previous_context}
        フィードバック: {feedback}

        上記のフィードバックを踏まえて、ユーザーの質問により適切に回答できる3つの異なる検索クエリを生成してください。"""
        
        
    else:
        user_message = f""" # タスク
        以下のユーザーに質問に対して、3つの異なる検索クエリを生成してください。

        ユーザーの質問: {user_input}"""

    messages = [
        SystemMessage(content=
    """# あなたの役割
    あなたはRAGシステムのクエリ生成エージェントです。

    ## タスク
    ユーザーの質問に対して、効果的な情報検索を行うための3つの異なる検索クエリを生成してください。

    ## 検索クエリの生成要件
    検索クエリは以下の観点で多様性を持たせてください。
    - 質問の異なる側面や解釈
    - 具体的なキーワードと抽象的な概念
    """),
        HumanMessage(content=user_message),
    ]

    model = ChatGoogleGenerativeAI(model=MODEL_NAME,temperature=TEMPERATURE)

    response:QueryGenarationOutput = model.with_structured_output(QueryGenarationOutput).invoke(messages)

    iteration = state.get("iteration",0)

    print(f"\n{iteration+1}回目の検索クエリ生成:")
    print(f"生成されたクエリ：")
    for i,query in enumerate(response.queries,start=1):
        print(f"{i}. {query}")

    return Command(
        update={
            "queries":response.queries,
        },
        goto=[
            Send("vector_search",query)
            for query in response.queries
        ]#"vector_search"ノードにクエリを送信
    )