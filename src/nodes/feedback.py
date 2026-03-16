from pydantic import BaseModel,Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage,HumanMessage
from langgraph.types import Command
from typing import Literal
from typing_extensions import TypedDict
from src.state import GraphState
from langgraph.graph import END
from config import MODEL_NAME,TEMPERATURE

def feedback(state:GraphState) -> Command:
    """回答の品質を評価する"""
    answer = state["answer"]
    user_input = state["user_input"]
    search_results = state["search_results"]
    iteration = state.get("iteration",0)

    if iteration >= 2:#最大反復回数は2回、3回にするとembeddingの上限に引っかかる
        print("\n最大反復回数に達しました。")
        return Command(
            goto=END#終了
        )


    context = "\n\n".join([f"文書{i+1}:\n{doc}\n出典:{source}" for i, (doc, source) in enumerate(search_results)])

    class FeedbackOutput(BaseModel):
        needs_improvement:bool = Field(description="回答の改善が必要かどうか")
        feedback:str = Field(description="改善のための具体的なフィードバック")
        retry_node:Literal["generate_queries","generate_answer"] = Field(
            description="やり直すノード。検索クエリの問題ならgenarate_queries、回答生成の問題ならgenerate_answer")

    messages = [
        SystemMessage(content=
        """# あなたの役割
        あなたはRAGシステムのフィードバックエージェントです。

        ## タスク
        検索結果と生成された回答を参考に、回答の改善が必要かどうかと、改善のための具体的なフィードバックを生成してください。

        ## 評価基準
        - 検索結果に含まれる重要な情報が回答に反映されているかどうか
        - 回答が質問に対して適切に答えているか
        - 検索結果の情報を過不足なく使用しているか

        ## 問題の種類と具体的なフィードバック
        - **検索クエリの問題**:検索結果自体が質問に関連していない、または不十分な場合
        - retry_node: "genarate_queries"
        - feedback: "「~というキーワードで検索すべきだ」「~という観点から検索すべきだ」など具体的な検索戦略を記述"

        - **回答生成の問題**:回答が質問に対して適切に答えていない、または不十分な場合
        - retry_node: "generate_answer"
        - feedback: "「~という情報が抜けている」「~という情報が間違っている」など具体的な改善点を記述"

        ## フィードバックの品質要件
        - 抽象的な指示ではなく、具体的なアクションを示すこと
        - 検索クエリの改善が必要な場合は、具体的なキーワードや検索の観点を示すこと
        - 回答生成の改善が必要な場合は、検索結果のどの情報をどのように使うかを明示すること
        """),
        HumanMessage(content=f"""# 評価対象
        ユーザーの質問: {user_input}
        検索結果: {context}
        回答: {answer}
        """)
    ]

    model = ChatGoogleGenerativeAI(model=MODEL_NAME,temperature=TEMPERATURE)

    evaluation:FeedbackOutput = model.with_structured_output(FeedbackOutput).invoke(messages)

    if evaluation.needs_improvement:
        print("\n回答の改善が必要です。")
        retry_node_jp = "クエリ生成" if evaluation.retry_node == "generate_queries" else "回答生成"
        print(f"次のアクション: {retry_node_jp}に戻って、{evaluation.feedback}をやり直し")
        print(f"改善指示：")
        print(f"{evaluation.feedback}")
    else:
        print("\n回答の改善は不要です。")

    if not evaluation.needs_improvement:
        print(f"完了：{iteration+1}回目の反復で回答が改善されました。")
        return Command(
            update={
                "iteration":iteration+1,
            },
            goto=END)


    return Command(
        update={
            "feedback":evaluation.feedback,
            "iteration":iteration+1
        },
        goto=evaluation.retry_node
    )