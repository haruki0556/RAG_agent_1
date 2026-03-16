from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage,HumanMessage
from langgraph.types import Command
from src.state import GraphState
from config import MODEL_NAME,TEMPERATURE

def generate_answer(state:GraphState) -> Command:
    """検索結果を統合して回答を生成する"""
    search_results = state["search_results"]
    context = "\n\n".join([f"文書{i+1}:\n{doc}\n出典:{source}" for i, (doc, source) in enumerate(search_results)])
    user_input = state["user_input"]
    feedback = state.get("feedback","")
    previous_answer = state.get("answer","")

    if feedback and previous_answer:
        user_message = f""" # タスク
        前回の回答が不十分でした、
        フィードバックを踏まえて、より適切な回答を生成してください。

        ユーザーの質問: {user_input}
        検索結果：{context}
        前回の回答: {previous_answer}
        フィードバック: {feedback}

        上記のフィードバックを踏まえて、ユーザーの質問により適切に回答を生成してください。"""
    else:
        user_message = f""" # タスク
        以下の検索結果を参考に、ユーザーの質問に対して回答を生成してください。

        ユーザーの質問: {user_input}
        検索結果：{context}
        """

    messages = [
        SystemMessage(content=
        """# あなたの役割
        あなたはRAGシステムの回答生成エージェントです。
        提供された検索結果をもとに、ユーザーの質問に対して正確で有用な回答をしてください。
        また、複数の文書から情報を統合して、包括的な回答を提供してください。
        また、回答には必ず出典を明記してください。

        ## 回答生成要件
        - 検索結果の情報のみを使用してください。
        - 検索結果に情報がない場合は、その旨を明確に伝えてください。
        - 複数の文書から情報を統合して、包括的な回答を提供してください。
        - 簡潔かつ明確に回答してください。
        - 回答テキストの文末には、[出典：ファイル名]をつけてください。
        """),
        HumanMessage(content=user_message)
    ]

    model = ChatGoogleGenerativeAI(model=MODEL_NAME,temperature=TEMPERATURE)

    response:str = model.invoke(messages).content

    print(f"\n回答生成:")
    print(response)
    

    return Command(
        update={"answer":response},
        goto="feedback"
    )