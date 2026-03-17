from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage,HumanMessage
from langgraph.types import Command
from src.state import GraphState
from langgraph.graph import END
from config import MODEL_NAME,TEMPERATURE

def summarize(state:GraphState) -> Command:
    """要約を行う"""
    search_results = state["search_results"]
    user_input = state["user_input"]

    context = "\n\n".join([f"文書{i+1}:\n{doc}\n出典:{source}" for i, (doc, source) in enumerate(search_results)])

    messages = [
        SystemMessage(content=
        """# あなたの役割
        あなたは要約を行うエージェントです。

        ## タスク
        以下の文書を要約してください。

        ## 要約の要件
        - 与えられたテキストの要点を、構成を保ちつつ簡潔に要約してください。
        - 与えられたテキスト以外の情報は使用しないでください。
        - 文末に、要約に利用した文書のURLを、[要約元:文書のURL]という形式で明記してください。"""
        ),
        HumanMessage(content=f"""# タスク
        以下の文書を要約してください。

        ユーザーの質問: {user_input}
        文書: {context}
        """),

    ]

    model = ChatGoogleGenerativeAI(model=MODEL_NAME,temperature=TEMPERATURE)
    response:str = model.invoke(messages).content

    print(f"\n要約:")
    print(response)

    return Command(
        update={
            "answer":response
        },
        goto=END
    )