from langgraph.types import Command
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage,HumanMessage
from config import MODEL_NAME,TEMPERATURE
from src.state import GraphState
from config import MAP_CHUNK_GROUP_SIZE,MAP_MAX_GROUPS
from langgraph.graph import END

def summary_reduce(state:GraphState) -> Command:
    partial_summaries = state["partial_summaries"]
    user_input = state["user_input"]

    context = "\n\n".join([f"要約{i+1}:\n{summary}" for i, summary in enumerate(partial_summaries)])

    messages = [
        SystemMessage(content=
        """# あなたの役割
        あなたは要約を行うエージェントです。

        ## タスク
        - 与えられた文書を、簡潔に要約してください。
        - その際、文書の構成、主張、根拠、結論という項目に分けて回答してください。
        - 各項目はそれぞれ100字以内に収めてください。
        - 結論の後に、短い総括を100字以内でまとめてください。
        - 総括の後に、出典を[要約元：ファイル名]として記載してください。

        ## 要約の要件
        - 構成、主張、根拠、結論、総括の項目を守ってください。
        - 文書以外の情報は使用しないでください。
        - 内容の重複は統合してください。
        """),
        HumanMessage(content=f"""# 要約対象
        文書: {context}
        """),
    ]

    model = ChatGoogleGenerativeAI(model=MODEL_NAME,temperature=TEMPERATURE)

    response:str = model.invoke(messages).content
    print(f"\n最終要約:")
    print(response)

    return Command(
        update={"answer":response},
        goto=END
    )
    