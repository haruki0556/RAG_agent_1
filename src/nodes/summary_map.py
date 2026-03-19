from langgraph.types import Command
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage,HumanMessage
from config import MODEL_NAME,TEMPERATURE
from src.state import GraphState
from config import MAP_CHUNK_GROUP_SIZE,MAP_MAX_GROUPS



def summary_map(state:GraphState) -> Command:
    search_results = state["search_results"]
    user_input = state["user_input"]

    groups = [search_results[i : i + MAP_CHUNK_GROUP_SIZE] for i in range(0, len(search_results), MAP_CHUNK_GROUP_SIZE)]
    group_count = len(groups)
    print(f"\n{group_count}個のグループに分割")
    groups = groups[:MAP_MAX_GROUPS]
    partial_summaries = []
    for i, group in enumerate(groups):
        context = "\n\n".join([f"文書{i+1}:\n{doc}\n出典:{source}" for i, (doc, source) in enumerate(group)])

        messages = [
            SystemMessage(content=
            """# あなたの役割
            あなたは要約を行うエージェントです。

            ## タスク
            与えられた文書を、簡潔に要約してください。

            ## 要約の要件
            - 与えられたテキスト以外の情報は使用しないでください。
            - 項目は、要旨のみで構成してください。
            - 文書の要点を落とさずに要約してください。
            - 300字以内で要約してください。
            """
            ),
            HumanMessage(content=f"""# タスク
            以下の文書を要約してください。
            文書: {context}
            """),
        ]

        model = ChatGoogleGenerativeAI(model=MODEL_NAME,temperature=TEMPERATURE)
        response:str = model.invoke(messages).content


        partial_summary = response
        print(f"\n要約{i+1}:")
        print(partial_summary)
        partial_summaries.append(partial_summary)

    return Command(
        update={"partial_summaries":partial_summaries},
        goto="summary_reduce"
    )

