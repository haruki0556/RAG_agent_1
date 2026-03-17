from pydantic import BaseModel,Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage,HumanMessage
from langgraph.types import Command
from typing import Literal
from typing_extensions import TypedDict
from src.state import GraphState
from config import MODEL_NAME,TEMPERATURE

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE,CHUNK_OVERLAP


def summarization_retrieve(state:GraphState) -> Command:
    """要約用のチャンクを取得する"""

    user_input = state["user_input"]

    class SummarizationRetrieveOutput(BaseModel):
        document_url:Literal["data/critical_thinking.pdf","data/dejihari.pdf","data/Tokyo_univ_brain.pdf"] = Field(description="要約する文書のURL")

    messages = [
        SystemMessage(content=
        """# あなたの役割
        あなたは要約する文書を選択するエージェントです。

        ## タスク
        ユーザーの質問に対して、適切な要約を行うためにはどの文書を要約すべきかを選択してください。

        ## 要約用チャンクの選択要件
        - ユーザーの質問に対して、適切な要約を行うためにはどの文書を要約すべきかを選択してください。
        - 選択した文書は、要約用のチャンクとして使用されます。
        - 必ず1つの文書を選択してください。
        """),
        HumanMessage(content=f""" # タスク
        以下のユーザーの質問から、適切な要約を行うためにはどの文書を要約すべきかを選択してください。
        必ず一つの文書を選択してください。

        ユーザーの質問: {user_input}
        """)
    ]

    model = ChatGoogleGenerativeAI(model=MODEL_NAME,temperature=TEMPERATURE)
    response:SummarizationRetrieveOutput = model.with_structured_output(SummarizationRetrieveOutput).invoke(messages)

    print(f"\n要約用チャンクの選択:")
    print(f"選択された文書: {response.document_url}")

    loader = PyPDFLoader(response.document_url)
    data = loader.load()
    python_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,chunk_overlap=CHUNK_OVERLAP)
    chunks = python_splitter.split_documents(data)

    search_results = [(chunk.page_content,chunk.metadata["source"]) for chunk in chunks]

    return Command(
        update={
            "search_results":search_results
        },
        goto=["summarize"]
    )
  

