from typing import Annotated
from typing import TypedDict,Literal
import operator

class GraphState(TypedDict):
    user_input: str
    queries: list[str]
    search_results:list[tuple[str,str]]
    answer:str
    feedback:str
    iteration:int
    ranking:Annotated[list[tuple[str,str,float]],operator.add]#反復処理した結果をすべて保持する
    task:Literal["検索です","要約です"]
    partial_summaries:list[str]#Mapの部分要約結果を貯める
    summary_doc:str
