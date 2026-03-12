from typing import Annotated
from typing import TypedDict
import operator

class GraphState(TypedDict):
    user_input: str
    queries: list[str]
    search_results:Annotated[list[str],operator.add]#反復処理した結果をすべて保持する
    answer:str
    feedback:str
    iteration:int
