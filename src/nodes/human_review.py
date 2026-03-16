



def human_review(state:GraphState) -> Command:
    """人間による回答のレビュー"""
    answer = state["answer"]
    return Command(
        update={
            "answer":answer
        },
        goto=END
    )