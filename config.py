def get_initial_state(user_input:str):
    return {
        "user_input":user_input,
        "iteration":0
    }

FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL_NAME = "models/gemini-embedding-001"
MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0.0
MAX_ITERATIONS = 3 #ループ回数の最大値
MAX_SEARCH_RESULTS = 3 #検索結果の最大値
MIN_SCORE = 0.5 #評価スコアの最小値
LOOP_THRESHOLD = 0.5 #ループ判定の閾値
CHUNK_SIZE = 1000 #チャンクサイズ
CHUNK_OVERLAP = 200 #チャンクオーバーラップ
MAX_LOOP_COUNT = 3 #ループ回数の最大値
SEARCH_K = 3 #検索結果の最大値
FILE_NAME = "data/rakusu_brexa.pdf"