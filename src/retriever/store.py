from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from config import EMBEDDING_MODEL_NAME,FAISS_INDEX_PATH,CHUNK_SIZE,CHUNK_OVERLAP,FILE_NAMES

def build_vectorStore():#初回起動時、文書差し替え時に実行

    filenames = FILE_NAMES
    pages = []
    for filename in filenames:
        loader = PyPDFLoader(filename)
        data = loader.load()

        for page in data:
            page.metadata["source"] = filename

        pages.extend(data)

    python_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,chunk_overlap=CHUNK_OVERLAP)
    chunks = python_splitter.split_documents(pages)

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)

    vectorStore = FAISS.from_documents(chunks,embeddings)

    vectorStore.save_local(FAISS_INDEX_PATH)

    return vectorStore

def load_vectorStore():#以降の起動時に実行
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    vectorStore = FAISS.load_local(FAISS_INDEX_PATH,embeddings,allow_dangerous_deserialization=True)
    return vectorStore