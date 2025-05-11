import os
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader, JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    huggingfacehub_api_token=token
)
def load_and_embed(csv_path: str, persist_directory: str):
    if csv_path.endswith(".csv"):
        loader = CSVLoader(file_path=csv_path)
    elif csv_path.endswith(".json"):
        loader = JSONLoader(file_path=csv_path, jq_schema=".", text_content=False)
    else:
        raise ValueError("Unsupported file format. Only .csv and .json are supported.")

    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    db = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    db.persist()
    print(f"✅ Embedding and saving completed: {persist_directory}")
# Essay retriever
essay_retriever = Chroma(
    persist_directory="essay_db",
    embedding_function=embedding_model
).as_retriever()

# Fitness retriever
fitness_retriever = Chroma(
    persist_directory="fitness_db",
    embedding_function=embedding_model
).as_retriever()

# Social Media retriever
social_retriever = Chroma(
    persist_directory="social_db",
    embedding_function=embedding_model
).as_retriever()