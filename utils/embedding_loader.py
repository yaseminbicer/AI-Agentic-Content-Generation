import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    huggingfacehub_api_token=token
)

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

# LangChain tools and agent setup
from langchain.agents import Tool, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# LLM tanımı (örnek olarak OpenAI, Hugging Face yerine değiştirebilirsin)
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Tool tanımları
essay_tool = Tool(
    name="Essay Generator",
    func=RetrievalQA.from_chain_type(llm=llm, retriever=essay_retriever),
    description="Generate academic essays based on user prompts."
)

fitness_tool = Tool(
    name="Fitness Plan Generator",
    func=RetrievalQA.from_chain_type(llm=llm, retriever=fitness_retriever),
    description="Generate personalized fitness programs."
)

social_tool = Tool(
    name="Social Media Content Generator",
    func=RetrievalQA.from_chain_type(llm=llm, retriever=social_retriever),
    description="Create social media posts tailored to user input."
)

# Agent oluştur
agent = initialize_agent(
    tools=[essay_tool, fitness_tool, social_tool],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)