from langchain.agents import Tool
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub


def get_social_tool():
    retriever = Chroma(
        persist_directory="vectorstores/socialmedia",
        embedding_function=HuggingFaceEmbeddings()
    ).as_retriever()

    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        model_kwargs={"temperature": 0.7}
    )

    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    return Tool(
        name="Social Media Generator",
        func=chain.run,
        description="Generate engaging social media posts"
    )