from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from shared_llm import model, tokenizer

def run_social_tool(prompt: str) -> str:
    retriever = Chroma(
        persist_directory="vectorstores/socialmedia",
        embedding_function=HuggingFaceEmbeddings()
    ).as_retriever()

    docs = retriever.get_relevant_documents(prompt)
    context = "\n".join([doc.page_content for doc in docs])

    chat_messages = [
        {"role": "system", "content": "You are a social media expert crafting engaging and trendy posts."},
        {"role": "user", "content": f"Context:\n{context}\n\nUser Prompt:\n{prompt}"}
    ]

    prompt_text = tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.5,
        top_k=50,
        top_p=0.95
    )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

    return response.strip()