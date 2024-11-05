from langchain_community.vectorstores import FAISS 
from langchain.chains import RetrievalQA
from src.llm_connection import get_llm
from src.prompt_templet import get_prompt
from src.vector_database_loader import vector_database,get_embedding

file = r"C:\Users\olivi\OneDrive\Documents\Undergrad. Research\ArgoBot\Custom-Question-Answering-Chatbot-using-Langchain-and-Gemini-AI\backend\vector_database_file"

def get_chain():
    
    llm = get_llm()
    embeddings = get_embedding()
    vectordb = FAISS.load_local(file,embeddings,allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever()
    prompt = get_prompt()
    
    chain = RetrievalQA.from_chain_type(
        
        llm = llm,
        chain_type = "stuff",
        retriever = retriever,
        input_key = "query",
        return_source_documents = True,
        chain_type_kwargs = {"prompt":prompt}
    
    )
    
    return chain
