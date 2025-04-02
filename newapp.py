import warnings
import os
import google.generativeai as genai
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA, create_retrieval_chain
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import Tool, AgentExecutor, create_react_agent

# Import necessary classes for state management and missing components
from langchain.schema import BaseMessage, HumanMessage, AIMessage  # Added imports
from langchain.memory import ConversationBufferMemory  # Replacing StateGraph
from typing import TypedDict, Sequence, Annotated

warnings.filterwarnings('ignore')

# Set your Google API key here
# os.environ["OPENAI_API_KEY"] = "key"

# Configure the Generative AI with the API key
genai.configure(api_key=os.getenv("OPENAI_API_KEY"))
model = ChatOpenAI(model_name='gpt-4', temperature=0, streaming=True, verbose=True, max_tokens=1024)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# Create the prompt template for RAG
rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

def get_qa_chain(pdf_path):
    # read file
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # split your docs into texts chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # embed the chunks into vectorstore (Chroma)
    vectorstore = Chroma.from_documents(documents=texts, embedding=OpenAIEmbeddings())

    # create retriever and rag chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    question_answer_chain = create_stuff_documents_chain(llm=model, prompt=rag_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

rag_chain = get_qa_chain(r"resources/22_studenthandbook-22-23_f2.pdf")

# Set up the Google Serper API key for search funtionality
# os.environ["SERPER_API_KEY"] = "key"
search = GoogleSerperAPIWrapper()

tools = [
    Tool(
        name="RAG",
        func=lambda x: rag_chain.invoke({"input": x}),
        description="Useful when you're asked Retrieval Augmented Generation (RAG) related questions about UWF."
    ),
    Tool(
        name="Google Search",
        description="For answering questions when you don't know the answer, use Google search to find the answer",
        func=search.run,
    )
]

# Character prompt for guiding the chatbot's responses
character_prompt = """
You are an chatbot specialized in AI advising, you are able to send emails, look at the UWF student handbook and searching informations online.
Answer the following questions as best you can. You have access to the following tools:
{tools}
You shouldn't use tools only if the answer is in the previous conversation history or if the input form the user is conversational like "Hello" or "How are you?"
For any questions requiring tools, you should first search the provided knowledge base, if you don't find any information you have to use tools.
If you don't find relevant information and the question is about UWF use the RAG to retrieve relevant information.
If you didn't find any relevant information then you can use Google search to find related information by informing the user in the final answer.
REMEMBER: you MUST inform the user in the final answer if your answer is based on information that you got with the tool Google Search, and say to the user that the information may be not accurate in this case (don't say up-to-date).
To use a tool, you MUST use the following format:
1. Thought: Do I need to use a tool? Yes
2. Action: the action to take, should be one of [{tool_names}]
3. Action Input: the input to the action
4. Observation: the result of the action
When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the following format:
1. Thought: Do I need to use a tool? No
2. Final Answer: [your response here]
It's very important to always include the 'Thought' before any 'Action' or 'Final Answer'. Ensure your output strictly follows the formats above.
Begin!
Previous conversation history:
{chat_history}
Question: {input}
Thought: {agent_scratchpad}
"""

# Create the agent and its tools
chat_model = model
prompt = PromptTemplate.from_template(character_prompt)
agent = create_react_agent(chat_model, tools, prompt)

# Memory setup for conversation history
memory = ConversationBufferWindowMemory(memory_key='chat_history', k=5, return_messages=True, output_key="output")
agent_chain = AgentExecutor(agent=agent, tools=tools, memory=memory, max_iterations=5, handle_parsing_errors=True, verbose=True)


# Interface function
def create_interface(call_model):
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Streamlit interface with a form
    with st.form(key='question_form', clear_on_submit=True):
        user_input = st.text_input("Enter your question here:")
        
        # Create a submit button inside the form
        submit_button = st.form_submit_button(label='Submit')

        # If the submit button is pressed and user_input is provided
        if submit_button and user_input:
            # Call the model with the user input and chat history
            state = {
                "input": user_input,
                "chat_history": st.session_state.chat_history,
                "context": "",
                "answer": ""
            }
            result = call_model(state)  # This will call agent_chain.invoke

            # If no answer is found, provide a default response
            if not result.get("answer"):
                result["answer"] = "Sorry, I couldn't generate an answer."
            
            # Update the conversation history
            st.session_state.chat_history.append(HumanMessage(user_input))
            st.session_state.chat_history.append(AIMessage(result["answer"]))

        # Display chat history after every form submission
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                if isinstance(message, HumanMessage):
                    st.markdown(f"<div class='chat-bubble human-message'><strong>You:</strong> {message.content}</div>", unsafe_allow_html=True)
                elif isinstance(message, AIMessage):
                    st.markdown(f"<div class='chat-bubble ai-message'><strong>Chatbot:</strong> {message.content}</div>", unsafe_allow_html=True)

            st.experimental_rerun()  # Refresh the app to show the updated chat history

        elif submit_button:
            st.write("Please enter a question.")
