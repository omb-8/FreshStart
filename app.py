import streamlit as st
from backend.new_chain import get_chain

# chain = None

# if chain == None:
#     chain = get_chain()

# st.title("ArgoBot")

# question = st.text_area("Enter your question here: ")

# if st.button("Submit"):
#     response = chain.invoke(question)
    
#     st.write(response['result'])

def main():
    st.title("ArgoBot: Multimodal Local ChatBot")
    chat_container = st.container()

    user_input = st.text_input("Enter your query here", key="user_input")

    send_button = st.button("Send", key="send_button")

    if send_button:
        llm_response = "This is a response from the LLM model"

        with chat_container:
            st.chat_message("user").write(user_input)
            st.chat_message("ai").write("here is an answer")

    if __name__ == "__main__":
        main()

