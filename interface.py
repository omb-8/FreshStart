import os
import base64
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# Background image encoding
def get_base64_image(image_path):
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode()

# Streamlit User Interface
def create_interface(call_model):  # Accept the 'call_model' argument here
    # Initialize chat history if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Encoding background image to Base64
    background_image_path = os.path.abspath("uwf-page-edited.jpg")
    base64_background_image = get_base64_image(background_image_path)

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{base64_background_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        h1 {{
            color: #000000;
        }}
        p {{
            color: #000000;
        }}
        .custom-caption {{
            color: #A5A4A4;
            font-size: 16px;
            text-align: center;
        }}
        .chat-bubble {{
            max-width: 70%;
            padding: 10px;
            margin: 5px 0;
            border-radius: 10px;
            color: #000000;
        }}
        .human-message {{
            background-color: #DCF8C6;
            align-self: flex-end;
            text-align: right;
        }}
        .ai-message {{
            background-color: #F1F0F0;
            align-self: flex-start;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Layout for logo and text
    col1, col2 = st.columns([1, 3])
    with col1:
        image_path = os.path.abspath(r"argie-argonaut-gif-edited.png")
        st.image(image_path, width=200)

    with col2:
        st.markdown("<div style='text-align: left;'><h1>ARGObot: UWF's Custom Question-Answering Chatbot</h1></div>", unsafe_allow_html=True)

    # Display description
    st.markdown(
        """
        <div style='text-align: center; font-size: 20px; font-weight: normal;'>
            <p style='font-size: 20px;'>Ask ARGObot a variety of questions based on the Student Handbook.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Display example topics
    st.write("Example topics:")
    st.markdown("<p style='color: #000000;'>- UWF's values</p>", unsafe_allow_html=True)
    st.markdown("<p style='color: #000000;'>- Student Rights and Responsibilities</p>", unsafe_allow_html=True)
    st.markdown("<p style='color: #000000;'>- UWF Policies and Regulations</p>", unsafe_allow_html=True)
    st.markdown("<p style='color: #000000;'>- UWF Appeals and Student Grievance Processes</p>", unsafe_allow_html=True)
    st.markdown("<p style='color: #000000;'>- Student Health and Wellbeing</p>", unsafe_allow_html=True)
    st.markdown("<p style='color: #000000;'>- Student Resources</p>", unsafe_allow_html=True)

    # Display chat container
    chat_container = st.container()

    # Display chat history
    with chat_container:
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                st.markdown(
                    f"<div class='chat-bubble human-message'><strong>You:</strong> {message.content}</div>",
                    unsafe_allow_html=True
                )
            elif isinstance(message, AIMessage):
                st.markdown(
                    f"<div class='chat-bubble ai-message'><strong>Chatbot:</strong> {message.content}</div>",
                    unsafe_allow_html=True
                )
        st.markdown("</div>", unsafe_allow_html=True)

    # Input field for user to type query
    user_input = st.text_input("Enter your question here:", key="user_input")

    # Handle submission and chat update
    def submit_and_clear():
        st.session_state.clear_input = st.session_state.user_input  # Store user input temporarily
        st.session_state.user_input = ""  # Clear the input field

    # Submit button to trigger model interaction
    if st.button("Submit", on_click=submit_and_clear):
        if st.session_state.clear_input:
            user_input = st.session_state.clear_input

            # Prepare state and invoke the model
            state = {"input": user_input, "chat_history": st.session_state.chat_history, "context": "", "answer": ""}
            config = {"configurable": {"thread_id": "246"}}

            result = call_model(state)  # Invoke the call_model function passed as an argument

            # Update chat history with user input and model response
            st.session_state.chat_history.append(HumanMessage(user_input))
            st.session_state.chat_history.append(AIMessage(result["answer"]))

            # Clear the temporary input field
            st.session_state.clear_input = ""

            # Redraw chat history (scrolls to bottom with new message)
            with chat_container:
                st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
                for message in st.session_state.chat_history:
                    if isinstance(message, HumanMessage):
                        st.markdown(
                            f"<div class='chat-bubble human-message'><strong>You:</strong> {message.content}</div>",
                            unsafe_allow_html=True
                        )
                    elif isinstance(message, AIMessage):
                        st.markdown(
                            f"<div class='chat-bubble ai-message'><strong>Chatbot:</strong> {message.content}</div>",
                            unsafe_allow_html=True
                        )
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.write("Please enter a question.")
