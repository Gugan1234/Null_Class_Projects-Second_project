import streamlit as st
import os
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import threading
import time

# Set up the API key for Gemini AI
os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Initialize the model
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

# Function to get the Gemini AI response
def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response

# Function to get vector store and periodically update
def get_vector_store_and_update():
    # Load or create the vector store initially
    text_chunks = ["Sample text data for your vector store."]  # Example, replace with your own data
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

    while True:
        # Simulate updating the vector store every hour
        print("Updating vector store...")
        # You would normally update this with new data, e.g., fresh text from documents or inputs.
        new_chunks = ["Updated data at this hour."]
        vector_store.add_texts(new_chunks)
        vector_store.save_local("faiss_index")
        time.sleep(3600)  # Wait for 1 hour

# Start a background thread to update the vector store every hour
update_thread = threading.Thread(target=get_vector_store_and_update, daemon=True)
update_thread.start()

# Initialize Streamlit app
st.set_page_config(page_title="GEMINI CHATBOT DEMO")

st.header("Gemini Application")

# Input field for the user to ask questions
input_text = st.text_input("Input: ", key="input")

# Button to send the query
submit = st.button("Click here to send")

# Display chat messages with emojis
if submit:
    if input_text:
        # User message
        st.chat_message("user", avatar="🧑‍💻").write(input_text)

        # Get Gemini AI response
        response = get_gemini_response(input_text)

        # Display bot response with emoji
        st.subheader("The Response is")
        for chunk in response:
            st.chat_message("assistant", avatar="🤖").write(chunk.text)

        # Optionally, display the full chat history
        st.write(chat.history)
    else:
        st.write("Please enter some text.")

