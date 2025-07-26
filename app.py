import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()

# Initialize model and components
model = ChatOpenAI(model="gpt-4o-mini")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embedding_func = OpenAIEmbeddings()

# Streamlit UI
st.set_page_config(page_title="📚 Conversational FAQ Bot")
st.title("🧠 Upload & Ask")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Upload file
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file is not None:
    # Load and split documents
    if uploaded_file.type == "application/pdf":
        with open("temp_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader("temp_file.pdf")
    elif uploaded_file.type == "text/plain":
        with open("temp_file.txt", "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = TextLoader("temp_file.txt", encoding="utf-8")

    documents = loader.load()
    chunks = text_splitter.split_documents(documents)

    # Create FAISS vector store from chunks
    st.session_state.vector_store = FAISS.from_documents(chunks, embedding_func)
    st.success("✅ File uploaded and indexed successfully.")

# Display chat history (above the input)
for speaker, msg in st.session_state.chat_history:
    with st.chat_message(name=speaker.lower()):
        st.markdown(msg)

# Chat input box at the bottom
user_input = st.chat_input("Type your question...")

if user_input:
    with st.chat_message("you"):
        st.markdown(user_input)
    st.session_state.chat_history.append(("You", user_input))

    if st.session_state.vector_store is not None:
        results = st.session_state.vector_store.similarity_search(user_input, k=3)
        context = "\n\n".join([doc.page_content for doc in results]) if results else "No relevant information found."
    else:
        context = "No document uploaded. Please upload a file first."

    # Prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "user_input"],
        template="""
You are an intelligent assistant helping answer questions from a document.

Context:
{context}

User Question:
{user_input}

Give a helpful and concise answer.
"""
    )

    messages = prompt_template.format_prompt(
        context=context,
        user_input=user_input
    )

    # Get AI response
    ai_response = model.invoke(messages)

    # Show AI response
    with st.chat_message("ai"):
        st.markdown(ai_response.content)
    st.session_state.chat_history.append(("AI", ai_response.content))

# Clean up temporary files
if os.path.exists("temp_file.pdf"):
    os.remove("temp_file.pdf")
if os.path.exists("temp_file.txt"):
    os.remove("temp_file.txt")
