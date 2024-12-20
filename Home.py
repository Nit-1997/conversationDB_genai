import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

# Setup API Key
GOOGLE_API_KEY = SecretStr("<API_KEY>")  # Replace with your API Key

# Initialize LLM and QA Chain
def initialize_qa_chain(file_content):
    # Split text
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(file_content)


    # Generate embeddings and vector index
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    print(embeddings)

    vector_index = Chroma.from_texts(texts, embeddings, persist_directory="./chroma_db").as_retriever(search_kwargs={"k": 5})

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY
    )

    # Create RetrievalQA chain
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_index,
        return_source_documents=True
    )

# Read the Cassandra schema
with open('cassandra_schema.cql', 'r') as file:
    file_content = file.read()

# Initialize QA Chain
qa_chain = initialize_qa_chain(file_content)

# Streamlit UI
st.title("Cassandra DB Query Assistant")
st.write("Ask questions about Cassandra DB and get natural language queries in CQL!")

# Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User Input
user_input = st.text_input("Ask your question:")
if user_input:
    with st.spinner("Thinking..."):
        result = qa_chain.invoke({"query": user_input})
        answer = result["result"]

        # Append to chat history
        st.session_state.chat_history.append({"user": user_input, "bot": answer})

# Display Chat History
for chat in st.session_state.chat_history:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Bot:** {chat['bot']}")
