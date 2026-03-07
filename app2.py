#fact verification
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
import tempfile
from langchain_core.runnables import RunnablePassthrough


#API keys
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")



st.title("fact verification")
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

uploaded_file = st.file_uploader("upload file", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name
    
    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        separators=["\n\n", "n", ".", " ", ""]
    )
    docs = text_splitter.split_documents(documents)
    embedding = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(docs, embedding)
    st.session_state.vectorstore = vectorstore
    st.success("doc uploaded successfully")

#llm setup

model = init_chat_model("groq:llama-3.1-8b-instant", temperature=0)

prompt = ChatPromptTemplate.from_template("""
you are a document verification assistant

your task is to answer questions about a document and provide supporting evidence if it exists, if it doesnt dont give a answer

rules:
1. only answer using information found in the section below as 'Context'
2. always provide the exact supporting passage
3. if the context is missing the claim, say "not supported by document".
4. do not rely on external knowledge or general answer. it's mandatory to be in the context subsection below
5. if context is empty, then simply respond with no pdf attached

output format:

Answer: <Yes / No / Not supported by document>

Evidence:
"<exact quote from the document>"
(Page <number>, Section <if available>)

Context:
{context}

Question:
{question}
""")

user_input = st.chat_input("ask")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    
    st.session_state.messages.append(("user",user_input))
    if st.session_state.vectorstore:
        retriever = st.session_state.vectorstore.as_retriever()
        def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | model
            | StrOutputParser()
        )

        response = rag_chain.invoke(user_input)
    else:
        chain = prompt | model
        response = chain.invoke({
            "context": " ",
            "question": user_input
        })
    
    with st.chat_message("assistant"):
        st.write(response)
    
    st.session_state.messages.append(("assistant", response))
    