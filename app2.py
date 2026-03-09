#fact verification backend using fastapi

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
import tempfile
from langchain_core.runnables import RunnablePassthrough
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder, bm25_encoder
from pinecone import Pinecone
from pinecone import ServerlessSpec
from fastapi import FastAPI, UploadFile, File #creating our backend
from pydantic import BaseModel #enforcing data type to input
import uuid #for creating a unique session_id
import uvicorn



#API keys
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

os.environ["api_key"] = os.getenv("pinecone_key")

api_key = os.getenv("pinecone_key")

#starting app
app = FastAPI()

index_name = "hybrid-search-langchain-pinecone-v2"

pc = Pinecone(api_key = api_key)


if index_name not in pc.list_indexes().names():
    pc.create_index(
        name = index_name,
        dimension = 384,
        metric = 'dotproduct',
        spec = ServerlessSpec(cloud ="aws", region="us-east-1")
)

index = pc.Index(index_name)

bm25_encoder = BM25Encoder().default() #uses DF-IDF

embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)


retrievers: dict = {} #for mupliple users (still uses in memory, to extend we can use a database like redis)

@app.post("/upload")
async def upload(file: UploadFile = File(...)): 
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name
    
    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 150,
        separators=["\n\n", "n", ".", " ", ""]
    )
    docs = text_splitter.split_documents(documents)

    texts = [doc.page_content for doc in docs]

    bm25_encoder.fit(texts)
    
    namespace = str(uuid.uuid4()) #to create separate namespace for each file uploaded in different sessions 
    retriever = PineconeHybridSearchRetriever(
        embeddings = embedding,
        sparse_encoder = bm25_encoder,
        index = index,
        top_k = 8,
        alpha = 0.35,
        namespace = namespace,
    )
    retriever.add_texts(texts = texts)

    retrievers[namespace] = retriever
    return {"status": "done", "session_id": namespace, "chunks": len(texts)}

#llm

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

class request_format(BaseModel):  #enforcing request format
    session_id: str
    question: str

@app.post("/response")
async def verify(request: request_format):
    retriever = retrievers.get(request.session_id)

    if retriever is None:
        return {"error": "Pls upload a doc"}

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
    response = rag_chain.invoke(request.question)
    return {"answer": response}


#starting the server

if __name__ == "__main__":
    uvicorn.run("app2:app", host="0.0.0.0", port=8000, reload=True)
