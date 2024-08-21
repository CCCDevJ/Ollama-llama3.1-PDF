from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
import os

app = FastAPI()

# Configuration
folder_path = "db"
pdf_save_path = "pdf"
pdf_static_path = "./pdf/Data_Privacy_Policy_updated.pdf"

# Ensure directories exist
os.makedirs(folder_path, exist_ok=True)
os.makedirs(pdf_save_path, exist_ok=True)

# Initialize components
cached_llm = Ollama(model="llama3.1")
embedding = FastEmbedEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)
raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] You are a technical assistant good at searching documents. If you do not have an answer from the provided information say so. [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
"""
)


class QueryRequest(BaseModel):
    query: str


@app.post("/ai")
async def ai_post(query_request: QueryRequest):
    query = query_request.query
    response = cached_llm.invoke(query)
    return JSONResponse(content={"answer": response})


@app.post("/ask_pdf")
async def ask_pdf_post(query_request: QueryRequest):
    query = query_request.query

    # Load the vector store
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

    # Create the retrieval and document processing chain
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 20, "score_threshold": 0.1},
    )
    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)

    # Perform the retrieval and generation
    result = chain.invoke({"input": query})

    # Extract sources
    sources = [
        {"source": doc.metadata["source"], "page_content": doc.page_content}
        for doc in result["context"]
    ]

    return JSONResponse(content={"answer": result["answer"], "sources": sources})


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Save the uploaded PDF
        file_name = file.filename
        save_file = os.path.join(pdf_save_path, file_name)
        with open(save_file, "wb") as f:
            f.write(file.file.read())

        # Load and split the PDF into documents
        loader = PDFPlumberLoader(save_file)
        docs = loader.load_and_split()

        # Split documents into chunks
        chunks = text_splitter.split_documents(docs)

        # Persist the documents in the vector store
        vector_store = Chroma.from_documents(
            documents=chunks, embedding=embedding, persist_directory=folder_path
        )
        vector_store.persist()

        return JSONResponse(
            content={
                "status": "Successfully Uploaded",
                "filename": file_name,
                "doc_len": len(docs),
                "chunks": len(chunks),
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def process_local_pdf(file_path: str):
    try:
        # Load and split the PDF into documents
        loader = PDFPlumberLoader(file_path)
        docs = loader.load_and_split()

        # Split documents into chunks
        chunks = text_splitter.split_documents(docs)

        # Persist the documents in the vector store
        vector_store = Chroma.from_documents(
            documents=chunks, embedding=embedding, persist_directory=folder_path
        )
        vector_store.persist()

        print({
            "status": "Successfully Processed",
            "filename": os.path.basename(file_path),
            "doc_len": len(docs),
            "chunks": len(chunks),
        })

    except Exception as e:
        print(str(e))


def start_app():
    import uvicorn
    process_local_pdf(pdf_static_path)
    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    start_app()
