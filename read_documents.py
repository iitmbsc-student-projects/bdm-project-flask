import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm


def read_and_split_pdfs(directory, vector_store_path):
    total_characters = 0
    total_pdfs = 0
    all_documents = []
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".pdf"):
            total_pdfs += 1
            file_path = os.path.join(directory, filename)
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            for i, document in enumerate(documents):
                document.metadata = {"source": filename, "page": i + 1}
                all_documents.append(document)
                total_characters += len(document.page_content)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=128000,
        chunk_overlap=200,
        add_start_index=True,
    )
    split_documents = text_splitter.split_documents(all_documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(vector_store_path):
        vector_store = FAISS.load_local(
            vector_store_path, embeddings, allow_dangerous_deserialization=True
        )
    else:
        vector_store = FAISS.from_documents(split_documents, embeddings)
    document_ids = vector_store.add_documents(documents=split_documents)
    vector_store.save_local(vector_store_path)
    return (
        vector_store,
        total_characters,
        total_pdfs,
        split_documents,
        len(document_ids),
    )
