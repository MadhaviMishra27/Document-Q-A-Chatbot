# doc_processor.py
import os
import streamlit as st
from typing import List, Optional, BinaryIO
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_DIR = "chromadb"

def extract_text_from_pdf_file(uploaded_file) -> str:
    """
    Extract text from uploaded PDF file object (in memory)
    """
    try:
        # Create a PdfReader from the uploaded file bytes
        pdf_reader = PdfReader(uploaded_file)
        texts = []
        for page in pdf_reader.pages:
            if text := page.extract_text():
                texts.append(text.strip())
        return "\n".join(texts)
    except Exception as e:
        st.error(f"❌ Error reading PDF {uploaded_file.name}: {str(e)}")
        return ""

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into chunks for processing
    """
    if not text.strip():
        return []
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)

def get_existing_collections() -> List[str]:
    """
    Get list of existing Chroma collections
    """
    if not os.path.exists(CHROMA_DIR):
        return []
    
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        return [col.name for col in client.list_collections()]
    except:
        return []

def create_chroma_from_uploaded_file(uploaded_file, collection_name: str = "docs") -> Optional[Chroma]:
    """
    Create or update Chroma vector database from uploaded file (in memory)
    """
    try:
        print(f"📄 Processing {uploaded_file.name} ...")
        
        # Extract text directly from uploaded file
        text = extract_text_from_pdf_file(uploaded_file)
        
        if not text:
            st.error(f"❌ No text extracted from {uploaded_file.name}")
            return None
        
        chunks = chunk_text(text)
        if not chunks:
            st.error(f"❌ No chunks created from {uploaded_file.name}")
            return None
        
        metadatas = [{
            "source": uploaded_file.name, 
            "chunk": i,
            "file_size": uploaded_file.size
        } for i in range(len(chunks))]

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Check if collection exists and add incrementally
        existing_collections = get_existing_collections()
        
        if collection_name in existing_collections:
            # Add to existing collection
            vectordb = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=embeddings,
                collection_name=collection_name,
            )
            vectordb.add_texts(texts=chunks, metadatas=metadatas)
            st.info(f"📥 Added {len(chunks)} chunks from '{uploaded_file.name}' to collection '{collection_name}'")
        else:
            # Create new collection
            vectordb = Chroma.from_texts(
                texts=chunks,
                embedding=embeddings,
                metadatas=metadatas,
                persist_directory=CHROMA_DIR,
                collection_name=collection_name,
            )
            st.info(f"🆕 Created collection '{collection_name}' with {len(chunks)} chunks from '{uploaded_file.name}'")
        
        vectordb.persist()
        print(f"✅ Ingested {len(chunks)} chunks from {uploaded_file.name}")
        return vectordb
        
    except Exception as e:
        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
        return None

def delete_collection(collection_name: str = "docs") -> bool:
    """
    Delete a Chroma collection
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        client.delete_collection(collection_name)
        st.success(f"✅ Collection '{collection_name}' deleted successfully")
        return True
    except Exception as e:
        st.error(f"❌ Error deleting collection: {str(e)}")
        return False

def get_collection_stats(collection_name: str = "docs") -> dict:
    """
    Get statistics about a collection
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectordb = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
        collection = vectordb._collection
        count = collection.count()
        
        # Get unique sources
        results = collection.get()
        sources = set(metadata.get('source', 'Unknown') for metadata in results['metadatas'])
        
        return {
            "total_chunks": count,
            "unique_documents": len(sources),
            "collection_name": collection_name,
            "sources": list(sources)[:10]  # Show first 10 sources
        }
    except:
        return {"total_chunks": 0, "unique_documents": 0, "collection_name": collection_name}