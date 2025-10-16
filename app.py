# app.py
import os
import streamlit as st
from doc_processor import (
    create_chroma_from_uploaded_file, 
    get_existing_collections, 
    delete_collection,
    get_collection_stats
)

# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# App setup
st.set_page_config(
    page_title="Document Q&A Chatbot (Local)", 
    layout="wide",
    page_icon="📄"
)

st.title("📄 Document Q&A Chatbot — Free & Local")
st.markdown("Ask questions about your uploaded PDF documents using local AI models!")

# Constants
CHROMA_DIR = "chromadb"
os.makedirs(CHROMA_DIR, exist_ok=True)  # Only need Chroma directory now

# Sidebar
st.sidebar.header("⚙️ Settings")

# Collection Management
st.sidebar.subheader("📚 Collection Management")
existing_collections = get_existing_collections()
collection_name = st.sidebar.text_input("Collection Name", value="docs")

if existing_collections:
    st.sidebar.info(f"Existing collections: {', '.join(existing_collections)}")
    
    # Show collection stats
    if collection_name in existing_collections:
        stats = get_collection_stats(collection_name)
        st.sidebar.write(f"**{stats['total_chunks']}** chunks from **{stats['unique_documents']}** documents")
        
        if stats.get('sources'):
            with st.sidebar.expander("View Documents"):
                for source in stats['sources']:
                    st.write(f"• {source}")
    
    # Delete collection button
    if st.sidebar.button("🗑️ Delete Current Collection", type="secondary"):
        if delete_collection(collection_name):
            st.rerun()

# Retrieval Settings
st.sidebar.subheader("🔍 Retrieval Settings")
top_k = st.sidebar.slider("Top K retrieval", 1, 10, 4)
chunk_size = st.sidebar.slider("Chunk Size", 500, 2000, 1000)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 500, 200)

# LLM Settings
st.sidebar.subheader("🧠 LLM Settings")
model_name = st.sidebar.selectbox(
    "Choose model",
    [
        "facebook/blenderbot_small-90M", 
        "google/flan-t5-small", 
        "google/flan-t5-base",
        "microsoft/DialoGPT-small"
    ],
    index=1,
)
st.sidebar.info("Using local Hugging Face model — free & runs on CPU/GPU.")

@st.cache_resource
def load_model(_model_name):
    """Load the text generation model"""
    try:
        return pipeline(
            "text2text-generation", 
            model=_model_name,
            max_length=512
        )
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_embeddings():
    """Load the embeddings model"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# Main Content Area
tab1, tab2 = st.tabs(["📤 Upload & Ingest", "❓ Ask Questions"])

with tab1:
    st.header("Upload PDF Documents")
    st.info("💡 Documents are processed in memory and not saved to disk. Only text chunks are stored in the database.")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Choose PDF files", 
        type=["pdf"], 
        accept_multiple_files=True,
        help="Upload one or more PDF documents to process"
    )
    
    if uploaded_files:
        st.success(f"📁 {len(uploaded_files)} file(s) ready for processing")
        
        # Show uploaded files
        for i, f in enumerate(uploaded_files):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{i+1}. {f.name}** ({f.size // 1024} KB)")
            with col2:
                st.write("✅ Ready")
    
    # Ingest button
    if st.button("🚀 Process PDFs", type="primary"):
        if not uploaded_files:
            st.warning("⚠️ Please upload at least one PDF first.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            successful_ingestions = 0
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}... ({i+1}/{len(uploaded_files)})")
                
                # Process PDF directly from memory
                result = create_chroma_from_uploaded_file(
                    uploaded_file, 
                    collection_name=collection_name
                )
                
                if result is not None:
                    successful_ingestions += 1
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.empty()
            if successful_ingestions > 0:
                st.success(f"✅ Successfully processed {successful_ingestions}/{len(uploaded_files)} PDFs!")
                st.rerun()
            else:
                st.error("❌ Failed to process any PDFs. Please check the files and try again.")

with tab2:
    st.header("Ask Questions about Your Documents")
    
    # Load vector DB
    embeddings = load_embeddings()
    vectordb = None
    
    try:
        if os.path.exists(CHROMA_DIR) and collection_name in get_existing_collections():
            vectordb = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=embeddings,
                collection_name=collection_name,
            )
    except Exception as e:
        st.error(f"Error loading vector database: {str(e)}")
    
    if vectordb:
        # Load QA model
        qa_pipeline = load_model(model_name)
        
        if qa_pipeline is None:
            st.error("❌ Failed to load the language model. Please try a different model.")
        else:
            st.success("✅ Database and model loaded successfully!")
            
            # Question input
            query = st.text_area(
                "Enter your question:",
                placeholder="e.g., What are the main findings in this document?",
                height=100
            )
            
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("🤖 Get Answer", type="primary", use_container_width=True):
                    if not query.strip():
                        st.warning("⚠️ Please enter a question.")
                    else:
                        with st.spinner("🔍 Searching documents and generating answer..."):
                            try:
                                # Retrieve relevant documents
                                docs = vectordb.similarity_search(query, k=top_k)
                                context = "\n\n".join([d.page_content for d in docs])
                                
                                # Create enhanced prompt
                                prompt = f"""Based on the following context, answer the question clearly and concisely. If the answer cannot be found in the context, say so.

Context:
{context}

Question: {query}

Answer:"""
                                
                                # Generate answer
                                result = qa_pipeline(
                                    prompt, 
                                    max_new_tokens=200,
                                    do_sample=True,
                                    temperature=0.3
                                )
                                answer = result[0]["generated_text"].strip()
                                
                                # Display results
                                st.markdown("### 🧠 Answer")
                                st.info(answer)
                                
                                # Show retrieved chunks
                                with st.expander("🔍 View Retrieved Context"):
                                    st.write(f"**Retrieved {len(docs)} most relevant chunks:**")
                                    for i, d in enumerate(docs):
                                        st.markdown(f"---")
                                        st.markdown(f"**Chunk {i+1}** | Source: `{d.metadata.get('source', 'Unknown')}`")
                                        st.write(d.page_content)
                                        st.write("")
                                
                            except Exception as e:
                                st.error(f"❌ Error generating answer: {str(e)}")
            
            with col2:
                if st.button("🔄 Clear", use_container_width=True):
                    st.rerun()
    
    else:
        st.warning("""
        ⚠️ No vector database found. 
        
        To start asking questions:
        1. Go to the **Upload & Ingest** tab
        2. Upload one or more PDF files  
        3. Click the **Process PDFs** button
        4. Return here to ask questions
        """)

# Footer
st.markdown("---")
st.markdown(
    "Built with ❤️ using Streamlit, LangChain, and Hugging Face | "
    "All processing happens locally on your machine | "
    "📁 Original files are not stored - only text chunks"
)