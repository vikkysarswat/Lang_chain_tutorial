#!/usr/bin/env python3
"""
📄 Document QA System - Ask Questions to Your PDFs!

Author: Senior AI Developer from Tier-2 City, India
Purpose: Building a complete production-ready document question-answering system

Arre bhai! Tired of reading through hundreds of pages to find one piece of info?
Want an AI that can instantly answer questions from your documents like a 
personal research assistant? This is it - a complete document QA system! 📚

Real-World Use Cases:
- Legal firms: Query legal documents and case files
- Educational: Students asking questions from textbooks
- Healthcare: Doctors searching medical records
- Corporate: HR searching through policy documents
- Government: Citizens querying government schemes

Think of it like this:
- Upload PDFs, Word docs, or text files
- Ask questions in natural language (Hindi or English)
- Get instant accurate answers with source citations
- Perfect for Indian businesses and students!

What this project includes:
1. Multi-format document processing (PDF, DOCX, TXT)
2. Intelligent text chunking and indexing
3. Vector store for semantic search
4. RAG pipeline for accurate answers
5. Source citation and confidence scores
6. Streamlit UI for easy interaction
7. Production-ready error handling
8. Multilingual support (Hindi + English)

Let's build something amazing! 🚀
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

# Core imports
from datetime import datetime
import json

# LangChain imports
try:
    from langchain_community.document_loaders import (
        PyPDFLoader,
        TextLoader,
        UnstructuredWordDocumentLoader
    )
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("⚠️ Install required packages: pip install langchain langchain-community langchain-openai faiss-cpu pypdf")
    LANGCHAIN_AVAILABLE = False

class DocumentQASystem:
    """
    Complete Document Question-Answering System
    
    Features:
    - Multi-format document support
    - Intelligent chunking
    - Semantic search with FAISS
    - RAG-based answering
    - Source citations
    - Multilingual support
    
    Perfect for Indian businesses, students, and professionals!
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model_name: str = "gpt-3.5-turbo"
    ):
        """
        Initialize Document QA System
        
        Args:
            openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks for context
            model_name: OpenAI model to use
        """
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        self.llm = ChatOpenAI(
            temperature=0.3,
            model_name=model_name,
            openai_api_key=self.api_key
        )
        
        # Storage
        self.vector_store = None
        self.documents = []
        self.qa_chain = None
        
        print("✅ Document QA System initialized!")
    
    def load_documents(self, file_paths: List[str]) -> int:
        """
        Load documents from various file formats
        
        Supports: PDF, DOCX, TXT
        
        Args:
            file_paths: List of file paths to load
            
        Returns:
            Number of documents loaded
        """
        print(f"\n📂 Loading {len(file_paths)} file(s)...")
        
        all_documents = []
        
        for file_path in file_paths:
            try:
                file_path = Path(file_path)
                
                if not file_path.exists():
                    print(f"❌ File not found: {file_path}")
                    continue
                
                # Choose loader based on file extension
                if file_path.suffix.lower() == '.pdf':
                    loader = PyPDFLoader(str(file_path))
                    print(f"📄 Loading PDF: {file_path.name}")
                    
                elif file_path.suffix.lower() in ['.doc', '.docx']:
                    loader = UnstructuredWordDocumentLoader(str(file_path))
                    print(f"📝 Loading Word doc: {file_path.name}")
                    
                elif file_path.suffix.lower() == '.txt':
                    loader = TextLoader(str(file_path), encoding='utf-8')
                    print(f"📃 Loading text file: {file_path.name}")
                    
                else:
                    print(f"⚠️ Unsupported format: {file_path.suffix}")
                    continue
                
                # Load documents
                docs = loader.load()
                
                # Add metadata
                for doc in docs:
                    doc.metadata['source_file'] = file_path.name
                    doc.metadata['file_type'] = file_path.suffix
                    doc.metadata['loaded_at'] = datetime.now().isoformat()
                
                all_documents.extend(docs)
                print(f"   ✅ Loaded {len(docs)} page(s) from {file_path.name}")
                
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
                continue
        
        self.documents = all_documents
        print(f"\n✅ Total documents loaded: {len(self.documents)}")
        return len(self.documents)
    
    def process_documents(self) -> int:
        """
        Process documents: chunk, embed, and index
        
        Returns:
            Number of chunks created
        """
        if not self.documents:
            print("❌ No documents loaded. Call load_documents() first.")
            return 0
        
        print("\n🔪 Processing documents...")
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Split documents
        chunks = text_splitter.split_documents(self.documents)
        print(f"   ✅ Created {len(chunks)} chunks")
        
        # Create vector store
        print("   🧬 Generating embeddings...")
        self.vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        print("   ✅ Vector store created")
        
        # Create QA chain
        self._create_qa_chain()
        
        print(f"\n✅ Processing complete! Ready to answer questions.")
        return len(chunks)
    
    def _create_qa_chain(self):
        """Create the QA chain with custom prompt"""
        
        # Custom prompt for Indian context
        prompt_template = """
आप एक helpful document assistant हैं। दिए गए context से सवाल का सटीक जवाब दें।

Context from documents:
{context}

Question: {question}

Instructions:
- Answer in the same language as the question (Hindi or English)
- Use ONLY information from the provided context
- If the answer is not in the context, say "मुझे दिए गए documents में इसका जवाब नहीं मिला" (I couldn't find this in the documents)
- Be specific and cite relevant information
- Keep answers clear and concise

Answer:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retriever
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}  # Retrieve top 4 relevant chunks
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
    
    def ask_question(
        self,
        question: str,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Ask a question to the documents
        
        Args:
            question: Question to ask
            return_sources: Whether to return source documents
            
        Returns:
            Dictionary with answer and sources
        """
        if not self.qa_chain:
            return {
                "error": "System not initialized. Load and process documents first.",
                "answer": None,
                "sources": []
            }
        
        print(f"\n❓ Question: {question}")
        print("🔍 Searching documents...")
        
        try:
            # Get answer
            result = self.qa_chain.invoke({"query": question})
            
            answer = result["result"]
            sources = result.get("source_documents", [])
            
            # Format response
            response = {
                "question": question,
                "answer": answer,
                "sources": [],
                "timestamp": datetime.now().isoformat()
            }
            
            # Add source information
            if return_sources and sources:
                seen_sources = set()
                for doc in sources:
                    source_file = doc.metadata.get('source_file', 'Unknown')
                    
                    # Avoid duplicate sources
                    if source_file not in seen_sources:
                        response["sources"].append({
                            "file": source_file,
                            "content_preview": doc.page_content[:200] + "...",
                            "metadata": doc.metadata
                        })
                        seen_sources.add(source_file)
            
            # Print response
            print(f"\n💡 Answer: {answer}")
            
            if response["sources"]:
                print(f"\n📚 Sources ({len(response['sources'])}):")
                for i, source in enumerate(response["sources"], 1):
                    print(f"   {i}. {source['file']}")
            
            return response
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return {
                "error": str(e),
                "answer": None,
                "sources": []
            }
    
    def save_index(self, save_path: str):
        """Save the vector store index to disk"""
        if not self.vector_store:
            print("❌ No vector store to save")
            return
        
        self.vector_store.save_local(save_path)
        print(f"💾 Index saved to: {save_path}")
    
    def load_index(self, load_path: str):
        """Load a previously saved vector store index"""
        try:
            self.vector_store = FAISS.load_local(
                load_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self._create_qa_chain()
            print(f"✅ Index loaded from: {load_path}")
        except Exception as e:
            print(f"❌ Error loading index: {e}")


def demo_document_qa_system():
    """
    Demonstration of the Document QA System
    
    This shows how to use the system with sample documents
    """
    print("=" * 70)
    print("🚀 Document QA System - Interactive Demo")
    print("=" * 70)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("""
❌ OpenAI API Key Required!

To use this system:
1. Get an API key from: https://platform.openai.com/api-keys
2. Set it as environment variable:
   export OPENAI_API_KEY='your-key-here'
   
Or provide it when initializing:
   qa_system = DocumentQASystem(openai_api_key='your-key')

For now, here's how the system works:
        """)
        print_system_architecture()
        return
    
    # Initialize system
    print("\n🏗️ Initializing Document QA System...")
    qa_system = DocumentQASystem()
    
    # Example usage guide
    print("\n📖 Usage Example:")
    print("""
# 1. Load your documents
file_paths = [
    "path/to/document1.pdf",
    "path/to/document2.pdf",
    "path/to/document3.txt"
]
qa_system.load_documents(file_paths)

# 2. Process documents (chunk, embed, index)
qa_system.process_documents()

# 3. Ask questions!
response = qa_system.ask_question("What is the main topic?")
print(response["answer"])

# 4. Ask in Hindi
response = qa_system.ask_question("मुख्य बिंदु क्या हैं?")
print(response["answer"])

# 5. Save index for later use
qa_system.save_index("my_documents_index")

# 6. Load saved index
qa_system.load_index("my_documents_index")
    """)
    
    print("\n🇮🇳 Perfect for Indian Use Cases:")
    print("   • Legal: Query case laws and legal documents")
    print("   • Education: Ask questions from textbooks")
    print("   • Healthcare: Search medical records")
    print("   • Corporate: Search policy documents")
    print("   • Research: Analyze research papers")


def print_system_architecture():
    """Print the system architecture diagram"""
    print("\n📐 System Architecture:")
    print("""
┌─────────────────────────────────────────────────────────────┐
│                  Document QA System Pipeline                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  📄 Documents (PDF/DOCX/TXT)                                │
│           ↓                                                  │
│  🔪 Text Splitter (Chunking)                                │
│           ↓                                                  │
│  🧬 Embeddings Generation                                    │
│           ↓                                                  │
│  🗄️ Vector Store (FAISS)                                    │
│           ↓                                                  │
│  ❓ User Question                                            │
│           ↓                                                  │
│  🔍 Semantic Search (Retrieve relevant chunks)              │
│           ↓                                                  │
│  🤖 LLM (Generate answer from context)                      │
│           ↓                                                  │
│  ✅ Answer + Source Citations                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Key Features:
✓ Multi-format support (PDF, Word, Text)
✓ Intelligent chunking with overlap
✓ Semantic search using embeddings
✓ RAG-based accurate answering
✓ Source citation and traceability
✓ Multilingual (Hindi + English)
✓ Production-ready error handling
    """)


def create_streamlit_app():
    """
    Create a Streamlit web interface for the QA system
    
    Save this as app.py and run: streamlit run app.py
    """
    streamlit_code = '''
import streamlit as st
from document_qa_system import DocumentQASystem
import os

st.set_page_config(
    page_title="Document QA System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Question-Answering System")
st.markdown("Upload your documents and ask questions in Hindi or English! 🇮🇳")

# Initialize session state
if 'qa_system' not in st.session_state:
    if os.getenv("OPENAI_API_KEY"):
        st.session_state.qa_system = DocumentQASystem()
        st.session_state.processed = False
    else:
        st.error("⚠️ Please set OPENAI_API_KEY environment variable")
        st.stop()

# Sidebar for document upload
with st.sidebar:
    st.header("📤 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'txt', 'docx'],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            # Save uploaded files temporarily
            temp_paths = []
            for uploaded_file in uploaded_files:
                temp_path = f"/tmp/{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                temp_paths.append(temp_path)
            
            # Load and process
            st.session_state.qa_system.load_documents(temp_paths)
            st.session_state.qa_system.process_documents()
            st.session_state.processed = True
            
            st.success("✅ Documents processed successfully!")

# Main area for Q&A
if st.session_state.processed:
    st.header("❓ Ask Questions")
    
    # Language selector
    language = st.radio("Select Language", ["English", "हिंदी"], horizontal=True)
    
    # Question input
    if language == "English":
        question = st.text_input("Enter your question:")
    else:
        question = st.text_input("अपना सवाल दर्ज करें:")
    
    if question and st.button("Get Answer"):
        with st.spinner("Searching..."):
            response = st.session_state.qa_system.ask_question(question)
            
            # Display answer
            st.markdown("### 💡 Answer")
            st.write(response["answer"])
            
            # Display sources
            if response.get("sources"):
                st.markdown("### 📚 Sources")
                for i, source in enumerate(response["sources"], 1):
                    with st.expander(f"Source {i}: {source['file']}"):
                        st.text(source['content_preview'])
else:
    st.info("👈 Please upload and process documents first")
'''
    
    print("\n🌐 Streamlit Web Interface Code:")
    print("=" * 70)
    print("Save this code as 'app.py' and run: streamlit run app.py")
    print("=" * 70)
    print(streamlit_code)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("📄 Document QA System - Production-Ready Implementation")
    print("=" * 70)
    
    # Run demonstration
    demo_document_qa_system()
    
    # Show architecture
    print_system_architecture()
    
    # Show Streamlit app code
    create_streamlit_app()
    
    print("\n" + "=" * 70)
    print("🎉 Complete Document QA System Ready!")
    print("=" * 70)
    print("""
Next Steps:
1. Set OPENAI_API_KEY environment variable
2. Place your PDF/DOCX/TXT files in a folder
3. Run the system with your documents
4. Ask questions and get instant answers!

For production deployment:
- Add user authentication
- Implement usage tracking
- Add caching for common questions
- Deploy with Docker/Kubernetes
- Monitor performance and costs

Happy Building! 🚀🇮🇳
    """)
