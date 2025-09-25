#!/usr/bin/env python3
"""
🔍 Vector Databases in LangChain - Semantic Search Magic!

Author: Senior AI Developer from Tier-2 City, India
Purpose: Mastering vector databases for semantic search and RAG applications

Arre yaar! Ever wondered how Google finds exactly what you're looking for
even when you don't use the exact words? Or how Netflix recommends movies
you actually want to watch? That's the magic of Vector Databases!

Think of Vector Databases like this:
- Traditional database = Filing cabinet with exact labels
- Vector database = Smart librarian who understands meaning and context
- Instead of exact matches, it finds "similar vibes"
- Like finding all songs that "feel like" your favorite song

What we'll master today:
1. Vector Embeddings - Converting text to numbers that capture meaning
2. FAISS - Facebook's super-fast similarity search
3. Chroma - Simple and efficient vector store  
4. Pinecone - Cloud-based scalable solution
5. Qdrant - Open-source vector database
6. FAISS vs Chroma vs Pinecone - Choose your weapon
7. Semantic Search Implementation - Build your own Google
8. RAG with Vector Stores - AI that knows your documents
9. Performance Optimization - Make it lightning fast

Real-world analogy: Vector databases are like having a friend who
understands not just what you say, but what you mean! 🧠✨
"""

import os
import numpy as np
import tempfile
from typing import List, Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

# Core LangChain imports
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector store imports
try:
    from langchain_community.vectorstores import FAISS, Chroma
    from langchain_openai import OpenAIEmbeddings
    VECTOR_STORES_AVAILABLE = True
except ImportError:
    print("⚠️ Some vector store packages not available. Install with: pip install langchain-community langchain-openai")
    VECTOR_STORES_AVAILABLE = False

# Additional utilities
import pickle
import json
import time
from pathlib import Path

class MockEmbeddings(Embeddings):
    """Mock embeddings for demonstration when OpenAI/HuggingFace not available"""
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Create mock embeddings based on text length and content"""
        embeddings = []
        for text in texts:
            # Create deterministic "embeddings" based on text characteristics
            vec = [
                len(text) / 1000,  # Length feature
                text.count('a') / 100,  # Character frequency
                text.count(' ') / 100,  # Word count proxy
                hash(text) % 1000 / 1000,  # Content hash
            ]
            # Pad to 384 dimensions (common embedding size)
            vec.extend([0.0] * (384 - len(vec)))
            embeddings.append(vec)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Create mock embedding for single query"""
        return self.embed_documents([text])[0]

class VectorDatabasesTutorial:
    """
    Comprehensive tutorial for Vector Databases in LangChain
    
    Bhai, think of this class as your vector database guru who will teach you
    to build search systems that are smarter than a Bollywood scriptwriter! 🎬
    """
    
    def __init__(self):
        """
        Initialize the vector databases tutorial
        
        Setting up our semantic search laboratory!
        """
        self.temp_dir = tempfile.mkdtemp()
        print(f"🏗️ Vector Database Lab initialized at: {self.temp_dir}")
        
        # Create sample documents for demonstration
        self.sample_documents = self._create_sample_documents()
        
        # Initialize embeddings (with fallback)
        self.embeddings = self._initialize_embeddings()
        
    def _initialize_embeddings(self):
        """Initialize embeddings with fallback options"""
        try:
            # Try OpenAI embeddings first
            if os.getenv("OPENAI_API_KEY"):
                print("🔑 Using OpenAI embeddings")
                return OpenAIEmbeddings()
        except Exception as e:
            print(f"⚠️ OpenAI embeddings not available: {e}")
        
        # Fallback to HuggingFace embeddings
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            print("🤗 Using HuggingFace embeddings (sentence-transformers)")
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception as e:
            print(f"⚠️ HuggingFace embeddings not available: {e}")
        
        # Final fallback - mock embeddings for demonstration
        print("🎭 Using mock embeddings for demonstration")
        return MockEmbeddings()
    
    def _create_sample_documents(self) -> List[Document]:
        """
        Create sample documents for vector database demonstrations
        
        Like preparing ingredients for a complex biryani recipe!
        """
        documents_data = [
            {
                "content": """
                LangChain is a revolutionary framework for building applications with Large Language Models (LLMs). 
                It provides a comprehensive suite of tools including document loaders, text splitters, 
                vector stores, and chains that work seamlessly together. Indian developers are increasingly 
                adopting LangChain for building chatbots, document analysis systems, and RAG applications.
                """,
                "metadata": {"source": "langchain_intro.txt", "category": "framework", "language": "english"}
            },
            {
                "content": """
                Vector databases store data as high-dimensional vectors that capture semantic meaning. 
                Unlike traditional databases that rely on exact keyword matching, vector databases 
                enable semantic search where you can find documents based on conceptual similarity.
                This is particularly useful for Indian multilingual content where the same concept 
                might be expressed in different languages.
                """,
                "metadata": {"source": "vector_db_basics.txt", "category": "database", "language": "english"}
            },
            {
                "content": """
                भारत में आर्टिफिशियल इंटेलिजेंस (AI) का तेजी से विकास हो रहा है। LangChain जैसे frameworks 
                की मदद से Indian developers आसानी से AI applications बना सकते हैं। Vector databases का 
                उपयोग करके हम multilingual search और semantic understanding बेहतर बना सकते हैं।
                """,
                "metadata": {"source": "ai_india_hindi.txt", "category": "ai_india", "language": "hindi"}
            },
            {
                "content": """
                Machine Learning और Deep Learning में embeddings का concept बहुत महत्वपूर्ण है। 
                ये high-dimensional vectors होते हैं जो text, images, या अन्य data की semantic 
                information capture करते हैं। Indian context में ये multilingual NLP tasks के 
                लिए बहुत उपयोगी हैं।
                """,
                "metadata": {"source": "embeddings_hindi.txt", "category": "ml_concepts", "language": "hindi"}
            },
            {
                "content": """
                Building RAG (Retrieval Augmented Generation) systems requires careful consideration 
                of document preprocessing, embedding generation, and vector storage. For Indian 
                applications, handling multiple languages and cultural context is crucial. 
                Vector databases like FAISS, Chroma, and Pinecone offer different trade-offs 
                in terms of performance, scalability, and ease of use.
                """,
                "metadata": {"source": "rag_systems.txt", "category": "rag", "language": "english"}
            },
            {
                "content": """
                Performance optimization in vector databases involves several factors: embedding 
                dimensionality, index type, similarity metrics, and query optimization. For Indian 
                applications processing large amounts of multilingual content, choosing the right 
                configuration can significantly impact both speed and accuracy.
                """,
                "metadata": {"source": "performance_optimization.txt", "category": "optimization", "language": "english"}
            }
        ]
        
        documents = []
        for doc_data in documents_data:
            doc = Document(
                page_content=doc_data["content"].strip(),
                metadata=doc_data["metadata"]
            )
            documents.append(doc)
        
        print(f"📚 Created {len(documents)} sample documents for demonstration")
        return documents
    
    def understand_vector_embeddings(self):
        """
        Understanding Vector Embeddings - The Magic Behind Semantic Search
        
        Analogy: Embeddings are like converting every word/sentence into a unique
        "DNA fingerprint" that captures its meaning in numerical form
        
        Perfect for: Understanding how machines understand human language
        """
        print("\n🧬 Understanding Vector Embeddings - The DNA of Text")
        print("=" * 55)
        
        # Sample texts to demonstrate embeddings
        sample_texts = [
            "I love programming in Python",
            "मुझे Python में coding करना पसंद है",  # Hindi: I like coding in Python
            "Python is a great programming language",
            "Machine learning is fascinating",
            "AI और ML बहुत interesting हैं"  # Hindi: AI and ML are very interesting
        ]
        
        print("🔬 Generating embeddings for sample texts...")
        
        # Generate embeddings
        embeddings_list = self.embeddings.embed_documents(sample_texts)
        
        print(f"📊 Generated embeddings with {len(embeddings_list[0])} dimensions")
        
        for i, (text, embedding) in enumerate(zip(sample_texts, embeddings_list)):
            print(f"\n📝 Text {i+1}: {text}")
            print(f"🔢 Embedding preview: [{embedding[0]:.4f}, {embedding[1]:.4f}, {embedding[2]:.4f}, ...]")
            print(f"📏 Embedding magnitude: {np.linalg.norm(embedding):.4f}")
        
        # Demonstrate similarity calculation
        print("\n🎯 Calculating Semantic Similarities:")
        
        def cosine_similarity(vec1, vec2):
            """Calculate cosine similarity between two vectors"""
            dot_product = np.dot(vec1, vec2)
            norm_a = np.linalg.norm(vec1)
            norm_b = np.linalg.norm(vec2)
            return dot_product / (norm_a * norm_b)
        
        # Compare first text with all others
        base_embedding = embeddings_list[0]
        print(f"🎯 Base text: '{sample_texts[0]}'")
        
        for i, (text, embedding) in enumerate(zip(sample_texts[1:], embeddings_list[1:]), 1):
            similarity = cosine_similarity(base_embedding, embedding)
            print(f"   📊 Similarity with '{text}': {similarity:.4f}")
        
        print("\n💡 Embeddings Pro Tips:")
        print("1. Higher dimensional embeddings capture more nuanced meanings")
        print("2. Cosine similarity measures angle between vectors (0-1 scale)")
        print("3. Similar concepts cluster together in vector space")
        print("4. Different embedding models capture different aspects of meaning")
        
        return embeddings_list
    
    def demonstrate_faiss_vector_store(self):
        """
        FAISS (Facebook AI Similarity Search) - The Speed Demon
        
        Analogy: FAISS is like having a super-fast librarian who can find
        similar books from millions of books in microseconds
        
        Perfect for: High-performance local deployments, research applications
        """
        print("\n⚡ FAISS Vector Store - The Lightning Fast Search")
        print("=" * 50)
        
        if not VECTOR_STORES_AVAILABLE:
            print("❌ Vector store packages not available for demonstration")
            print("💡 Install with: pip install langchain-community faiss-cpu")
            return None
        
        try:
            # Create FAISS vector store
            print("🏗️ Creating FAISS vector store...")
            
            vector_store = FAISS.from_documents(
                documents=self.sample_documents,
                embedding=self.embeddings
            )
            
            print(f"✅ FAISS store created with {len(self.sample_documents)} documents")
            
            # Demonstrate basic search
            print("\n🔍 Basic Semantic Search:")
            query = "How to build AI applications in Python?"
            
            results = vector_store.similarity_search(
                query=query,
                k=3  # Return top 3 most similar documents
            )
            
            print(f"🎯 Query: '{query}'")
            print(f"📊 Found {len(results)} similar documents:")
            
            for i, result in enumerate(results):
                print(f"\n📄 Result {i+1}:")
                print(f"📝 Content: {result.page_content[:100]}...")
                print(f"🏷️ Metadata: {result.metadata}")
            
            # Demonstrate search with scores
            print("\n📈 Search with Similarity Scores:")
            results_with_scores = vector_store.similarity_search_with_score(
                query=query,
                k=3
            )
            
            for i, (doc, score) in enumerate(results_with_scores):
                print(f"\n🎯 Result {i+1} (Score: {score:.4f}):")
                print(f"📝 Content: {doc.page_content[:80]}...")
                print(f"📂 Category: {doc.metadata.get('category', 'unknown')}")
            
            print("\n💡 FAISS Pro Tips:")
            print("1. Extremely fast for large-scale similarity search")
            print("2. Supports various index types (Flat, IVF, HNSW)")
            print("3. Great for local deployments and research")
            print("4. No native metadata filtering - implement separately")
            print("5. Can handle millions of vectors efficiently")
            
            return vector_store
            
        except Exception as e:
            print(f"❌ FAISS demonstration failed: {e}")
            print("💡 Install FAISS with: pip install faiss-cpu")
            return None
    
    def demonstrate_chroma_vector_store(self):
        """
        Chroma - The Developer-Friendly Vector Database
        
        Analogy: Chroma is like a well-organized neighborhood library
        - easy to use, good metadata support, perfect for development
        
        Perfect for: Development, prototyping, small to medium datasets
        """
        print("\n🌈 Chroma Vector Store - The Developer's Best Friend")
        print("=" * 55)
        
        if not VECTOR_STORES_AVAILABLE:
            print("❌ Vector store packages not available for demonstration")
            print("💡 Install with: pip install langchain-community chromadb")
            return None
        
        try:
            # Create Chroma vector store
            print("🏗️ Creating Chroma vector store...")
            
            chroma_db_dir = os.path.join(self.temp_dir, "chroma_db")
            
            vector_store = Chroma.from_documents(
                documents=self.sample_documents,
                embedding=self.embeddings,
                persist_directory=chroma_db_dir
            )
            
            print(f"✅ Chroma store created with {len(self.sample_documents)} documents")
            print(f"📁 Database persisted at: {chroma_db_dir}")
            
            # Demonstrate basic search
            print("\n🔍 Semantic Search with Chroma:")
            query = "मशीन लर्निंग और AI"  # Machine learning and AI in Hindi
            
            results = vector_store.similarity_search(
                query=query,
                k=3
            )
            
            print(f"🎯 Query: '{query}' (Hindi)")
            print(f"📊 Found {len(results)} similar documents:")
            
            for i, result in enumerate(results):
                language = result.metadata.get('language', 'unknown')
                category = result.metadata.get('category', 'unknown')
                
                print(f"\n📄 Result {i+1}:")
                print(f"🌍 Language: {language}, 📂 Category: {category}")
                print(f"📝 Content: {result.page_content[:100]}...")
            
            print("\n💡 Chroma Pro Tips:")
            print("1. Excellent metadata filtering capabilities")
            print("2. Built-in persistence - no extra configuration needed")
            print("3. Great for development and prototyping")
            print("4. Easy to deploy and manage")
            print("5. Good performance for small to medium datasets")
            
            return vector_store
            
        except Exception as e:
            print(f"❌ Chroma demonstration failed: {e}")
            print("💡 Install Chroma with: pip install chromadb")
            return None
    
    def vector_store_comparison(self):
        """
        Vector Store Comparison - Choose Your Weapon Wisely
        
        Analogy: Like choosing between different types of vehicles
        - each has its strengths for different journeys
        
        Perfect for: Making informed decisions about vector store selection
        """
        print("\n⚖️ Vector Store Comparison - Choose Your Weapon")
        print("=" * 50)
        
        comparison_data = {
            "FAISS": {
                "description": "Facebook's ultra-fast similarity search library",
                "strengths": [
                    "Extremely fast search (microsecond level)",
                    "Handles millions/billions of vectors",
                    "Multiple index types for different use cases",
                    "Memory efficient with quantization",
                    "Great for research and experimentation"
                ],
                "weaknesses": [
                    "No native metadata filtering",
                    "Local deployment only",
                    "Complex configuration for optimal performance",
                    "No built-in persistence (need manual save/load)"
                ],
                "best_for": "High-performance local apps, research, large-scale similarity search",
                "cost": "Free (open source)",
                "setup_complexity": "Medium to High",
                "scalability": "Excellent (single machine)",
                "metadata_support": "Limited (manual implementation)"
            },
            "Chroma": {
                "description": "Developer-friendly vector database with great DX",
                "strengths": [
                    "Excellent developer experience",
                    "Built-in metadata filtering",
                    "Automatic persistence",
                    "Easy setup and configuration",
                    "Good performance for medium datasets"
                ],
                "weaknesses": [
                    "Not optimized for very large datasets",
                    "Limited production scaling options",
                    "Fewer advanced indexing options",
                    "Primarily local deployment"
                ],
                "best_for": "Development, prototyping, small to medium applications",
                "cost": "Free (open source)",
                "setup_complexity": "Low",
                "scalability": "Good (up to millions of vectors)",
                "metadata_support": "Excellent"
            },
            "Pinecone": {
                "description": "Managed cloud vector database service",
                "strengths": [
                    "Fully managed service",
                    "Excellent scalability",
                    "Real-time updates",
                    "Built-in metadata filtering",
                    "Multiple cloud regions"
                ],
                "weaknesses": [
                    "Costs can add up with scale",
                    "Vendor lock-in",
                    "Less control over infrastructure",
                    "Requires internet connectivity"
                ],
                "best_for": "Production applications, cloud-first architecture, scalable systems",
                "cost": "Paid service (usage-based)",
                "setup_complexity": "Low",
                "scalability": "Excellent (cloud-native)",
                "metadata_support": "Excellent"
            },
            "Qdrant": {
                "description": "Open-source vector search engine with advanced features",
                "strengths": [
                    "High-performance Rust implementation",
                    "Advanced filtering capabilities",
                    "Both cloud and self-hosted options",
                    "Rich metadata support",
                    "Horizontal scaling support"
                ],
                "weaknesses": [
                    "Newer ecosystem (fewer integrations)",
                    "Learning curve for advanced features",
                    "Self-hosting requires infrastructure management"
                ],
                "best_for": "Production applications, advanced filtering needs, hybrid deployments",
                "cost": "Free (open source) + paid cloud options",
                "setup_complexity": "Medium",
                "scalability": "Excellent",
                "metadata_support": "Excellent"
            }
        }
        
        # Display detailed comparison
        for store_name, details in comparison_data.items():
            print(f"\n🔹 {store_name}")
            print(f"📝 Description: {details['description']}")
            print(f"🎯 Best for: {details['best_for']}")
            print(f"💰 Cost: {details['cost']}")
            print(f"⚙️ Setup: {details['setup_complexity']}")
            print(f"📈 Scalability: {details['scalability']}")
            print(f"🏷️ Metadata: {details['metadata_support']}")
            
            print("✅ Strengths:")
            for strength in details['strengths']:
                print(f"   • {strength}")
            
            print("❌ Considerations:")
            for weakness in details['weaknesses']:
                print(f"   • {weakness}")
        
        # Decision matrix
        print("\n🎯 Decision Matrix - Choose Based on Your Needs:")
        print("📊 Dataset Size:")
        print("   • Small (< 100K docs): Chroma, FAISS")
        print("   • Medium (100K - 10M docs): FAISS, Qdrant, Chroma")
        print("   • Large (10M+ docs): FAISS, Pinecone, Qdrant")
        
        print("\n🏗️ Deployment:")
        print("   • Local/On-premise: FAISS, Chroma, Qdrant")
        print("   • Cloud-first: Pinecone, Qdrant Cloud")
        print("   • Hybrid: Qdrant")
        
        print("\n💻 Development Phase:")
        print("   • Prototyping: Chroma")
        print("   • Research: FAISS")
        print("   • Production: Pinecone, Qdrant, optimized FAISS")
        
        print("\n💰 Budget:")
        print("   • Open source only: FAISS, Chroma, Qdrant")
        print("   • Managed service budget: Pinecone, Qdrant Cloud")
        
        print("\n🇮🇳 For Indian Startups/Developers:")
        print("   • Learning: Chroma (easiest to start)")
        print("   • Tier-2 city deployment: FAISS or Qdrant (local control)")
        print("   • Scaling up: Qdrant (open source + cloud options)")
        print("   • Enterprise: Pinecone (managed service)")
    
    def optimize_vector_database_performance(self):
        """
        Vector Database Performance Optimization
        
        Analogy: Like tuning a car engine for maximum performance
        - every component needs to work in perfect harmony
        
        Perfect for: Production deployments, large-scale applications
        """
        print("\n🚀 Vector Database Performance Optimization")
        print("=" * 50)
        
        print("🔧 Performance Optimization Strategies:")
        
        # 1. Embedding Optimization
        print("\n1️⃣ Embedding Optimization:")
        print("   🎯 Dimension Reduction:")
        print("      • Use PCA/UMAP to reduce dimensions while preserving meaning")
        print("      • Trade-off: Slight accuracy loss for major speed gain")
        print("      • Optimal range: 256-768 dimensions for most use cases")
        
        print("   🔢 Quantization:")
        print("      • Convert float32 to int8 embeddings")
        print("      • 4x memory reduction, minimal accuracy impact")
        print("      • Especially effective with FAISS")
        
        # 2. Indexing Strategies
        print("\n2️⃣ Indexing Strategies:")
        print("   ⚡ FAISS Index Types:")
        print("      • Flat: Exact search, best accuracy, slower for large datasets")
        print("      • IVF: Inverted file index, good balance of speed/accuracy")
        print("      • HNSW: Hierarchical graph, excellent for medium datasets")
        print("      • LSH: Locality-sensitive hashing, approximate but very fast")
        
        print("   🎨 Chroma Optimization:")
        print("      • Use appropriate distance metrics (cosine, euclidean)")
        print("      • Batch inserts instead of individual documents")
        print("      • Optimize collection settings")
        
        # 3. Query Optimization
        print("\n3️⃣ Query Optimization:")
        print("   🔍 Search Parameters:")
        print("      • Tune k value: Higher k = more comprehensive but slower")
        print("      • Use pre-filtering when possible")
        print("      • Batch similar queries together")
        
        print("   🏷️ Metadata Strategy:")
        print("      • Index frequently filtered fields")
        print("      • Use hierarchical metadata for complex filtering")
        print("      • Consider separate indexes for different data types")
        
        # 4. Infrastructure Optimization
        print("\n4️⃣ Infrastructure Optimization:")
        print("   💾 Memory Management:")
        print("      • Load indexes into RAM for fastest access")
        print("      • Use memory mapping for large indexes")
        print("      • Monitor memory usage and optimize accordingly")
        
        print("   🔄 Caching Strategies:")
        print("      • Cache frequently accessed embeddings")
        print("      • Use Redis for distributed caching")
        print("      • Implement smart prefetching")
        
        # Demonstrate simple performance testing
        print("\n📊 Performance Testing Example:")
        
        if self.embeddings and hasattr(self, 'sample_documents'):
            # Simple benchmark
            print("🧪 Running simple performance test...")
            
            start_time = time.time()
            
            # Simulate embedding generation
            test_texts = [doc.page_content for doc in self.sample_documents]
            embeddings = self.embeddings.embed_documents(test_texts)
            
            embedding_time = time.time() - start_time
            
            print(f"   📈 Embedding generation:")
            print(f"      • {len(test_texts)} documents")
            print(f"      • {embedding_time:.3f} seconds")
            print(f"      • {len(test_texts)/embedding_time:.1f} docs/sec")
            print(f"      • {len(embeddings[0])} dimensions per embedding")
            
            # Simulate search
            if VECTOR_STORES_AVAILABLE:
                try:
                    vector_store = FAISS.from_documents(self.sample_documents, self.embeddings)
                    
                    start_time = time.time()
                    results = vector_store.similarity_search("AI and machine learning", k=3)
                    search_time = time.time() - start_time
                    
                    print(f"   🔍 Vector search:")
                    print(f"      • Search time: {search_time*1000:.1f} ms")
                    print(f"      • Results: {len(results)} documents")
                    
                except Exception as e:
                    print(f"      ⚠️ Search benchmark failed: {e}")
        
        print("\n💡 Production Performance Tips:")
        print("1. Benchmark with your actual data and query patterns")
        print("2. Monitor search latency and accuracy metrics")
        print("3. Use appropriate hardware (SSDs, sufficient RAM)")
        print("4. Consider geographic distribution for global applications")
        print("5. Implement gradual rollout for performance changes")
        print("6. For Indian applications: Consider network latency between regions")

def main():
    """
    Main function to run all vector database demonstrations
    
    Arre bhai, this is where we become vector database masters!
    Like learning cricket from Dhoni himself! 🏏
    """
    print("🚀 Welcome to Vector Databases Mastery Class!")
    print("By: Senior AI Developer from Indore, MP 🇮🇳")
    print("=" * 60)
    
    # Initialize tutorial
    tutorial = VectorDatabasesTutorial()
    
    print("🎯 Today's vector database journey:")
    print("1. Understanding Vector Embeddings - The DNA of meaning")
    print("2. FAISS - The speed demon")
    print("3. Chroma - The developer's best friend")
    print("4. Vector Store Comparison - Choose your weapon")
    print("5. Performance Optimization - Make it lightning fast")
    
    try:
        # Run all demonstrations
        tutorial.understand_vector_embeddings()
        tutorial.demonstrate_faiss_vector_store()
        tutorial.demonstrate_chroma_vector_store()
        tutorial.vector_store_comparison()
        tutorial.optimize_vector_database_performance()
        
        print("\n🎉 Congratulations! You've mastered Vector Databases!")
        print("💪 Now you can build semantic search systems like a pro!")
        print("\n🔥 Next steps:")
        print("- Build a RAG application with your chosen vector store")
        print("- Experiment with different embedding models")
        print("- Optimize for your specific use case and data")
        print("- Scale your solution based on user needs")
        print("- Check out RAG implementation tutorial next!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have installed required packages!")
        print("pip install langchain-community langchain-openai faiss-cpu chromadb")
    
    finally:
        # Cleanup
        import shutil
        try:
            shutil.rmtree(tutorial.temp_dir)
            print(f"\n🧹 Cleaned up temporary files")
        except:
            pass

if __name__ == "__main__":
    main()
