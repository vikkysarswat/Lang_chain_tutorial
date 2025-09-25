#!/usr/bin/env python3
"""
✂️ Text Splitters in LangChain - Chunking Text Intelligently!

Author: Senior AI Developer from Tier-2 City, India
Purpose: Teaching how to split text optimally for AI processing

Arre yaar! Ever tried to eat a whole paratha in one bite? Impossible, right?
Same with AI - it can't process entire documents at once efficiently.
That's where Text Splitters come in - they're like your mom cutting rotis
into perfect bite-sized pieces for better digestion!

Think of Text Splitters like this:
- Your document is a big family meal
- Text Splitter is the wise elder who portions it perfectly
- Each chunk is a perfect bite that AI can understand and process
- Too big = AI chokes, Too small = loses context

What we'll master today:
1. Character Text Splitter - Simple character-based splitting
2. Recursive Character Text Splitter - Smart hierarchical splitting
3. Token Text Splitter - Split by AI tokens (most efficient)
4. Semantic Text Splitter - Split by meaning (advanced)
5. Code Splitters - Special handling for programming languages
6. Markdown/HTML Splitters - Preserve document structure
7. Custom Splitters - Build your own splitting logic
8. Optimization Techniques - Get the perfect chunk size

Real-world analogy: Text Splitters are like a master chef who knows
exactly how to cut vegetables for perfect cooking - not too big, not too small!
"""

import os
import tempfile
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings("ignore")

# Core LangChain text splitter imports
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
    JavaScriptTextSplitter,
    HTMLTextSplitter,
)

# Document processing
from langchain.schema import Document

# Additional utilities
import re
import math
from typing import Callable

class TextSplittersTutorial:
    """
    Comprehensive tutorial for LangChain Text Splitters
    
    Bhai, think of this class as your personal text-cutting expert
    who can slice any document perfectly like cutting vegetables for biryani! 🍛
    """
    
    def __init__(self):
        """
        Initialize the text splitters tutorial
        
        Setting up our text-cutting workshop!
        """
        self.sample_texts = self._create_sample_texts()
        print("🔪 Text Splitting Workshop is ready!")
    
    def _create_sample_texts(self) -> Dict[str, str]:
        """
        Create various sample texts for demonstration
        
        Like preparing different ingredients for different cutting techniques!
        """
        return {
            "simple_text": """
            LangChain is a framework for developing applications powered by language models. It was created to make it easier to build applications that can utilize the power of large language models.

            The framework provides several key components that work together to create powerful AI applications. These include document loaders for ingesting data from various sources, text splitters for breaking down large documents into manageable chunks, and vector stores for semantic search capabilities.

            One of the most popular use cases for LangChain is Retrieval Augmented Generation (RAG). This technique allows you to combine the power of language models with your own data, creating applications that can answer questions about your specific documents or knowledge base.

            Indian developers have been rapidly adopting LangChain for various applications, from customer service chatbots to document analysis systems. The framework's flexibility makes it particularly suitable for handling multilingual content, which is essential in the Indian context.

            The future of LangChain looks promising with continuous updates and improvements. As more developers contribute to the ecosystem, we can expect even more powerful features and integrations that will make AI application development more accessible to everyone, especially developers in tier-2 and tier-3 cities of India.
            """,
            
            "technical_text": """
            # LangChain Architecture Overview
            
            ## Core Components
            
            LangChain consists of several modular components:
            
            ### 1. Language Models (LLMs)
            - OpenAI GPT models
            - Anthropic Claude
            - Local models via Ollama
            - Custom model integrations
            
            ### 2. Prompt Templates
            - Dynamic prompt generation
            - Template variables and formatting
            - Chain-specific prompts
            - Multi-language support
            
            ### 3. Chains
            - Sequential processing pipelines
            - Conditional logic chains
            - Parallel execution chains
            - Custom chain implementations
            
            ### 4. Memory Systems
            - Conversation buffer memory
            - Summary memory
            - Vector store memory
            - Custom memory implementations
            
            ### 5. Tools and Agents
            - External API integrations
            - Database connectors
            - Web search capabilities
            - Custom tool development
            
            ## Implementation Example
            
            ```python
            from langchain import LLMChain
            from langchain.prompts import PromptTemplate
            from langchain_openai import OpenAI
            
            # Create prompt template
            template = "Translate the following text to Hindi: {text}"
            prompt = PromptTemplate(template=template, input_variables=["text"])
            
            # Initialize LLM
            llm = OpenAI(temperature=0.7)
            
            # Create chain
            chain = LLMChain(llm=llm, prompt=prompt)
            
            # Execute
            result = chain.run("Hello, how are you?")
            print(result)
            ```
            
            This architecture allows for flexible and scalable AI application development.
            """,
            
            "code_text": """
            class LangChainRAGSystem:
                '''
                A complete RAG (Retrieval Augmented Generation) system using LangChain
                Perfect for Indian developers who want to build AI-powered applications
                '''
                
                def __init__(self, openai_api_key: str, vector_store_path: str = "vectorstore"):
                    '''Initialize the RAG system with necessary components'''
                    self.openai_api_key = openai_api_key
                    self.vector_store_path = vector_store_path
                    self.setup_components()
                
                def setup_components(self):
                    '''Set up all the necessary components for RAG'''
                    # Initialize embeddings
                    from langchain_openai import OpenAIEmbeddings
                    self.embeddings = OpenAIEmbeddings()
                    
                    # Initialize LLM
                    from langchain_openai import ChatOpenAI
                    self.llm = ChatOpenAI(
                        temperature=0.7,
                        model="gpt-3.5-turbo"
                    )
                
                def query(self, question: str, k: int = 3) -> str:
                    '''Query the RAG system with a question'''
                    return "Response from RAG system"
            """,
            
            "multilingual_text": """
            भारत में आर्टिफिशियल इंटेलिजेंस का विकास

            आर्टिफिशियल इंटेलिजेंस (AI) आज के समय में तकनीकी क्रांति का एक महत्वपूर्ण हिस्सा है। India is rapidly becoming a major player in the global AI landscape, with significant investments in research and development.

            मुख्य क्षेत्र जहाँ AI का उपयोग हो रहा है:
            1. Healthcare - स्वास्थ्य सेवाओं में सुधार
            2. Education - शिक्षा प्रणाली का डिजिटलीकरण  
            3. Agriculture - कृषि में नई तकनीकों का प्रयोग
            4. Finance - वित्तीय सेवाओं में बेहतर security और efficiency

            LangChain जैसे frameworks की मदद से, Indian developers आसानी से AI applications बना सकते हैं। यह विशेष रूप से tier-2 और tier-3 cities के developers के लिए बहुत उपयोगी है।

            The future of AI in India looks very promising. With government initiatives like Digital India and the increasing availability of high-speed internet, more and more developers are getting access to advanced AI tools and frameworks.

            भविष्य में हमें उम्मीद है कि AI technology से भारत की कई समस्याओं का समाधान मिलेगा और देश के development में महत्वपूर्ण योगदान देगी।
            """
        }
    
    def demonstrate_character_text_splitter(self):
        """
        CharacterTextSplitter - The Simple Knife
        
        Analogy: Like using a basic knife to cut vegetables - straightforward
        but not always the smartest way to cut
        
        Perfect for: Simple text, uniform content, basic splitting needs
        Limitation: Doesn't respect natural boundaries like sentences
        """
        print("\n✂️ CharacterTextSplitter - The Simple Chopper")
        print("=" * 50)
        
        # Create basic character splitter
        splitter = CharacterTextSplitter(
            chunk_size=500,  # Maximum characters per chunk
            chunk_overlap=50,  # Overlap between chunks
            length_function=len,  # How to measure chunk size
            separator="\n\n"  # Primary separator to use
        )
        
        # Split the simple text
        text = self.sample_texts["simple_text"]
        chunks = splitter.split_text(text)
        
        print(f"📊 Original text length: {len(text)} characters")
        print(f"📦 Created {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            print(f"\n📄 Chunk {i+1} ({len(chunk)} chars):")
            print(chunk[:100] + "..." if len(chunk) > 100 else chunk)
        
        # Demonstrate with documents
        doc = Document(page_content=text, metadata={"source": "sample"})
        doc_chunks = splitter.split_documents([doc])
        
        print(f"\n📋 Document splitting: {len(doc_chunks)} chunks created")
        print(f"🏷️ First chunk metadata: {doc_chunks[0].metadata}")
        
        print("\n💡 Character Splitter Pro Tips:")
        print("1. Simple and fast, good for uniform text")
        print("2. May break sentences or paragraphs awkwardly")
        print("3. Use separator parameter for better boundaries")
        print("4. Good starting point for experimentation")
        
        return chunks
    
    def demonstrate_recursive_character_splitter(self):
        """
        RecursiveCharacterTextSplitter - The Smart Surgeon
        
        Analogy: Like a skilled surgeon who cuts precisely along natural lines
        - tries to respect paragraphs, then sentences, then words
        
        Perfect for: Most general text processing, maintains context better
        Best practice: This is usually your go-to splitter
        """
        print("\n🧠 RecursiveCharacterTextSplitter - The Smart Surgeon")
        print("=" * 55)
        
        # Create recursive splitter with smart separators
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            length_function=len,
            # Hierarchy of separators - tries in order
            separators=[
                "\n\n",  # Double newline (paragraphs)
                "\n",    # Single newline (lines)
                ". ",    # Sentence endings
                " ",     # Word boundaries
                ""       # Character level (last resort)
            ]
        )
        
        # Split complex text
        text = self.sample_texts["technical_text"]
        chunks = splitter.split_text(text)
        
        print(f"📊 Original text length: {len(text)} characters")
        print(f"📦 Created {len(chunks)} smart chunks")
        
        for i, chunk in enumerate(chunks):
            print(f"\n📄 Smart Chunk {i+1} ({len(chunk)} chars):")
            # Show first and last few words to see boundaries
            words = chunk.strip().split()
            if len(words) > 10:
                preview = " ".join(words[:8]) + " ... " + " ".join(words[-5:])
            else:
                preview = " ".join(words)
            print(f"📝 {preview}")
        
        print("\n💡 Recursive Splitter Pro Tips:")
        print("1. Best general-purpose splitter for most use cases")
        print("2. Respects natural text boundaries hierarchically")
        print("3. Maintains context better than simple character splitting")
        print("4. Customize separators for your specific content type")
        
        return chunks
    
    def create_custom_splitter(self):
        """
        Custom Text Splitter - Build Your Own Intelligence
        
        Analogy: Like training a personal chef who knows exactly how
        you like your food cut based on your specific preferences
        
        Perfect for: Domain-specific content, unique splitting requirements
        Power: Complete control over splitting logic
        """
        print("\n🛠️ Custom Text Splitter - Your Personal Cutting Expert")
        print("=" * 60)
        
        class IndianContextSplitter:
            """
            Custom splitter optimized for Indian content and context
            
            Features:
            - Recognizes Indian language mixed content
            - Handles code-switching between Hindi and English
            - Preserves cultural context and references
            - Optimized chunk sizes for Indian content patterns
            """
            
            def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
            
            def split_text(self, text: str) -> List[str]:
                """Split text with Indian context awareness"""
                chunks = []
                
                # Custom separators for Indian content
                separators = [
                    "\n\n",  # Paragraph breaks
                    "। ",    # Hindi sentence ending
                    ". ",    # English sentence ending
                    "? ",    # Question marks
                    "! ",    # Exclamations
                    "\n",    # Line breaks
                    " ",     # Word boundaries
                    ""       # Character level
                ]
                
                current_chunk = ""
                sentences = self._split_by_separators(text, separators)
                
                for sentence in sentences:
                    # Check if adding sentence would exceed chunk size
                    if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                        chunks.append(current_chunk.strip())
                        # Start new chunk with overlap
                        overlap_text = self._get_overlap(current_chunk)
                        current_chunk = overlap_text + sentence
                    else:
                        current_chunk += sentence
                
                # Add final chunk
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                return chunks
            
            def _split_by_separators(self, text: str, separators: List[str]) -> List[str]:
                """Split text using hierarchy of separators"""
                if not separators:
                    return [text]
                
                separator = separators[0]
                if separator == "":
                    return list(text)
                
                splits = text.split(separator)
                if len(splits) == 1:
                    return self._split_by_separators(text, separators[1:])
                
                result = []
                for split in splits:
                    if split.strip():
                        result.extend(self._split_by_separators(split, separators[1:]))
                        if split != splits[-1]:  # Add separator back except for last
                            result.append(separator)
                
                return result
            
            def _get_overlap(self, text: str) -> str:
                """Get overlap text from the end of current chunk"""
                if len(text) <= self.chunk_overlap:
                    return text + " "
                
                # Try to find sentence boundary for clean overlap
                overlap_candidate = text[-self.chunk_overlap:]
                sentence_end = overlap_candidate.rfind(". ")
                if sentence_end != -1:
                    return overlap_candidate[sentence_end + 2:] + " "
                
                return overlap_candidate + " "
        
        # Demonstrate custom splitter
        custom_splitter = IndianContextSplitter(chunk_size=300, chunk_overlap=50)
        multilingual_text = self.sample_texts["multilingual_text"]
        
        custom_chunks = custom_splitter.split_text(multilingual_text)
        
        print(f"📊 Custom splitter created {len(custom_chunks)} culturally-aware chunks")
        
        for i, chunk in enumerate(custom_chunks):
            print(f"\n🎯 Cultural Chunk {i+1} ({len(chunk)} chars):")
            
            # Detect language mixing
            hindi_chars = len(re.findall(r'[\u0900-\u097F]', chunk))
            english_chars = len(re.findall(r'[A-Za-z]', chunk))
            
            if hindi_chars > 0 and english_chars > 0:
                print("🌐 Mixed Hindi-English content detected")
            elif hindi_chars > 0:
                print("🇮🇳 Hindi content detected")
            else:
                print("🔤 English content")
            
            # Show preview
            preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
            print(f"📝 Content: {preview}")
        
        print("\n💡 Custom Splitter Pro Tips:")
        print("1. Build domain-specific logic for your use case")
        print("2. Consider language patterns and cultural context")
        print("3. Test with various content types")
        print("4. Balance chunk size with context preservation")
        print("5. Handle edge cases gracefully")
        
        return custom_chunks
    
    def optimize_chunk_parameters(self):
        """
        Chunk Size Optimization - Finding the Perfect Balance
        
        Analogy: Like a master chef finding the perfect spice balance
        - too little context = bland results
        - too much context = overwhelming flavors
        - just right = perfect harmony
        
        Perfect for: Fine-tuning your RAG system performance
        """
        print("\n⚖️ Chunk Size Optimization - Finding the Sweet Spot")
        print("=" * 55)
        
        text = self.sample_texts["simple_text"]
        
        # Test different chunk sizes
        test_sizes = [200, 500, 1000, 1500]
        overlap_ratios = [0.1, 0.2, 0.3]  # Overlap as percentage of chunk size
        
        print("🧪 Testing different chunk configurations:")
        print(f"📄 Sample text length: {len(text)} characters")
        
        results = []
        
        for chunk_size in test_sizes:
            for overlap_ratio in overlap_ratios:
                chunk_overlap = int(chunk_size * overlap_ratio)
                
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len
                )
                
                chunks = splitter.split_text(text)
                
                # Calculate metrics
                avg_chunk_size = sum(len(chunk) for chunk in chunks) / len(chunks)
                total_chars = sum(len(chunk) for chunk in chunks)
                efficiency = len(text) / total_chars  # Less redundancy = higher efficiency
                
                result = {
                    'chunk_size': chunk_size,
                    'overlap_ratio': overlap_ratio,
                    'overlap': chunk_overlap,
                    'num_chunks': len(chunks),
                    'avg_size': avg_chunk_size,
                    'efficiency': efficiency,
                    'total_chars': total_chars
                }
                results.append(result)
                
                print(f"\n📊 Config: Size={chunk_size}, Overlap={overlap_ratio:.0%}")
                print(f"   📦 Chunks: {len(chunks)}, Avg Size: {avg_chunk_size:.0f}")
                print(f"   ⚡ Efficiency: {efficiency:.2f}, Total: {total_chars} chars")
        
        # Find optimal configuration
        best_config = max(results, key=lambda x: x['efficiency'])
        
        print(f"\n🏆 Optimal Configuration Found:")
        print(f"   📏 Chunk Size: {best_config['chunk_size']}")
        print(f"   🔄 Overlap: {best_config['overlap']} ({best_config['overlap_ratio']:.0%})")
        print(f"   📦 Chunks: {best_config['num_chunks']}")
        print(f"   ⚡ Efficiency: {best_config['efficiency']:.2f}")
        
        print("\n💡 Optimization Pro Tips:")
        print("1. Balance chunk size with context requirements")
        print("2. More overlap = better context, but more redundancy")
        print("3. Test with your specific content and use case")
        print("4. Consider token limits of your LLM")
        print("5. Monitor RAG system performance with different settings")
        
        return best_config
    
    def splitter_comparison_guide(self):
        """
        Comprehensive comparison of all text splitters
        
        Helping you choose the right splitter for your use case!
        """
        print("\n📊 Text Splitters Comparison Guide")
        print("=" * 45)
        
        comparison = {
            "CharacterTextSplitter": {
                "best_for": "Simple, uniform text content",
                "pros": ["Simple", "Fast", "Predictable"],
                "cons": ["May break context", "Not smart about boundaries"],
                "use_case": "Basic text processing, prototyping"
            },
            "RecursiveCharacterTextSplitter": {
                "best_for": "General text processing (recommended)",
                "pros": ["Context-aware", "Flexible", "Hierarchical splitting"],
                "cons": ["Slightly slower", "More complex"],
                "use_case": "Most production applications, documentation"
            },
            "TokenTextSplitter": {
                "best_for": "LLM optimization, cost control",
                "pros": ["Token-accurate", "Cost-effective", "Model-specific"],
                "cons": ["Requires specific packages", "Less readable chunks"],
                "use_case": "Production RAG, API cost optimization"
            },
            "MarkdownTextSplitter": {
                "best_for": "Structured documents, technical docs",
                "pros": ["Structure-preserving", "Hierarchy-aware"],
                "cons": ["Markdown-specific", "May create uneven chunks"],
                "use_case": "README files, technical documentation"
            },
            "PythonCodeTextSplitter": {
                "best_for": "Python code analysis and documentation",
                "pros": ["Syntax-aware", "Preserves code structure"],
                "cons": ["Language-specific", "Complex setup"],
                "use_case": "Code documentation, API references"
            },
            "Custom Splitter": {
                "best_for": "Domain-specific requirements",
                "pros": ["Complete control", "Domain-optimized"],
                "cons": ["Development time", "Maintenance overhead"],
                "use_case": "Specialized content, unique business logic"
            }
        }
        
        for splitter_name, details in comparison.items():
            print(f"\n🔹 {splitter_name}")
            print(f"🎯 Best for: {details['best_for']}")
            print(f"✅ Pros: {', '.join(details['pros'])}")
            print(f"❌ Cons: {', '.join(details['cons'])}")
            print(f"💼 Use case: {details['use_case']}")
        
        print("\n💡 Selection Guide:")
        print("1. General text → RecursiveCharacterTextSplitter")
        print("2. Cost optimization → TokenTextSplitter")
        print("3. Markdown docs → MarkdownTextSplitter")
        print("4. Code analysis → Language-specific splitters")
        print("5. Special needs → Custom splitter")
        print("6. Simple prototype → CharacterTextSplitter")

def main():
    """
    Main function to run all text splitter demonstrations
    
    Arre bhai, this is where we master the art of text cutting!
    Like learning to cut vegetables from your grandmother - precision matters!
    """
    print("🚀 Welcome to LangChain Text Splitters Tutorial!")
    print("By: Senior AI Developer from Indore, MP 🇮🇳")
    print("=" * 60)
    
    # Initialize tutorial
    tutorial = TextSplittersTutorial()
    
    print("🎯 Today's cutting techniques menu:")
    print("1. Character Splitter - The basic knife")
    print("2. Recursive Splitter - The smart surgeon")
    print("3. Token Splitter - The AI-native precision tool")
    print("4. Markdown Splitter - The structure preserver")
    print("5. Code Splitters - The programming masters")
    print("6. Custom Splitter - Your personal expert")
    print("7. Optimization - Finding the perfect balance")
    print("8. Comparison Guide - Choose wisely")
    
    try:
        # Run all demonstrations
        tutorial.demonstrate_character_text_splitter()
        tutorial.demonstrate_recursive_character_splitter()
        tutorial.create_custom_splitter()
        tutorial.optimize_chunk_parameters()
        tutorial.splitter_comparison_guide()
        
        print("\n🎉 Congratulations! You've mastered Text Splitters!")
        print("💪 Now you can cut text like a master chef cuts vegetables!")
        print("\n🔥 Next steps:")
        print("- Experiment with different splitter configurations")
        print("- Test with your specific content types")
        print("- Combine with document loaders for complete pipelines")
        print("- Optimize chunk sizes for your RAG applications")
        print("- Check out vector databases tutorial next!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have installed all required packages!")
        print("Check requirements.txt and setup_guide.md")

if __name__ == "__main__":
    main()
