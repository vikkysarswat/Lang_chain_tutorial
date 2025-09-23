#!/usr/bin/env python3
"""
🧠 Memory Systems in LangChain - Give Your AI Memory!

Author: Senior AI Developer from Tier-2 City, India
Purpose: Making AI remember conversations like humans do

Arre yaar! Ever wondered how to make your AI remember what you talked about?
Like when you're chatting with your dost and they remember your previous jokes?
That's exactly what we'll learn today - Memory Systems!

Think of it like this:
- Without Memory: AI is like Dory from Finding Nemo - forgets everything instantly
- With Memory: AI is like your best friend - remembers your conversations

What we'll cover:
1. ConversationBufferMemory - Simple conversation memory
2. ConversationBufferWindowMemory - Remember last N messages only
3. ConversationSummaryMemory - Smart summarization
4. ConversationSummaryBufferMemory - Best of both worlds
5. VectorStoreRetrieverMemory - Semantic memory search
6. Custom Memory Implementation - Build your own memory system

Real-world analogy: Memory is like your brain's WhatsApp chat history!
"""

import os
from typing import Dict, List, Any
import warnings
warnings.filterwarnings("ignore")

# Core LangChain imports
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory,
    VectorStoreRetrieverMemory
)
from langchain.memory.chat_message_histories import ChatMessageHistory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

# Vector store imports for advanced memory
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

class MemorySystemsTutorial:
    """
    Comprehensive tutorial for LangChain Memory Systems
    
    Bhai, think of this class as your memory guru who will teach you
    all the memory tricks that even Hrithik Roshan would be jealous of!
    """
    
    def __init__(self):
        """
        Initialize the memory tutorial
        
        Just like setting up your brain for a new day of learning!
        """
        # Set up your OpenAI API key
        # Arre! Don't forget to set your API key in environment variables
        if not os.getenv("OPENAI_API_KEY"):
            print("🚨 Oye! Set your OPENAI_API_KEY environment variable")
            print("export OPENAI_API_KEY='your-api-key-here'")
        
        self.llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-3.5-turbo"
        )
        
    def demonstrate_conversation_buffer_memory(self):
        """
        ConversationBufferMemory - The Simplest Memory System
        
        Analogy: Like writing everything in a notebook and reading the whole thing
        every time someone asks you something. Simple but can get heavy!
        
        Perfect for: Short conversations, debugging, development
        Not good for: Long conversations (token limit issues)
        """
        print("\n🧠 ConversationBufferMemory - The Complete Notebook")
        print("=" * 50)
        
        # Create simple buffer memory
        memory = ConversationBufferMemory(
            return_messages=True,  # Return as message objects
            memory_key="chat_history"  # Key to store memory in prompt
        )
        
        # Create a conversation chain
        conversation = ConversationChain(
            llm=self.llm,
            memory=memory,
            verbose=True  # See what's happening behind the scenes
        )
        
        # Simulate a conversation
        print("🗣️ Starting conversation simulation...")
        
        # First exchange
        response1 = conversation.predict(input="Hi! I'm Vikky from Indore. I love cricket!")
        print(f"AI Response 1: {response1}")
        
        # Second exchange - AI should remember previous context
        response2 = conversation.predict(input="What's my name and which city am I from?")
        print(f"AI Response 2: {response2}")
        
        # Check what's stored in memory
        print("\n📝 What's in memory:")
        print(memory.buffer)
        
        return memory
    
    def demonstrate_conversation_window_memory(self):
        """
        ConversationBufferWindowMemory - The Smart Forgetter
        
        Analogy: Like having a notebook that only keeps the last 3 pages
        Everything older gets torn out automatically!
        
        Perfect for: Long conversations, controlling token usage
        Use when: You want to remember recent context but not everything
        """
        print("\n🪟 ConversationBufferWindowMemory - The Smart Window")
        print("=" * 55)
        
        # Create window memory that remembers only last 2 exchanges
        memory = ConversationBufferWindowMemory(
            k=2,  # Remember last 2 interactions only
            return_messages=True,
            memory_key="chat_history"
        )
        
        conversation = ConversationChain(
            llm=self.llm,
            memory=memory,
            verbose=True
        )
        
        # Simulate multiple exchanges
        exchanges = [
            "Hi, I'm Vikky and I'm learning AI",
            "I live in Indore, Madhya Pradesh",
            "I work as a Python developer",
            "I love biryani and samosas",
            "What do you remember about me?"  # This will test the window
        ]
        
        for i, user_input in enumerate(exchanges, 1):
            print(f"\n💬 Exchange {i}: {user_input}")
            response = conversation.predict(input=user_input)
            print(f"🤖 AI: {response}")
            
            # Show current memory window
            print(f"📱 Memory Window: {memory.buffer}")
        
        return memory
    
    def demonstrate_conversation_summary_memory(self):
        """
        ConversationSummaryMemory - The Smart Summarizer
        
        Analogy: Like having a friend who takes notes of important points
        instead of writing everything word-by-word
        
        Perfect for: Very long conversations, important context retention
        Trade-off: Uses tokens for summarization but saves overall tokens
        """
        print("\n📋 ConversationSummaryMemory - The Smart Summarizer")
        print("=" * 55)
        
        # Create summary memory
        memory = ConversationSummaryMemory(
            llm=self.llm,  # LLM to use for summarization
            return_messages=True,
            memory_key="chat_history"
        )
        
        conversation = ConversationChain(
            llm=self.llm,
            memory=memory,
            verbose=True
        )
        
        # Simulate a longer conversation
        long_conversation = [
            "Hi! I'm Vikky, a software developer from Indore",
            "I specialize in Python, machine learning, and AI development",
            "I've been working on LangChain tutorials to help developers",
            "My goal is to make AI accessible to everyone, especially in tier-2 cities",
            "I believe that language shouldn't be a barrier in tech education",
            "What can you tell me about my background based on our conversation?"
        ]
        
        for i, user_input in enumerate(long_conversation, 1):
            print(f"\n💬 Message {i}: {user_input}")
            response = conversation.predict(input=user_input)
            print(f"🤖 AI: {response}")
            
            # Show the summary that's being maintained
            if hasattr(memory, 'buffer'):
                print(f"📝 Current Summary: {memory.buffer}")
        
        return memory
    
    def demonstrate_summary_buffer_memory(self):
        """
        ConversationSummaryBufferMemory - The Best of Both Worlds
        
        Analogy: Like having a smart notebook that keeps recent stuff as-is
        but summarizes older content. Smart balance!
        
        Perfect for: Long conversations with nuanced recent context
        Best when: You need detailed recent memory + summarized history
        """
        print("\n🎯 ConversationSummaryBufferMemory - The Perfect Balance")
        print("=" * 60)
        
        # Create summary buffer memory
        memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=100,  # When to start summarizing
            return_messages=True,
            memory_key="chat_history"
        )
        
        conversation = ConversationChain(
            llm=self.llm,
            memory=memory,
            verbose=True
        )
        
        # Extended conversation to trigger summarization
        extended_chat = [
            "Hello! I'm Vikky from Indore, working on AI education",
            "I'm creating comprehensive LangChain tutorials for Indian developers",
            "The goal is to bridge the gap between tier-1 and tier-2 city developers",
            "I want to ensure that language and cultural context don't become barriers",
            "Currently focusing on making complex AI concepts simple and relatable",
            "Using analogies and examples that resonate with Indian developers",
            "What aspects of my work seem most important to you?"
        ]
        
        for i, user_input in enumerate(extended_chat, 1):
            print(f"\n💬 Turn {i}: {user_input}")
            response = conversation.predict(input=user_input)
            print(f"🤖 AI: {response}")
            
            # Show memory state
            print(f"🧠 Memory State: {memory.buffer}")
        
        return memory
    
    def demonstrate_vector_store_memory(self):
        """
        VectorStoreRetrieverMemory - The Semantic Memory Master
        
        Analogy: Like having a super-smart librarian who can find related
        books even if you don't remember the exact title!
        
        Perfect for: Large knowledge bases, semantic search in conversations
        Magic: Finds similar conversations/topics even from long ago
        """
        print("\n🔍 VectorStoreRetrieverMemory - The Semantic Search Master")
        print("=" * 65)
        
        try:
            # Create embeddings (need OpenAI API key)
            embeddings = OpenAIEmbeddings()
            
            # Create vector store
            vectorstore = FAISS.from_texts(
                ["Initial conversation started"],  # Start with something
                embeddings
            )
            
            # Create retriever memory
            retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
            memory = VectorStoreRetrieverMemory(
                retriever=retriever,
                memory_key="chat_history"
            )
            
            # Simulate adding memories and retrieving
            conversation_points = [
                "Vikky loves working on AI projects",
                "Indore is a beautiful tier-2 city in Madhya Pradesh",
                "Python is the go-to language for AI development",
                "LangChain makes building AI apps much easier",
                "Education should be accessible to everyone"
            ]
            
            # Add memories
            for point in conversation_points:
                memory.save_context(
                    {"input": f"Learning about: {point}"}, 
                    {"output": f"Understood: {point}"}
                )
            
            # Test semantic retrieval
            test_queries = [
                "Tell me about programming languages",
                "What do you know about cities in India?",
                "Anything about making education better?"
            ]
            
            for query in test_queries:
                print(f"\n🔍 Query: {query}")
                # Load relevant memories
                relevant_memories = memory.load_memory_variables({"prompt": query})
                print(f"🎯 Retrieved: {relevant_memories}")
                
        except Exception as e:
            print(f"⚠️ Vector memory needs OpenAI API key: {e}")
            print("💡 Set OPENAI_API_KEY environment variable to try this!")
        
        return None
    
    def create_custom_memory_system(self):
        """
        Custom Memory Implementation - Build Your Own Memory!
        
        Analogy: Like designing your own personal assistant's brain
        exactly how you want it to work!
        
        Perfect for: Specific use cases, custom business logic
        Power: Full control over what to remember and how
        """
        print("\n🛠️ Custom Memory System - Build Your Own Brain!")
        print("=" * 50)
        
        class IndianContextMemory:
            """
            Custom memory that understands Indian context better
            
            Features:
            - Remembers cultural references
            - Tracks food preferences
            - Understands regional context
            - Maintains relationship context
            """
            
            def __init__(self):
                self.personal_info = {}
                self.cultural_context = []
                self.food_preferences = []
                self.location_context = {}
                self.relationships = []
                
            def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]):
                """Save conversation context with Indian cultural understanding"""
                user_input = inputs.get("input", "").lower()
                
                # Extract personal information
                if "my name is" in user_input or "i'm" in user_input:
                    # Simple name extraction
                    words = user_input.split()
                    for i, word in enumerate(words):
                        if word in ["name", "i'm", "i am"] and i + 1 < len(words):
                            self.personal_info["name"] = words[i + 1].title()
                
                # Detect Indian cities
                indian_cities = ["mumbai", "delhi", "bangalore", "indore", "pune", "chennai", "kolkata", "hyderabad"]
                for city in indian_cities:
                    if city in user_input:
                        self.location_context["city"] = city.title()
                
                # Detect food preferences
                indian_foods = ["biryani", "samosa", "dosa", "chai", "paratha", "dal", "roti"]
                for food in indian_foods:
                    if food in user_input and "love" in user_input:
                        if food not in self.food_preferences:
                            self.food_preferences.append(food)
                
                # Detect cultural/professional context
                if any(term in user_input for term in ["developer", "engineer", "programmer"]):
                    if "profession" not in self.personal_info:
                        self.personal_info["profession"] = "developer"
                
            def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
                """Load relevant memory variables"""
                memory_string = "Context about user:\n"
                
                if self.personal_info:
                    memory_string += f"Personal: {self.personal_info}\n"
                
                if self.location_context:
                    memory_string += f"Location: {self.location_context}\n"
                
                if self.food_preferences:
                    memory_string += f"Food preferences: {', '.join(self.food_preferences)}\n"
                
                return {"chat_history": memory_string}
            
            def clear(self):
                """Clear all memory"""
                self.__init__()
        
        # Demonstrate custom memory
        custom_memory = IndianContextMemory()
        
        # Simulate conversation
        test_inputs = [
            {"input": "Hi, I'm Vikky from Indore and I'm a Python developer"},
            {"input": "I love biryani and samosas, they're my favorite foods"},
            {"input": "What do you know about me?"}
        ]
        
        for inputs in test_inputs:
            print(f"\n👤 User: {inputs['input']}")
            
            # Save context
            custom_memory.save_context(inputs, {"output": "Understood"})
            
            # Load and display memory
            memory_vars = custom_memory.load_memory_variables({})
            print(f"🧠 Memory: {memory_vars['chat_history']}")
        
        return custom_memory
    
    def memory_comparison_guide(self):
        """
        Comprehensive comparison of all memory types
        
        Helping you choose the right memory for your use case!
        """
        print("\n📊 Memory Systems Comparison Guide")
        print("=" * 40)
        
        comparison = {
            "ConversationBufferMemory": {
                "pros": ["Simple", "Complete context", "Easy to debug"],
                "cons": ["Token usage grows", "Can hit limits", "Expensive for long chats"],
                "best_for": "Short conversations, prototyping, debugging"
            },
            "ConversationBufferWindowMemory": {
                "pros": ["Fixed token usage", "Recent context", "Memory efficient"],
                "cons": ["Loses old context", "Fixed window size", "May forget important info"],
                "best_for": "Long conversations, controlled token usage"
            },
            "ConversationSummaryMemory": {
                "pros": ["Scalable", "Keeps important info", "Token efficient for long talks"],
                "cons": ["Uses tokens for summary", "May lose nuances", "Dependent on LLM quality"],
                "best_for": "Very long conversations, important context retention"
            },
            "ConversationSummaryBufferMemory": {
                "pros": ["Best of both worlds", "Flexible", "Nuanced recent + summarized old"],
                "cons": ["Complex", "Still uses tokens", "Needs tuning"],
                "best_for": "Production apps, balanced approach"
            },
            "VectorStoreRetrieverMemory": {
                "pros": ["Semantic search", "Scalable", "Smart retrieval"],
                "cons": ["Complex setup", "Needs embeddings", "Additional infrastructure"],
                "best_for": "Large knowledge bases, semantic similarity"
            }
        }
        
        for memory_type, details in comparison.items():
            print(f"\n🔹 {memory_type}")
            print(f"✅ Pros: {', '.join(details['pros'])}")
            print(f"❌ Cons: {', '.join(details['cons'])}")
            print(f"🎯 Best for: {details['best_for']}")
        
        print("\n💡 Pro Tips for Memory Selection:")
        print("1. Start with ConversationBufferMemory for prototyping")
        print("2. Use WindowMemory for fixed-length conversations")
        print("3. Choose SummaryMemory for cost-effective long conversations")
        print("4. Use SummaryBufferMemory for production apps")
        print("5. Implement VectorMemory for semantic search needs")
        print("6. Build custom memory for specific business logic")

def main():
    """
    Main function to run all memory system demonstrations
    
    Arre bhai, this is where the magic happens!
    Run this to see all memory systems in action.
    """
    print("🚀 Welcome to LangChain Memory Systems Tutorial!")
    print("By: Senior AI Developer from Indore, MP 🇮🇳")
    print("=" * 60)
    
    # Initialize tutorial
    tutorial = MemorySystemsTutorial()
    
    print("🎯 What we'll cover today:")
    print("1. Basic Buffer Memory - The simple notebook")
    print("2. Window Memory - The smart forgetter") 
    print("3. Summary Memory - The intelligent summarizer")
    print("4. Summary+Buffer - The perfect balance")
    print("5. Vector Memory - The semantic search master")
    print("6. Custom Memory - Build your own brain")
    print("7. Comparison Guide - Choose wisely")
    
    try:
        # Run all demonstrations
        tutorial.demonstrate_conversation_buffer_memory()
        tutorial.demonstrate_conversation_window_memory()
        tutorial.demonstrate_conversation_summary_memory()
        tutorial.demonstrate_summary_buffer_memory()
        tutorial.demonstrate_vector_store_memory()
        tutorial.create_custom_memory_system()
        tutorial.memory_comparison_guide()
        
        print("\n🎉 Congratulations! You've mastered LangChain Memory Systems!")
        print("💪 Now you can make your AI remember like an elephant!")
        print("\n🔥 Next steps:")
        print("- Try different memory types in your projects")
        print("- Experiment with custom memory implementations")
        print("- Combine multiple memory systems for complex needs")
        print("- Check out document loaders and text splitters next!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have set up your environment correctly!")
        print("Check requirements.txt and setup_guide.md")

if __name__ == "__main__":
    main()
