#!/usr/bin/env python3
"""
🚀 01_introduction.py - Your First Hello World with LangChain!

Learning Outcome: 
By the end of this file, you'll understand:
- What is LangChain and why it's awesome
- How to set up your first LLM call
- Basic LangChain architecture
- How to troubleshoot common issues

Created with ❤️ by a fellow developer from tier-2 city
Arre yaar, let's start this LangChain journey together! 🎉
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
# Yeh line bohot important hai, API keys load karne ke liye!
load_dotenv()

try:
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage
    print("✅ All imports successful! LangChain is ready to rock! 🎸")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("📝 Solution: Make sure you've installed all requirements:")
    print("   pip install -r requirements.txt")
    sys.exit(1)


def check_api_key():
    """
    Check if OpenAI API key is properly set
    
    Bhai, without API key, our LangChain is like chai without sugar - useless! 😅
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OpenAI API Key not found!")
        print("📝 Please set your API key in .env file or environment variable")
        print("   OPENAI_API_KEY=sk-your-key-here")
        return False
    
    if not api_key.startswith("sk-"):
        print("⚠️ API key format looks wrong. Should start with 'sk-'")
        return False
    
    print("✅ API key found and format looks good!")
    return True


def basic_llm_example():
    """
    Our first LangChain example - Traditional LLM completion
    
    Think of LLM as your smart friend who completes your sentences
    You give a prompt, it gives a completion. Simple na? 😊
    """
    print("\n" + "="*50)
    print("🤖 BASIC LLM EXAMPLE")
    print("="*50)
    
    try:
        # Initialize the LLM
        # Temperature 0.7 means creative but not too crazy
        # Max tokens 100 means keep the response short and sweet
        llm = OpenAI(
            temperature=0.7,    # How creative should the AI be? (0-1)
            max_tokens=100,     # Maximum response length
            model_name="gpt-3.5-turbo-instruct"  # Which model to use
        )
        
        # Simple prompt - like asking your friend a question
        prompt = "Explain LangChain in simple words, as if talking to a friend from a small town in India:"
        
        print(f"📝 Prompt: {prompt}")
        print("🤔 AI is thinking...")
        
        # Get the response
        response = llm(prompt)
        
        print(f"🤖 AI Response:\n{response.strip()}")
        
    except Exception as e:
        print(f"❌ Error in basic LLM: {e}")
        print("💡 Check your API key and internet connection")


def chat_model_example():
    """
    Chat model example - More like WhatsApp conversation
    
    Chat models understand conversation better than basic LLMs
    They're like your AI friend who remembers context! 💬
    """
    print("\n" + "="*50)
    print("💬 CHAT MODEL EXAMPLE")
    print("="*50)
    
    try:
        # Initialize the chat model
        # Think of this as starting a WhatsApp chat with AI
        chat = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo"  # This is the ChatGPT model!
        )
        
        # Create a human message - like sending a WhatsApp message
        message = HumanMessage(content="""
        Hi! I'm learning LangChain from a tutorial made by someone in a tier-2 city. 
        Can you explain what makes LangChain special in a fun way?
        """)
        
        print("📱 Sending message to AI...")
        
        # Get response - like getting a reply on WhatsApp
        response = chat([message])
        
        print(f"🤖 AI Reply:\n{response.content}")
        
    except Exception as e:
        print(f"❌ Error in chat model: {e}")
        print("💡 Make sure your API key is valid and has credits")


def explain_langchain_basics():
    """
    Let me explain LangChain architecture in simple terms
    
    No fancy words, just pure understanding! 🎓
    """
    print("\n" + "="*50)
    print("🏗️ LANGCHAIN BASICS EXPLAINED")
    print("="*50)
    
    explanation = """
    LangChain क्या है? (What is LangChain?)
    
    Imagine you're building a house:
    🏠 House = Your AI Application
    🧱 Bricks = Different AI models (OpenAI, Hugging Face, etc.)
    🔧 Tools = LangChain components
    👷 You = The developer (that's you!)
    
    LangChain gives you:
    
    1. 🔗 CHAINS: Connect multiple AI operations
       Like: Get user question → Search database → Generate answer
    
    2. 🧠 MEMORY: Make AI remember previous conversations
       Like WhatsApp chat history!
    
    3. 🛠️ TOOLS: Give AI access to external tools
       Like calculator, web search, database queries
    
    4. 📄 DOCUMENT LOADERS: Read from PDFs, websites, etc.
       Feed knowledge to your AI!
    
    5. 🎯 PROMPTS: Smart templates for better AI responses
       Instead of typing same thing again and again
    
    Why LangChain rocks:
    ✅ Saves development time
    ✅ Handles complex AI workflows
    ✅ Works with multiple AI providers
    ✅ Production-ready components
    ✅ Active community support
    
    Think of it as Django/Flask for AI applications! 🚀
    """
    
    print(explanation)


def main():
    """
    Main function - Let's run our first LangChain examples!
    
    Yahan se shuru hota hai humara journey! 🎯
    """
    print("🎉 Welcome to LangChain Tutorial!")
    print("👨‍💻 Created by a fellow developer from tier-2 city")
    print("🚀 Let's build something awesome together!\n")
    
    # Step 1: Check if everything is set up correctly
    if not check_api_key():
        print("\n❌ Setup incomplete. Please check setup_guide.md")
        return
    
    # Step 2: Explain LangChain basics
    explain_langchain_basics()
    
    # Step 3: Try basic LLM
    basic_llm_example()
    
    # Step 4: Try chat model
    chat_model_example()
    
    # Final message
    print("\n" + "="*50)
    print("🎊 CONGRATULATIONS!")
    print("="*50)
    print("✅ You've successfully run your first LangChain examples!")
    print("✅ Your setup is working perfectly!")
    print("✅ You understand the basics of LLMs vs Chat models!")
    print("\n📚 Next Steps:")
    print("   1. Run 02_llm_basics.py to dive deeper")
    print("   2. Experiment with different prompts")
    print("   3. Try changing temperature values")
    print("\n💡 Pro Tip: Higher temperature = more creative responses")
    print("          Lower temperature = more focused/deterministic responses")
    print("\n🎯 Happy Learning! Next stop: Understanding LLMs in detail!")


if __name__ == "__main__":
    # This runs when you execute: python 01_introduction.py
    # Chalo shuru karte hain! Let's start! 🚀
    main()
