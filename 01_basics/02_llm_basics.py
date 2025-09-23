#!/usr/bin/env python3
"""
🧠 02_llm_basics.py - Deep Dive into LLMs and Chat Models

Learning Outcomes:
After completing this tutorial, you'll understand:
- Difference between LLMs and Chat Models (and when to use which!)
- How temperature affects AI creativity
- Token limits and why they matter  
- Different model providers and their strengths
- How to handle API errors like a pro

Bhai, LLMs are the heart of LangChain! Let's understand them properly 💗
Created by your friendly neighborhood developer from tier-2 city 🏘️
"""

import os
import time
from dotenv import load_dotenv

load_dotenv()

try:
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, AIMessage, SystemMessage
    print("✅ All imports successful! Ready to explore LLMs!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Run: pip install -r requirements.txt")
    exit(1)


class LLMExplorer:
    """
    Your friendly LLM exploration class!
    
    Think of this as your guide through the world of Language Models
    Iske saath hum explore karenge LLMs ki duniya! 🗺️
    """
    
    def __init__(self):
        """
        Initialize our LLM explorer
        Setting up our AI companions for the journey!
        """
        # Check API key first
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ Please set OPENAI_API_KEY in your .env file!")
            exit(1)
        
        print("🚀 LLM Explorer initialized! Let's dive deep into AI models!")
    
    def demonstrate_temperature_effects(self):
        """
        Show how temperature affects AI creativity
        
        Temperature is like chai - too cold (0) = boring, too hot (1) = chaotic
        Perfect temperature (0.7) = just right! 🫖
        """
        print("\n" + "="*60)
        print("🌡️ TEMPERATURE EXPERIMENT - AI CREATIVITY LEVELS")
        print("="*60)
        
        prompt = "Write a tagline for a street food stall in India:"
        temperatures = [0.0, 0.5, 0.9]
        
        for temp in temperatures:
            print(f"\n🌡️ Temperature: {temp}")
            if temp == 0.0:
                print("   (Deterministic - Same output every time)")
            elif temp == 0.5:
                print("   (Balanced - Good mix of consistency and creativity)")
            else:
                print("   (Creative - Random and wild responses!)")
            
            try:
                llm = OpenAI(
                    temperature=temp,
                    max_tokens=50,
                    model_name="gpt-3.5-turbo-instruct"
                )
                
                response = llm(prompt)
                print(f"🤖 Response: {response.strip()}")
                
                # Small delay to avoid rate limits (API ki respect karna chahiye!)
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Error at temperature {temp}: {e}")
    
    def llm_vs_chat_comparison(self):
        """
        Compare traditional LLM vs Chat Models
        
        LLM = Your friend who completes your sentences
        Chat = Your friend who has proper conversations
        """
        print("\n" + "="*60)
        print("🥊 LLM vs CHAT MODEL - The Ultimate Showdown!")
        print("="*60)
        
        test_prompt = "Explain quantum computing to a 10-year-old"
        
        print("🔵 ROUND 1: Traditional LLM")
        print("-" * 30)
        try:
            llm = OpenAI(
                temperature=0.7,
                max_tokens=100,
                model_name="gpt-3.5-turbo-instruct"
            )
            
            llm_response = llm(test_prompt)
            print(f"📝 LLM Response:\n{llm_response.strip()}")
            
        except Exception as e:
            print(f"❌ LLM Error: {e}")
        
        print("\n🔴 ROUND 2: Chat Model")
        print("-" * 30)
        try:
            chat = ChatOpenAI(
                temperature=0.7,
                max_tokens=100,
                model_name="gpt-3.5-turbo"
            )
            
            message = HumanMessage(content=test_prompt)
            chat_response = chat([message])
            print(f"💬 Chat Response:\n{chat_response.content}")
            
        except Exception as e:
            print(f"❌ Chat Error: {e}")
        
        print("\n🏆 WINNER: Generally Chat Models! Why?")
        print("   ✅ Better conversation understanding")
        print("   ✅ Handles context more naturally")
        print("   ✅ Designed for back-and-forth dialogue")
        print("   ✅ Better instruction following")
    
    def demonstrate_system_messages(self):
        """
        Show the power of system messages in chat models
        
        System message = Giving your AI friend a personality/role
        Like telling your friend "Act like a teacher" or "Be funny"
        """
        print("\n" + "="*60)
        print("🎭 SYSTEM MESSAGES - Giving AI Different Personalities")
        print("="*60)
        
        user_question = "What's the best way to learn programming?"
        
        # Different system personalities
        personalities = [
            {
                "name": "Professional Teacher",
                "system": "You are a professional programming instructor with 10 years of experience. Be structured and educational."
            },
            {
                "name": "Friendly Bhai",
                "system": "You are a friendly senior developer from India who loves helping juniors. Use casual language and include some Hindi words naturally."
            },
            {
                "name": "Motivational Coach", 
                "system": "You are an enthusiastic motivational coach who believes everyone can learn programming. Be energetic and encouraging!"
            }
        ]
        
        try:
            chat = ChatOpenAI(
                temperature=0.7,
                max_tokens=120,
                model_name="gpt-3.5-turbo"
            )
            
            for personality in personalities:
                print(f"\n🎪 Personality: {personality['name']}")
                print("-" * 40)
                
                messages = [
                    SystemMessage(content=personality['system']),
                    HumanMessage(content=user_question)
                ]
                
                response = chat(messages)
                print(f"🤖 Response: {response.content}")
                
                time.sleep(1)  # Be nice to the API
                
        except Exception as e:
            print(f"❌ Error in personality demo: {e}")
    
    def token_limits_explanation(self):
        """
        Explain tokens and why they matter
        
        Tokens are like words, but not exactly words
        Think of them as the "currency" of AI conversations 💰
        """
        print("\n" + "="*60)
        print("🪙 TOKENS - The Currency of AI Conversations")
        print("="*60)
        
        explanation = """
        What are Tokens? 🤔
        
        Tokens ≠ Words exactly!
        
        Examples:
        • "Hello" = 1 token
        • "Programming" = 1 token  
        • "LangChain" = 2 tokens (Lang + Chain)
        • "AI/ML" = 3 tokens (AI + / + ML)
        • "🚀" = 1 token (yes, emojis count!)
        
        Why Tokens Matter:
        💰 Cost: You pay per token used
        📏 Limits: Models have max token limits
        ⏱️ Speed: More tokens = slower response
        
        Rough Conversion:
        • 1 token ≈ 0.75 words (English)
        • 1 token ≈ 4 characters (average)
        
        Model Token Limits:
        • GPT-3.5-turbo: 4,096 tokens
        • GPT-4: 8,192 tokens (some versions: 32k)
        • GPT-4-turbo: 128,000 tokens
        
        Pro Tips:
        ✅ Keep prompts concise but clear
        ✅ Use max_tokens to control response length
        ✅ Monitor token usage for cost control
        ✅ For long documents, use chunking strategies
        """
        
        print(explanation)
        
        # Demonstrate token counting
        print("\n🧮 TOKEN COUNTING DEMO")
        print("-" * 30)
        
        try:
            # Import tiktoken for accurate token counting
            import tiktoken
            
            # Get the tokenizer for GPT-3.5-turbo
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            
            test_texts = [
                "Hello World!",
                "LangChain is awesome!",
                "भारत में AI का भविष्य उज्ज्वल है।",  # Hindi text
                "🚀 Building AI apps with LangChain 🤖",
                "def hello_world():\n    return 'Hello, LangChain!'"
            ]
            
            for text in test_texts:
                tokens = encoding.encode(text)
                print(f"Text: '{text}'")
                print(f"Token Count: {len(tokens)} tokens")
                print(f"Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
                print("-" * 30)
                
        except ImportError:
            print("⚠️ tiktoken not installed. Install it for accurate token counting:")
            print("   pip install tiktoken")
    
    def error_handling_demo(self):
        """
        Show how to handle common API errors gracefully
        
        Errors happen - network issues, quota exceeded, invalid keys
        Let's handle them like pros! 🛡️
        """
        print("\n" + "="*60)
        print("🛡️ ERROR HANDLING - Dealing with API Issues Like a Pro")
        print("="*60)
        
        print("Common API Errors and Solutions:")
        print("1. 🔑 Invalid API Key → Check your .env file")
        print("2. 💰 Quota Exceeded → Add billing info or wait")
        print("3. 🌐 Network Issues → Retry with backoff")
        print("4. 🔢 Token Limit → Reduce input/output size")
        print("5. ⚡ Rate Limit → Add delays between calls")
        
        # Demonstrate proper error handling
        print("\n🔬 Testing Error Handling:")
        
        try:
            # This might fail if quota is exceeded or network issues
            chat = ChatOpenAI(
                temperature=0.7,
                max_tokens=50,
                model_name="gpt-3.5-turbo",
                request_timeout=10  # 10 second timeout
            )
            
            message = HumanMessage(content="Test message for error handling demo")
            response = chat([message])
            print(f"✅ Success: {response.content[:100]}...")
            
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            
            print(f"❌ Caught {error_type}: {error_msg}")
            
            # Provide specific solutions based on error type
            if "API key" in error_msg.lower():
                print("🔧 Solution: Check your OPENAI_API_KEY in .env file")
            elif "quota" in error_msg.lower() or "billing" in error_msg.lower():
                print("🔧 Solution: Add payment method to your OpenAI account")
            elif "timeout" in error_msg.lower():
                print("🔧 Solution: Check internet connection or increase timeout")
            elif "rate" in error_msg.lower():
                print("🔧 Solution: Add delays between API calls")
            else:
                print("🔧 Solution: Check OpenAI status page and your internet connection")
    
    def model_comparison_guide(self):
        """
        Compare different OpenAI models
        
        Har model ka apna character hai - let's understand them! 🎭
        """
        print("\n" + "="*60)
        print("🏆 MODEL COMPARISON - Choosing the Right AI for the Job")
        print("="*60)
        
        models_info = """
        🤖 OpenAI Model Family:
        
        1. 💰 GPT-3.5-turbo (Most Popular!)
           • Cost: $0.002 per 1K tokens
           • Speed: Fast ⚡
           • Use for: Chatbots, basic text generation
           • Best for: Learning, prototypes, cost-effective apps
        
        2. 🧠 GPT-4 (The Smart One!)
           • Cost: $0.03 per 1K tokens (15x more expensive!)
           • Speed: Slower 🐌  
           • Use for: Complex reasoning, coding, analysis
           • Best for: Production apps that need high quality
        
        3. 📚 GPT-4-turbo (The Balanced One!)
           • Cost: $0.01 per 1K tokens
           • Speed: Faster than GPT-4
           • Context: 128K tokens (huge!)
           • Best for: Long documents, complex tasks
        
        4. 🏃‍♂️ GPT-3.5-turbo-instruct (The Classic!)
           • Traditional completion model
           • Good for: Simple text completion
           • Use when: You need completion, not conversation
        
        💡 Pro Tips for Model Selection:
        
        For Learning → GPT-3.5-turbo (cheap and fast!)
        For Production Chatbots → GPT-3.5-turbo
        For Complex Analysis → GPT-4
        For Long Documents → GPT-4-turbo
        For Simple Completion → GPT-3.5-turbo-instruct
        
        Remember: Start with GPT-3.5-turbo, upgrade only when needed!
        """
        
        print(models_info)


def main():
    """
    Main function to run all LLM demonstrations
    
    Chalo shuru karte hain LLM ki detailed exploration! 🚀
    """
    print("🧠 Welcome to LLM Deep Dive!")
    print("👨‍💻 Your guide from tier-2 city is here to help!")
    print("🎯 Let's master LLMs and Chat Models together!\n")
    
    # Initialize our explorer
    explorer = LLMExplorer()
    
    # Run all demonstrations
    try:
        explorer.demonstrate_temperature_effects()
        explorer.llm_vs_chat_comparison()
        explorer.demonstrate_system_messages()
        explorer.token_limits_explanation()
        explorer.error_handling_demo()
        explorer.model_comparison_guide()
        
        # Final summary
        print("\n" + "="*60)
        print("🎊 CONGRATULATIONS! You've mastered LLM basics!")
        print("="*60)
        print("✅ You understand temperature effects")
        print("✅ You know LLM vs Chat Model differences")
        print("✅ You can use system messages for personality")
        print("✅ You understand tokens and their importance")
        print("✅ You can handle API errors gracefully")
        print("✅ You can choose the right model for your needs")
        
        print("\n📚 What's Next?")
        print("   → 03_prompts_and_templates.py - Master the art of prompting!")
        print("   → Experiment with different temperatures")
        print("   → Try different models and compare results")
        print("   → Practice error handling in your own projects")
        
        print("\n💡 Key Takeaway:")
        print("   Temperature 0.7 is your sweet spot for most applications!")
        print("   GPT-3.5-turbo is perfect for learning and most production use cases!")
        
    except KeyboardInterrupt:
        print("\n\n👋 Tutorial interrupted! Come back anytime!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Check your internet connection and API key")


if __name__ == "__main__":
    # Let's explore the fascinating world of LLMs!
    # Chaliye LLMs ki duniya mein ghumte hain! 🌟
    main()
