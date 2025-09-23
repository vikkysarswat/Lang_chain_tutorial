#!/usr/bin/env python3
"""
🤖 13_chatbot_with_memory.py - Build Your Personal AI Assistant!

Learning Outcomes:
After completing this project, you'll understand:
- How to build a complete chatbot application
- Implementing conversation memory effectively
- Creating personality and context for your AI
- Handling different types of user inputs
- Building a simple CLI interface
- Managing conversation history
- Adding helpful features like commands and shortcuts

This is a complete, production-ready chatbot that remembers conversations!
Built with love by your coding buddy from tier-2 city 🏘️
"""

import os
import json
import datetime
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

try:
    from langchain.chat_models import ChatOpenAI
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.chains import ConversationChain
    from langchain.prompts import PromptTemplate
    from langchain.schema import HumanMessage, AIMessage
    import colorama
    from colorama import Fore, Back, Style
    colorama.init()  # Initialize colorama for cross-platform colored output
    
    print("✅ All imports successful! Ready to build an awesome chatbot!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Run: pip install -r requirements.txt colorama")
    exit(1)


class PersonalAssistant:
    """
    Your Personal AI Assistant with Memory and Personality!
    
    Think of this as your friendly AI companion who remembers your conversations
    and gets to know you better over time! 🤖❤️
    """
    
    def __init__(self, name: str = "Mitra", personality: str = "friendly"):
        """
        Initialize your personal assistant
        
        Args:
            name: Name of your AI assistant
            personality: Personality type (friendly, professional, casual, witty)
        """
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ Please set OPENAI_API_KEY in your .env file!")
            exit(1)
        
        self.name = name
        self.personality = personality
        self.user_name = None
        self.conversation_count = 0
        
        # Initialize the chat model
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
            max_tokens=300  # Keep responses concise for chat
        )
        
        # Set up memory to remember conversations
        # WindowMemory keeps last N exchanges in memory
        self.memory = ConversationBufferWindowMemory(
            k=10,  # Remember last 10 exchanges
            return_messages=True
        )
        
        # Create personality-based system prompt
        self.system_prompt = self._create_system_prompt()
        
        # Create the conversation chain
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=self._create_conversation_prompt(),
            verbose=False  # Set to True for debugging
        )
        
        # Chat history for saving/loading
        self.chat_history_file = f"chat_history_{name.lower()}.json"
        self._load_chat_history()
        
        print(f"🤖 {self.name} initialized with {self.personality} personality!")
    
    def _create_system_prompt(self) -> str:
        """Create personality-based system prompt"""
        personalities = {
            "friendly": """You are Mitra, a friendly and helpful AI assistant from India. You're warm, encouraging, and love helping people. You occasionally use Hindi words naturally and understand Indian culture well. You're like a supportive friend who remembers conversations and cares about the user's wellbeing.""",
            
            "professional": """You are a professional AI assistant who provides clear, concise, and accurate information. You maintain a respectful and business-like tone while being helpful and efficient. You remember previous conversations to provide better assistance.""",
            
            "casual": """You're a chill, laid-back AI buddy who talks like a friend. You're helpful but keep things relaxed and conversational. You remember what users tell you and build on previous chats.""",
            
            "witty": """You're a clever and witty AI assistant with a good sense of humor. You like to add light jokes and playful comments while still being helpful. You remember conversations and can reference previous funny moments."""
        }
        
        return personalities.get(self.personality, personalities["friendly"])
    
    def _create_conversation_prompt(self) -> PromptTemplate:
        """Create the conversation prompt template"""
        template = f"""{self.system_prompt}

Previous conversation:
{{history}}

Human: {{input}}
Assistant:"""
        
        return PromptTemplate(
            input_variables=["history", "input"],
            template=template
        )
    
    def _load_chat_history(self):
        """Load previous chat history if it exists"""
        try:
            if os.path.exists(self.chat_history_file):
                with open(self.chat_history_file, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
                    
                # Load user name if saved
                self.user_name = history_data.get('user_name')
                self.conversation_count = history_data.get('conversation_count', 0)
                
                # Load conversation history into memory
                chat_history = history_data.get('conversations', [])
                for exchange in chat_history[-10:]:  # Load last 10 exchanges
                    self.memory.chat_memory.add_user_message(exchange['human'])
                    self.memory.chat_memory.add_ai_message(exchange['ai'])
                
                print(f"📚 Loaded {len(chat_history)} previous conversations")
        except Exception as e:
            print(f"⚠️ Could not load chat history: {e}")
    
    def _save_chat_history(self):
        """Save current chat history to file"""
        try:
            # Extract conversations from memory
            conversations = []
            messages = self.memory.chat_memory.messages
            
            for i in range(0, len(messages), 2):
                if i + 1 < len(messages):
                    human_msg = messages[i].content
                    ai_msg = messages[i + 1].content
                    conversations.append({
                        'human': human_msg,
                        'ai': ai_msg,
                        'timestamp': datetime.datetime.now().isoformat()
                    })
            
            # Save to file
            history_data = {
                'user_name': self.user_name,
                'conversation_count': self.conversation_count,
                'conversations': conversations,
                'last_updated': datetime.datetime.now().isoformat()
            }
            
            with open(self.chat_history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"⚠️ Could not save chat history: {e}")
    
    def _handle_special_commands(self, user_input: str) -> bool:
        """Handle special commands like /help, /clear, etc."""
        if not user_input.startswith('/'):
            return False
        
        command = user_input.lower().strip()
        
        if command == '/help':
            self._show_help()
        elif command == '/clear':
            self._clear_memory()
        elif command == '/history':
            self._show_history()
        elif command == '/stats':
            self._show_stats()
        elif command == '/name':
            self._change_user_name()
        elif command == '/personality':
            self._show_personality_info()
        elif command == '/quit' or command == '/exit':
            return True
        else:
            print(f"❓ Unknown command: {command}. Type /help for available commands.")
        
        return False
    
    def _show_help(self):
        """Show available commands"""
        print(f"\n{Fore.CYAN}🤖 {self.name}'s Commands:{Style.RESET_ALL}")
        print(f"{Fore.GREEN}/help{Style.RESET_ALL}        - Show this help message")
        print(f"{Fore.GREEN}/clear{Style.RESET_ALL}       - Clear conversation memory")
        print(f"{Fore.GREEN}/history{Style.RESET_ALL}     - Show recent conversation history")
        print(f"{Fore.GREEN}/stats{Style.RESET_ALL}       - Show conversation statistics")
        print(f"{Fore.GREEN}/name{Style.RESET_ALL}        - Change your name")
        print(f"{Fore.GREEN}/personality{Style.RESET_ALL} - Show personality information")
        print(f"{Fore.GREEN}/quit{Style.RESET_ALL}        - Exit the chat")
        print(f"\n{Fore.YELLOW}💡 Tip: Just type normally to chat with me!{Style.RESET_ALL}\n")
    
    def _clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        print(f"🧹 Memory cleared! Starting fresh conversation with {self.name}.")
    
    def _show_history(self):
        """Show recent conversation history"""
        messages = self.memory.chat_memory.messages
        if not messages:
            print("📝 No conversation history yet!")
            return
        
        print(f"\n{Fore.CYAN}📚 Recent Conversation History:{Style.RESET_ALL}")
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                human_msg = messages[i].content[:100] + ("..." if len(messages[i].content) > 100 else "")
                ai_msg = messages[i + 1].content[:100] + ("..." if len(messages[i + 1].content) > 100 else "")
                print(f"{Fore.BLUE}You:{Style.RESET_ALL} {human_msg}")
                print(f"{Fore.GREEN}{self.name}:{Style.RESET_ALL} {ai_msg}")
                print("-" * 40)
    
    def _show_stats(self):
        """Show conversation statistics"""
        message_count = len(self.memory.chat_memory.messages)
        print(f"\n{Fore.CYAN}📊 Conversation Stats:{Style.RESET_ALL}")
        print(f"💬 Messages in memory: {message_count}")
        print(f"🗣️  Total conversations: {self.conversation_count}")
        print(f"👤 Your name: {self.user_name or 'Not set'}")
        print(f"🤖 AI personality: {self.personality}")
        print(f"💾 History file: {self.chat_history_file}")
    
    def _change_user_name(self):
        """Change user's name"""
        new_name = input(f"{Fore.YELLOW}What would you like me to call you?{Style.RESET_ALL} ")
        if new_name.strip():
            self.user_name = new_name.strip()
            print(f"✅ Got it! I'll call you {self.user_name} from now on.")
        else:
            print("❌ Name cannot be empty!")
    
    def _show_personality_info(self):
        """Show current personality information"""
        print(f"\n{Fore.CYAN}🎭 Personality Information:{Style.RESET_ALL}")
        print(f"Name: {self.name}")
        print(f"Personality: {self.personality}")
        print(f"Description: {self.system_prompt[:200]}...")
    
    def chat(self, user_input: str) -> str:
        """Main chat method"""
        try:
            # Handle special commands
            if self._handle_special_commands(user_input):
                return "quit"
            
            # Get AI response using the conversation chain
            response = self.conversation.predict(input=user_input)
            
            # Increment conversation counter
            self.conversation_count += 1
            
            # Save chat history periodically
            if self.conversation_count % 5 == 0:
                self._save_chat_history()
            
            return response.strip()
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)[:100]}..."
            print(f"❌ {error_msg}")
            return "I apologize, but I'm having trouble responding right now. Please try again!"
    
    def start_chat_interface(self):
        """Start the interactive chat interface"""
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}🤖 Welcome to {self.name} - Your Personal AI Assistant!{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        
        if not self.user_name:
            user_name = input(f"{Fore.YELLOW}What should I call you?{Style.RESET_ALL} ")
            if user_name.strip():
                self.user_name = user_name.strip()
                print(f"Nice to meet you, {self.user_name}! 😊")
        else:
            print(f"Welcome back, {self.user_name}! 😊")
        
        print(f"\n{Fore.GREEN}💡 Type '/help' for commands or just start chatting!{Style.RESET_ALL}")
        print(f"{Fore.GREEN}💡 Type '/quit' to exit{Style.RESET_ALL}\n")
        
        # Main chat loop
        while True:
            try:
                # Get user input
                user_input = input(f"{Fore.BLUE}{self.user_name or 'You'}:{Style.RESET_ALL} ")
                
                if not user_input.strip():
                    continue
                
                # Get AI response
                response = self.chat(user_input)
                
                if response == "quit":
                    break
                
                # Display AI response with colors
                print(f"{Fore.GREEN}{self.name}:{Style.RESET_ALL} {response}\n")
                
            except KeyboardInterrupt:
                print(f"\n\n{Fore.YELLOW}Chat interrupted! Saving your conversation...{Style.RESET_ALL}")
                break
            except EOFError:
                print(f"\n{Fore.YELLOW}Chat ended! Saving your conversation...{Style.RESET_ALL}")
                break
        
        # Save final chat history
        self._save_chat_history()
        print(f"\n{Fore.CYAN}👋 Thanks for chatting with {self.name}! Your conversation has been saved.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}💾 History saved to: {self.chat_history_file}{Style.RESET_ALL}")


def create_custom_assistant():
    """Allow users to create a custom assistant"""
    print(f"{Fore.CYAN}🤖 Let's create your personal AI assistant!{Style.RESET_ALL}\n")
    
    # Get assistant name
    name = input(f"{Fore.YELLOW}What would you like to name your assistant? (default: Mitra): {Style.RESET_ALL}")
    if not name.strip():
        name = "Mitra"
    
    # Get personality
    print(f"\n{Fore.YELLOW}Choose a personality:{Style.RESET_ALL}")
    print(f"{Fore.GREEN}1.{Style.RESET_ALL} Friendly - Warm and encouraging (default)")
    print(f"{Fore.GREEN}2.{Style.RESET_ALL} Professional - Business-like and efficient")
    print(f"{Fore.GREEN}3.{Style.RESET_ALL} Casual - Relaxed and conversational")
    print(f"{Fore.GREEN}4.{Style.RESET_ALL} Witty - Humorous and playful")
    
    personality_choice = input(f"\n{Fore.YELLOW}Enter choice (1-4): {Style.RESET_ALL}")
    
    personality_map = {
        "1": "friendly",
        "2": "professional", 
        "3": "casual",
        "4": "witty"
    }
    
    personality = personality_map.get(personality_choice, "friendly")
    
    print(f"\n✅ Creating {name} with {personality} personality...")
    
    return PersonalAssistant(name=name, personality=personality)


def main():
    """
    Main function to run the chatbot application
    
    Chalo apna personal AI assistant banate hain! 🤖
    """
    print(f"{Fore.CYAN}🚀 Welcome to Personal AI Assistant Builder!{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Built with ❤️ by a developer from tier-2 city{Style.RESET_ALL}\n")
    
    try:
        # Create the assistant
        assistant = create_custom_assistant()
        
        # Start the chat interface
        assistant.start_chat_interface()
        
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}Setup interrupted! Come back anytime to create your assistant!{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Make sure your OpenAI API key is set correctly")
    
    print(f"\n{Fore.CYAN}🎊 Thanks for using Personal AI Assistant Builder!{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}🌟 You now know how to build production-ready chatbots with memory!{Style.RESET_ALL}")
    
    # Show what they learned
    print(f"\n{Fore.GREEN}🎓 What you've mastered:{Style.RESET_ALL}")
    print("   ✅ Building conversational AI with memory")
    print("   ✅ Creating personality-driven assistants")
    print("   ✅ Handling user input and commands")
    print("   ✅ Saving and loading conversation history")
    print("   ✅ Building CLI interfaces with colors")
    print("   ✅ Error handling and graceful degradation")
    
    print(f"\n{Fore.YELLOW}🚀 Next challenges:{Style.RESET_ALL}")
    print("   → Add voice input/output capabilities")
    print("   → Connect to external APIs and databases")
    print("   → Deploy as a web application")
    print("   → Add multi-language support")
    print("   → Implement advanced memory systems")
    
    print(f"\n{Fore.BLUE}Happy coding! Keep building awesome AI applications! 💪{Style.RESET_ALL}")


if __name__ == "__main__":
    # Let's build an amazing personal AI assistant!
    # Chalo ek kamaal ka AI assistant banate hain! 🤖✨
    main()
