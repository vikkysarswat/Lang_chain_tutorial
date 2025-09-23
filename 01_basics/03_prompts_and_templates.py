#!/usr/bin/env python3
"""
🎨 03_prompts_and_templates.py - Master the Art of Prompting!

Learning Outcomes:
By the end of this tutorial, you'll understand:
- How to write effective prompts that get better AI responses
- Using PromptTemplate for reusable prompts
- ChatPromptTemplate for conversation flows
- Few-shot prompting (teaching AI by examples)
- Different prompting strategies and when to use them
- How to debug and improve your prompts

Prompting is like giving directions to your AI friend - clear directions = better results! 🗺️
Created with love by a fellow developer from tier-2 city 🏘️
"""

import os
from dotenv import load_dotenv

load_dotenv()

try:
    from langchain.prompts import PromptTemplate, ChatPromptTemplate
    from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
    from langchain.prompts import FewShotPromptTemplate
    from langchain.chat_models import ChatOpenAI
    from langchain.llms import OpenAI
    from langchain.schema import HumanMessage, SystemMessage
    print("✅ All imports successful! Ready to master prompting!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Run: pip install -r requirements.txt")
    exit(1)


class PromptMaster:
    """
    Your friendly neighborhood prompt engineering mentor!
    
    Think of me as your guide to writing prompts that make AI say "Wow!" 🤩
    Prompt writing is an art - let's master it together!
    """
    
    def __init__(self):
        """Initialize our prompt master with AI models"""
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ Please set OPENAI_API_KEY in your .env file!")
            exit(1)
        
        # Initialize both LLM and Chat model for different examples
        self.llm = OpenAI(temperature=0.7, max_tokens=150)
        self.chat = ChatOpenAI(temperature=0.7, max_tokens=150, model_name="gpt-3.5-turbo")
        
        print("🎨 Prompt Master initialized! Let's create some amazing prompts!")
    
    def basic_prompt_template_demo(self):
        """
        Demonstrate basic PromptTemplate usage
        
        PromptTemplate is like a form with blanks to fill in
        Instead of writing same prompt again and again, create a template! 📝
        """
        print("\n" + "="*60)
        print("📝 BASIC PROMPT TEMPLATES - Reusable Prompts Made Easy!")
        print("="*60)
        
        # Bad way: Writing prompts manually every time 😫
        print("❌ Bad Way (Manual prompts every time):")
        manual_prompts = [
            "Write a product description for a smartphone",
            "Write a product description for a laptop", 
            "Write a product description for a headphone"
        ]
        for prompt in manual_prompts:
            print(f"   '{prompt}'")
        
        print("\n✅ Good Way (Using PromptTemplate):")
        
        # Create a reusable template
        product_template = PromptTemplate(
            input_variables=["product_name", "key_features"],
            template="""
            Write an engaging product description for {product_name}.
            
            Key features to highlight:
            {key_features}
            
            Make it sound exciting but honest, like you're recommending it to a friend.
            Write in a conversational tone that appeals to Indian consumers.
            """.strip()
        )
        
        print("📋 Template created!")
        print(f"Template: {product_template.template[:100]}...")
        
        # Test the template with different products
        test_products = [
            {
                "product_name": "OnePlus Nord CE 3",
                "key_features": "50MP camera, 120Hz display, fast charging, 5G ready"
            },
            {
                "product_name": "MacBook Air M2",
                "key_features": "Apple M2 chip, 18-hour battery, ultra-thin design, Retina display"
            }
        ]
        
        print("\n🧪 Testing template with different products:")
        
        try:
            for i, product in enumerate(test_products, 1):
                print(f"\n🔸 Test {i}: {product['product_name']}")
                
                # Format the template with actual values
                formatted_prompt = product_template.format(**product)
                print(f"📝 Generated Prompt:\n{formatted_prompt[:200]}...")
                
                # Get AI response
                response = self.llm(formatted_prompt)
                print(f"🤖 AI Response:\n{response.strip()}")
                print("-" * 50)
                
        except Exception as e:
            print(f"❌ Error in template demo: {e}")
    
    def chat_prompt_template_demo(self):
        """
        Demonstrate ChatPromptTemplate for conversation flows
        
        ChatPromptTemplate is like directing a movie scene
        You set the role, context, and guide the conversation! 🎬
        """
        print("\n" + "="*60)
        print("💬 CHAT PROMPT TEMPLATES - Directing AI Conversations")
        print("="*60)
        
        # Create a chat template with system and human messages
        chat_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """You are a helpful programming mentor from India who loves teaching.
                Your teaching style is friendly, encouraging, and you use simple analogies.
                You occasionally use Hindi words naturally in your explanations.
                Always end your responses with a motivational note."""
            ),
            HumanMessagePromptTemplate.from_template(
                """I'm a beginner learning {programming_language}. 
                Can you explain {concept} in simple terms with a practical example?
                I'm particularly interested in {use_case}."""
            )
        ])
        
        print("🎭 Chat template created with System + Human message flow!")
        
        # Test scenarios
        learning_scenarios = [
            {
                "programming_language": "Python",
                "concept": "functions",
                "use_case": "web scraping"
            },
            {
                "programming_language": "JavaScript", 
                "concept": "async/await",
                "use_case": "API calls"
            }
        ]
        
        try:
            for i, scenario in enumerate(learning_scenarios, 1):
                print(f"\n🎯 Scenario {i}: Learning {scenario['concept']} in {scenario['programming_language']}")
                
                # Format the chat template
                formatted_messages = chat_template.format_messages(**scenario)
                
                print("📨 Generated Messages:")
                for msg in formatted_messages:
                    msg_type = type(msg).__name__.replace("Message", "")
                    print(f"   {msg_type}: {msg.content[:100]}...")
                
                # Get AI response
                response = self.chat(formatted_messages)
                print(f"\n🤖 AI Response:\n{response.content}")
                print("-" * 60)
                
        except Exception as e:
            print(f"❌ Error in chat template demo: {e}")
    
    def few_shot_prompting_demo(self):
        """
        Demonstrate few-shot prompting - teaching AI by examples
        
        Few-shot prompting is like showing your friend examples before asking them to do something
        "Dekho, aise karna hai!" (Look, do it like this!) 🎯
        """
        print("\n" + "="*60)
        print("🎯 FEW-SHOT PROMPTING - Teaching AI by Examples")
        print("="*60)
        
        # Define examples for the AI to learn from
        examples = [
            {
                "input": "Samosa",
                "output": "A crispy, triangular pastry filled with spiced potatoes and peas. Best enjoyed hot with mint chutney! 🥟"
            },
            {
                "input": "Biryani", 
                "output": "Fragrant basmati rice layered with tender meat/vegetables and aromatic spices. A celebration on a plate! 🍛"
            },
            {
                "input": "Masala Chai",
                "output": "Spiced tea brewed with milk, ginger, cardamom, and love. The perfect companion for any mood! ☕"
            }
        ]
        
        # Create the example template (format for each example)
        example_template = PromptTemplate(
            input_variables=["input", "output"],
            template="Food: {input}\nDescription: {output}"
        )
        
        # Create the few-shot template
        few_shot_template = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_template,
            prefix="You are a food enthusiast who writes engaging descriptions of Indian foods. Write descriptions that make people hungry!\n\nHere are some examples:",
            suffix="\nFood: {input}\nDescription:",
            input_variables=["input"]
        )
        
        print("📚 Few-shot template created with examples!")
        print("🍔 Examples provided to AI:")
        for i, example in enumerate(examples, 1):
            print(f"   {i}. {example['input']} → {example['output'][:50]}...")
        
        # Test with new foods
        test_foods = ["Dosa", "Gulab Jamun", "Pani Puri"]
        
        try:
            for food in test_foods:
                print(f"\n🍽️ Testing with: {food}")
                
                # Generate the complete prompt with examples
                formatted_prompt = few_shot_template.format(input=food)
                print(f"📝 Generated prompt includes examples + new food")
                
                # Get AI response
                response = self.llm(formatted_prompt)
                print(f"🤖 AI Response: {response.strip()}")
                print("-" * 40)
                
        except Exception as e:
            print(f"❌ Error in few-shot demo: {e}")
    
    def prompt_engineering_strategies(self):
        """
        Show different prompting strategies and when to use them
        
        Different problems need different approaches
        Let's learn when to use which strategy! 🧠
        """
        print("\n" + "="*60)
        print("🧠 PROMPT ENGINEERING STRATEGIES - The Professional Toolkit")
        print("="*60)
        
        strategies = {
            "🎯 Direct Instruction": {
                "when": "When you want simple, straightforward responses",
                "example": "Explain machine learning in 50 words.",
                "template": "Explain {topic} in {word_count} words."
            },
            
            "🎭 Role-Based": {
                "when": "When you need domain expertise or specific perspective",
                "example": "As a senior software engineer, review this code...",
                "template": "As a {role}, {task}"
            },
            
            "📝 Step-by-Step": {
                "when": "For complex tasks that need systematic approach",
                "example": "Let's solve this problem step by step:\n1. First...",
                "template": "Let's {task} step by step:\n1. First, {step1}\n2. Then, {step2}"
            },
            
            "🔄 Chain of Thought": {
                "when": "For reasoning and problem-solving tasks",
                "example": "Think through this problem: What would happen if...",
                "template": "Think through this problem: {problem}\nLet's reason step by step:"
            },
            
            "📊 Format Specification": {
                "when": "When you need structured output",
                "example": "Provide response in JSON format with keys: name, age, city",
                "template": "Provide response in {format} format with {structure}"
            }
        }
        
        print("🛠️ Professional Prompting Strategies:\n")
        
        for strategy_name, details in strategies.items():
            print(f"{strategy_name}")
            print(f"   📌 Use when: {details['when']}")
            print(f"   💡 Example: {details['example']}")
            print(f"   🔧 Template: {details['template']}")
            print()
        
        # Demonstrate one strategy in action
        print("🧪 Testing Step-by-Step Strategy:")
        
        step_by_step_template = PromptTemplate(
            input_variables=["problem"],
            template="""
            Let's solve this problem step by step:
            
            Problem: {problem}
            
            Please provide a detailed solution following these steps:
            1. First, understand what's being asked
            2. Then, identify the key components
            3. Next, outline the approach
            4. Finally, provide the solution
            
            Think through each step carefully and explain your reasoning.
            """.strip()
        )
        
        try:
            test_problem = "How would you design a simple URL shortener service like bit.ly?"
            
            formatted_prompt = step_by_step_template.format(problem=test_problem)
            print(f"📝 Problem: {test_problem}")
            
            response = self.chat([HumanMessage(content=formatted_prompt)])
            print(f"🤖 Step-by-step solution:\n{response.content}")
            
        except Exception as e:
            print(f"❌ Error in strategy demo: {e}")
    
    def prompt_debugging_tips(self):
        """
        Tips for debugging and improving prompts
        
        When your prompts don't work as expected, here's how to fix them!
        Debugging prompts is like debugging code - systematic approach works! 🔧
        """
        print("\n" + "="*60)
        print("🔧 PROMPT DEBUGGING - Making Your Prompts Work Better!")
        print("="*60)
        
        debugging_guide = """
        🚨 Common Prompt Problems & Solutions:
        
        1. 🤔 VAGUE RESPONSES
           Problem: "Tell me about Python"
           Solution: "Explain Python programming language focusing on its use in web development, with 3 practical examples"
        
        2. 📏 WRONG LENGTH
           Problem: Getting too long/short responses
           Solution: Specify length explicitly - "in 100 words" or "in bullet points"
        
        3. 🎭 WRONG TONE
           Problem: Formal responses when you want casual
           Solution: Add tone instructions - "explain like you're talking to a friend"
        
        4. 📊 UNSTRUCTURED OUTPUT
           Problem: Messy, hard to parse responses
           Solution: Request specific format - "provide as numbered list" or "use JSON format"
        
        5. 🎯 OFF-TOPIC RESPONSES
           Problem: AI goes on tangents
           Solution: Add constraints - "focus only on..." or "don't include..."
        
        6. 🔄 INCONSISTENT RESPONSES
           Problem: Different outputs for same input
           Solution: Lower temperature or add more specific instructions
        
        🛠️ Debugging Process:
        
        Step 1: Identify the problem (too long, wrong tone, off-topic, etc.)
        Step 2: Add specific constraints or instructions
        Step 3: Test with multiple examples
        Step 4: Iterate and refine
        
        💡 Pro Tips:
        ✅ Be specific - vague prompts get vague responses
        ✅ Give examples - show don't just tell
        ✅ Set constraints - tell AI what NOT to do
        ✅ Test with edge cases - try unusual inputs
        ✅ Use role-playing - "act as a..." works well
        """
        
        print(debugging_guide)
        
        # Demonstrate before/after prompt improvement
        print("\n🔄 BEFORE/AFTER PROMPT IMPROVEMENT EXAMPLE:")
        
        print("❌ BEFORE (Vague prompt):")
        bad_prompt = "Write about Indian food"
        print(f"   Prompt: '{bad_prompt}'")
        
        print("\n✅ AFTER (Improved prompt):")
        good_template = PromptTemplate(
            input_variables=["cuisine_region", "audience", "word_count"],
            template="""
            Write an engaging article about {cuisine_region} cuisine for {audience}.
            
            Requirements:
            - Length: approximately {word_count} words
            - Include 3 specific dishes with brief descriptions
            - Mention cultural significance
            - Use warm, inviting tone
            - End with a call-to-action to try the food
            
            Focus on making readers hungry and curious about the cuisine!
            """.strip()
        )
        
        try:
            improved_prompt = good_template.format(
                cuisine_region="South Indian",
                audience="food enthusiasts who haven't tried Indian food",
                word_count="150"
            )
            
            print(f"   Improved Prompt: {improved_prompt[:200]}...")
            
            response = self.llm(improved_prompt)
            print(f"\n🤖 Much better response:\n{response.strip()}")
            
        except Exception as e:
            print(f"❌ Error in debugging demo: {e}")
    
    def advanced_prompt_techniques(self):
        """
        Show advanced prompting techniques for power users
        
        These are professional-level techniques that senior developers use
        Yeh techniques tumhein pro bana dengi! 🚀
        """
        print("\n" + "="*60)
        print("🚀 ADVANCED PROMPT TECHNIQUES - Pro Level Skills!")
        print("="*60)
        
        techniques = """
        🎓 Advanced Techniques:
        
        1. 🎪 PERSONA + CONTEXT + TASK (PCT Framework)
           Persona: "You are a senior DevOps engineer with 10 years experience"
           Context: "Working at a fast-growing startup with microservices"
           Task: "Design a CI/CD pipeline for our Node.js applications"
        
        2. 🔗 CHAIN OF PROMPTS
           Instead of one complex prompt, break into smaller prompts:
           Prompt 1: "Analyze this problem"
           Prompt 2: "Based on the analysis, suggest solutions" 
           Prompt 3: "Pick the best solution and create implementation plan"
        
        3. 🎯 CONSTRAINT-BASED PROMPTING
           Add specific constraints to guide AI:
           "Solve this problem but:
           - Don't use external libraries
           - Maximum 50 lines of code
           - Must work on Python 3.8+"
        
        4. 🔄 ITERATIVE REFINEMENT
           Prompt: "Give me a draft solution"
           Follow-up: "Make it more secure"
           Follow-up: "Optimize for performance"
           Follow-up: "Add error handling"
        
        5. 🎨 CREATIVE CONSTRAINTS
           "Write a function that does X, but explain it as if you're teaching your younger sibling"
           "Solve this problem using only analogies from cooking"
        """
        
        print(techniques)
        
        # Demonstrate PCT Framework
        print("\n🎪 Demonstrating PCT Framework:")
        
        pct_template = PromptTemplate(
            input_variables=["persona", "context", "task"],
            template="""
            PERSONA: {persona}
            
            CONTEXT: {context}
            
            TASK: {task}
            
            Please provide a comprehensive solution considering your expertise and the given context.
            Structure your response with clear reasoning and actionable steps.
            """.strip()
        )
        
        try:
            pct_example = pct_template.format(
                persona="You are a senior full-stack developer who specializes in building scalable web applications",
                context="You're working on a social media app that's expecting rapid user growth from 1000 to 100,000 users",
                task="Design the database schema and explain how you'd handle the scaling challenges"
            )
            
            print("📋 PCT Framework Example:")
            print(f"Generated prompt: {pct_example[:300]}...")
            
            response = self.chat([HumanMessage(content=pct_example)])
            print(f"\n🤖 Professional response:\n{response.content}")
            
        except Exception as e:
            print(f"❌ Error in PCT demo: {e}")


def main():
    """
    Main function to run all prompting demonstrations
    
    Chalo prompt engineering ki duniya mein ghuste hain! 🎨
    """
    print("🎨 Welcome to Prompt Engineering Masterclass!")
    print("👨‍💻 Your mentor from tier-2 city is here to guide you!")
    print("🎯 Let's turn you into a prompt engineering pro!\n")
    
    # Initialize our prompt master
    master = PromptMaster()
    
    try:
        # Run all demonstrations
        master.basic_prompt_template_demo()
        master.chat_prompt_template_demo()
        master.few_shot_prompting_demo()
        master.prompt_engineering_strategies()
        master.prompt_debugging_tips()
        master.advanced_prompt_techniques()
        
        # Final graduation message
        print("\n" + "="*60)
        print("🎊 CONGRATULATIONS! You're now a Prompt Engineering Pro!")
        print("="*60)
        print("✅ You can create reusable PromptTemplates")
        print("✅ You master chat conversations with ChatPromptTemplate")
        print("✅ You can teach AI by examples (few-shot prompting)")
        print("✅ You know different prompting strategies")
        print("✅ You can debug and improve prompts")
        print("✅ You've learned advanced professional techniques")
        
        print("\n🎓 Your New Skills:")
        print("   → Write prompts that get consistent, quality responses")
        print("   → Create templates for reusable prompts")
        print("   → Guide AI conversations with system messages")
        print("   → Use examples to teach AI new patterns")
        print("   → Debug prompts when they don't work as expected")
        
        print("\n📚 What's Next?")
        print("   → 04_output_parsers.py - Learn to structure AI responses")
        print("   → Practice with your own use cases")
        print("   → Experiment with different prompting strategies")
        print("   → Build a prompt library for your projects")
        
        print("\n💡 Remember:")
        print("   Good prompts = Good AI responses")
        print("   Practice makes perfect!")
        print("   Every expert was once a beginner!")
        
    except KeyboardInterrupt:
        print("\n\n👋 Tutorial interrupted! Come back anytime to continue learning!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Check your internet connection and API key")


if __name__ == "__main__":
    # Let's master the art of prompting!
    # Chalo prompt engineering sikhtee hain! 🎨
    main()
