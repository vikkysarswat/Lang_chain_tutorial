#!/usr/bin/env python3
"""
🔗 05_chains_basics.py - Chaining Operations Like a Boss!

Learning Outcomes:
After completing this tutorial, you'll understand:
- What are chains and why they're powerful
- Different types of chains and their use cases
- How to create simple and complex chains
- Sequential chains for multi-step workflows
- Using chains with different input/output types
- Error handling in chain operations
- Building reusable chain components

Chains are like assembly lines - each step builds upon the previous one! 🏭
Created by your chain-loving friend from tier-2 city 🔗
"""

import os
from dotenv import load_dotenv

load_dotenv()

try:
    from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
    from langchain.chains import TransformChain
    from langchain.chat_models import ChatOpenAI
    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate, ChatPromptTemplate
    from langchain.output_parsers import CommaSeparatedListOutputParser
    from langchain.schema import BaseOutputParser
    import json
    print("✅ All imports successful! Ready to chain operations together!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Run: pip install -r requirements.txt")
    exit(1)


class ChainMaster:
    """
    Your friendly chain-building mentor!
    
    Think of chains as a series of connected operations
    Like making chai: boil water → add tea → add milk → add sugar → serve! ☕
    Each step depends on the previous one!
    """
    
    def __init__(self):
        """Initialize our chain master"""
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ Please set OPENAI_API_KEY in your .env file!")
            exit(1)
        
        # Initialize different model types for various chain examples
        self.chat_model = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
        self.llm = OpenAI(temperature=0.7)
        
        print("🔗 Chain Master initialized! Let's build some awesome chains!")
    
    def why_chains_matter(self):
        """
        Explain the concept and importance of chains in LangChain
        
        Chains are the backbone of complex AI applications
        Single LLM call = Making instant noodles
        Chain = Cooking a full 7-course meal! 🍽️
        """
        print("\n" + "="*60)
        print("🤔 WHY CHAINS MATTER - From Simple to Sophisticated")
        print("="*60)
        
        explanation = """
        🔗 What are Chains?
        
        A chain is a sequence of calls to LLMs, tools, or data processing steps.
        Instead of one AI call, you create a pipeline of operations.
        
        🏭 Real-World Analogy:
        Imagine a food processing factory:
        
        Raw Input → Cleaning → Cutting → Cooking → Packaging → Final Product
        
        In LangChain:
        User Query → Analysis → Processing → Generation → Formatting → Response
        
        ✨ Why Use Chains?
        
        1. 🎯 COMPLEX TASKS
           • Break down big problems into smaller steps
           • Each step focuses on one specific task
           • Easier to debug and maintain
        
        2. 🔄 REUSABILITY
           • Create once, use multiple times
           • Mix and match different chain components
           • Build libraries of common operations
        
        3. 🛡️ RELIABILITY
           • Handle errors at specific steps
           • Add validation between steps
           • Recover from failures gracefully
        
        4. 🎨 FLEXIBILITY
           • Different models for different tasks
           • Conditional logic in chains
           • Dynamic behavior based on input
        
        🌟 Types of Chains:
        
        • LLMChain: Basic prompt + LLM combination
        • SimpleSequentialChain: Output of one → Input of next
        • SequentialChain: Multiple inputs/outputs between steps
        • TransformChain: Data processing without LLM calls
        • Custom Chains: Your own creative combinations!
        
        Think of it as:
        Single LLM = One-person band 🎸
        Chains = Full orchestra 🎼
        """
        
        print(explanation)
    
    def basic_llm_chain_demo(self):
        """
        Demonstrate the most basic chain - LLMChain
        
        LLMChain is like having a reliable friend who always responds in the same way
        Give it a template, it fills in the blanks and asks AI! 🤝
        """
        print("\n" + "="*60)
        print("🔗 BASIC LLM CHAIN - Your First Chain Building Block")
        print("="*60)
        
        print("🏗️ Building a simple LLMChain...")
        
        # Create a prompt template
        prompt_template = PromptTemplate(
            input_variables=["product", "target_audience"],
            template="""You are a creative marketing expert from India who understands local culture.
            
            Create an engaging marketing tagline for {product} targeting {target_audience}.
            
            The tagline should be:
            - Catchy and memorable
            - Culturally relevant to India
            - Appealing to the target audience
            - Maximum 10 words
            
            Tagline:"""
        )
        
        # Create the LLMChain
        marketing_chain = LLMChain(
            llm=self.chat_model,
            prompt=prompt_template,
            output_key="tagline"  # Name the output for clarity
        )
        
        print("✅ LLMChain created successfully!")
        print(f"📋 Input variables: {marketing_chain.input_keys}")
        print(f"📤 Output variables: {marketing_chain.output_keys}")
        
        # Test the chain with different inputs
        test_cases = [
            {
                "product": "Online food delivery app",
                "target_audience": "busy professionals in tier-1 cities"
            },
            {
                "product": "Organic farming course",
                "target_audience": "young entrepreneurs in rural areas"
            },
            {
                "product": "AI coding bootcamp",
                "target_audience": "college students and career changers"
            }
        ]
        
        try:
            for i, test_case in enumerate(test_cases, 1):
                print(f"\n🧪 Test {i}: {test_case['product']}")
                print(f"🎯 Target: {test_case['target_audience']}")
                
                # Run the chain
                result = marketing_chain.run(test_case)
                print(f"🎨 Generated tagline: {result.strip()}")
                print("-" * 40)
                
        except Exception as e:
            print(f"❌ Error in basic chain demo: {e}")
    
    def simple_sequential_chain_demo(self):
        """
        Demonstrate SimpleSequentialChain for linear workflows
        
        Think of this as a relay race - output of one step becomes input of the next
        Perfect for workflows where each step builds on the previous! 🏃‍♀️➡️🏃‍♂️
        """
        print("\n" + "="*60)
        print("🔄 SIMPLE SEQUENTIAL CHAIN - Building Linear Workflows")
        print("="*60)
        
        print("🏗️ Building a 3-step blog writing chain...")
        
        # Step 1: Generate blog outline
        outline_prompt = PromptTemplate(
            input_variables=["topic"],
            template="""You are a skilled content strategist.
            
            Create a detailed blog post outline for: {topic}
            
            The outline should include:
            - Catchy title
            - 5-7 main points
            - Brief description for each point
            - Target word count
            
            Make it engaging and informative for Indian readers."""
        )
        
        outline_chain = LLMChain(
            llm=self.chat_model,
            prompt=outline_prompt
        )
        
        # Step 2: Write the introduction
        intro_prompt = PromptTemplate(
            input_variables=["outline"],
            template="""You are a skilled blog writer with a conversational style.
            
            Based on this blog outline, write an engaging introduction (150-200 words):
            
            {outline}
            
            The introduction should:
            - Hook the reader immediately
            - Set the context
            - Preview what's coming
            - Use a friendly, conversational tone"""
        )
        
        intro_chain = LLMChain(
            llm=self.chat_model,
            prompt=intro_prompt
        )
        
        # Step 3: Add a compelling conclusion
        conclusion_prompt = PromptTemplate(
            input_variables=["intro"],
            template="""You are a content expert who writes powerful conclusions.
            
            Based on this blog introduction, write a compelling conclusion (100-150 words):
            
            {intro}
            
            The conclusion should:
            - Summarize key takeaways
            - Include a call to action
            - End with an inspiring note
            - Encourage reader engagement"""
        )
        
        conclusion_chain = LLMChain(
            llm=self.chat_model,
            prompt=conclusion_prompt
        )
        
        # Combine all chains into a sequential chain
        blog_writing_chain = SimpleSequentialChain(
            chains=[outline_chain, intro_chain, conclusion_chain],
            verbose=True  # This shows the intermediate steps
        )
        
        print("✅ 3-step sequential chain created!")
        print("📝 Steps: Outline → Introduction → Conclusion")
        
        # Test the sequential chain
        test_topics = [
            "How AI is Revolutionizing Small Businesses in India",
            "The Future of Remote Work: Lessons from Tier-2 Cities"
        ]
        
        try:
            for i, topic in enumerate(test_topics, 1):
                print(f"\n🧪 Test {i}: {topic}")
                print("🔄 Running sequential chain...")
                
                # Run the entire chain
                final_result = blog_writing_chain.run(topic)
                
                print(f"✅ Final output (Conclusion):\n{final_result}")
                print("=" * 60)
                
        except Exception as e:
            print(f"❌ Error in sequential chain demo: {e}")
    
    def sequential_chain_demo(self):
        """
        Demonstrate SequentialChain with multiple inputs/outputs
        
        This is like a complex assembly line where different parts come together
        More flexible than SimpleSequentialChain! 🏭
        """
        print("\n" + "="*60)
        print("🎛️ SEQUENTIAL CHAIN - Complex Multi-Input/Output Workflows")
        print("="*60)
        
        print("🏗️ Building a startup idea validator chain...")
        
        # Chain 1: Market Analysis
        market_analysis_prompt = PromptTemplate(
            input_variables=["startup_idea", "target_market"],
            template="""You are a market research expert specializing in the Indian startup ecosystem.
            
            Analyze the market potential for this startup idea:
            Idea: {startup_idea}
            Target Market: {target_market}
            
            Provide analysis on:
            - Market size in India
            - Competition level
            - Growth potential
            - Key challenges
            
            Keep it concise and data-driven."""
        )
        
        market_chain = LLMChain(
            llm=self.chat_model,
            prompt=market_analysis_prompt,
            output_key="market_analysis"
        )
        
        # Chain 2: Technical Feasibility
        tech_feasibility_prompt = PromptTemplate(
            input_variables=["startup_idea"],
            template="""You are a senior technical architect with experience in Indian tech startups.
            
            Assess the technical feasibility of: {startup_idea}
            
            Consider:
            - Technology stack requirements
            - Development complexity (1-10 scale)
            - Required team size
            - Estimated development time
            - Technical risks
            
            Provide practical, actionable insights."""
        )
        
        tech_chain = LLMChain(
            llm=self.chat_model,
            prompt=tech_feasibility_prompt,
            output_key="tech_feasibility"
        )
        
        # Chain 3: Final Recommendation
        recommendation_prompt = PromptTemplate(
            input_variables=["startup_idea", "market_analysis", "tech_feasibility"],
            template="""You are a startup mentor who has guided many successful Indian startups.
            
            Based on the analysis below, provide a final recommendation:
            
            Startup Idea: {startup_idea}
            Market Analysis: {market_analysis}
            Technical Feasibility: {tech_feasibility}
            
            Provide:
            - Overall viability score (1-10)
            - Top 3 strengths
            - Top 3 concerns
            - Next steps recommendation
            - Timeline for MVP development
            
            Be honest but constructive in your feedback."""
        )
        
        recommendation_chain = LLMChain(
            llm=self.chat_model,
            prompt=recommendation_prompt,
            output_key="final_recommendation"
        )
        
        # Create the sequential chain with multiple inputs/outputs
        startup_validator = SequentialChain(
            chains=[market_chain, tech_chain, recommendation_chain],
            input_variables=["startup_idea", "target_market"],
            output_variables=["market_analysis", "tech_feasibility", "final_recommendation"],
            verbose=True
        )
        
        print("✅ Startup validator chain created!")
        print("📊 Input: idea + target market")
        print("📈 Output: market analysis + tech feasibility + final recommendation")
        
        # Test the complex chain
        test_startups = [
            {
                "startup_idea": "AI-powered regional language tutor app for government exam preparation",
                "target_market": "Students preparing for SSC, Banking, and Railway exams in tier-2/3 cities"
            },
            {
                "startup_idea": "Blockchain-based farmer-to-consumer direct marketplace",
                "target_market": "Urban consumers who want fresh produce and farmers seeking better prices"
            }
        ]
        
        try:
            for i, startup in enumerate(test_startups, 1):
                print(f"\n🧪 Test {i}: {startup['startup_idea'][:50]}...")
                print(f"🎯 Target: {startup['target_market'][:50]}...")
                print("🔄 Running comprehensive analysis...")
                
                # Run the complex chain
                results = startup_validator(startup)
                
                print("\n📊 ANALYSIS RESULTS:")
                print(f"\n📈 Market Analysis:\n{results['market_analysis'][:300]}...")
                print(f"\n💻 Technical Feasibility:\n{results['tech_feasibility'][:300]}...")
                print(f"\n🎯 Final Recommendation:\n{results['final_recommendation'][:300]}...")
                print("=" * 70)
                
        except Exception as e:
            print(f"❌ Error in sequential chain demo: {e}")
    
    def transform_chain_demo(self):
        """
        Demonstrate TransformChain for data processing without LLM calls
        
        Sometimes you just need to process data without calling AI
        Like washing vegetables before cooking - pure data transformation! 🥕➡️🥕✨
        """
        print("\n" + "="*60)
        print("⚙️ TRANSFORM CHAIN - Pure Data Processing Power")
        print("="*60)
        
        print("🔄 Creating data preprocessing chains...")
        
        def clean_text_transform(inputs: dict) -> dict:
            """Clean and normalize text data"""
            text = inputs["raw_text"]
            
            # Basic text cleaning
            cleaned = text.strip()
            cleaned = ' '.join(cleaned.split())  # Remove extra whitespace
            cleaned = cleaned.replace('\n', ' ')
            
            # Count words and characters
            word_count = len(cleaned.split())
            char_count = len(cleaned)
            
            return {
                "cleaned_text": cleaned,
                "word_count": word_count,
                "char_count": char_count,
                "is_long": word_count > 100
            }
        
        def extract_keywords_transform(inputs: dict) -> dict:
            """Extract simple keywords from text"""
            text = inputs["cleaned_text"].lower()
            
            # Simple keyword extraction (in real world, use proper NLP libraries)
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
            
            words = text.split()
            keywords = [word for word in words if len(word) > 3 and word not in common_words]
            
            # Get top 5 keywords by frequency
            from collections import Counter
            keyword_freq = Counter(keywords)
            top_keywords = [word for word, count in keyword_freq.most_common(5)]
            
            return {
                "keywords": top_keywords,
                "total_unique_words": len(set(words)),
                "keyword_density": len(keywords) / len(words) if words else 0
            }
        
        # Create transform chains
        text_cleaner = TransformChain(
            input_variables=["raw_text"],
            output_variables=["cleaned_text", "word_count", "char_count", "is_long"],
            transform=clean_text_transform
        )
        
        keyword_extractor = TransformChain(
            input_variables=["cleaned_text"],
            output_variables=["keywords", "total_unique_words", "keyword_density"],
            transform=extract_keywords_transform
        )
        
        # Create LLM chain for summary (only if text is long)
        summary_prompt = PromptTemplate(
            input_variables=["cleaned_text", "keywords"],
            template="""Summarize this text in 2-3 sentences, focusing on these key themes: {keywords}
            
            Text: {cleaned_text}
            
            Summary:"""
        )
        
        summary_chain = LLMChain(
            llm=self.chat_model,
            prompt=summary_prompt,
            output_key="summary"
        )
        
        # Combine transform chains with LLM chain
        text_processor = SequentialChain(
            chains=[text_cleaner, keyword_extractor, summary_chain],
            input_variables=["raw_text"],
            output_variables=["cleaned_text", "word_count", "keywords", "summary"],
            verbose=True
        )
        
        print("✅ Text processing pipeline created!")
        print("🔄 Steps: Clean text → Extract keywords → Generate summary")
        
        # Test the transform chains
        test_texts = [
            """
            Artificial Intelligence is revolutionizing the way businesses operate in India. 
            From small startups in Bangalore to large enterprises in Mumbai, AI adoption is 
            accelerating rapidly. Machine learning algorithms are helping companies optimize 
            their operations, improve customer experience, and drive innovation. 
            
            The Indian government's Digital India initiative has also played a crucial role 
            in promoting AI adoption. Companies are investing heavily in AI talent and 
            infrastructure. However, challenges remain in terms of data privacy, ethical AI 
            development, and skill gaps in the workforce.
            
            Looking ahead, AI is expected to contribute significantly to India's GDP growth 
            and create new job opportunities in emerging technologies.
            """,
            
            "Short text about AI in India for testing."
        ]
        
        try:
            for i, text in enumerate(test_texts, 1):
                print(f"\n🧪 Test {i}: Processing text ({len(text)} characters)")
                
                # Process the text through the pipeline
                results = text_processor({"raw_text": text})
                
                print(f"📊 Analysis Results:")
                print(f"   Word Count: {results.get('word_count', 'N/A')}")
                print(f"   Keywords: {results.get('keywords', 'N/A')}")
                print(f"   Summary: {results.get('summary', 'N/A')[:200]}...")
                print("-" * 50)
                
        except Exception as e:
            print(f"❌ Error in transform chain demo: {e}")
    
    def error_handling_in_chains_demo(self):
        """
        Demonstrate error handling strategies in chains
        
        Chains can break at any step - let's learn to handle failures gracefully!
        Like having backup plans when cooking - if one dish burns, make another! 🍳
        """
        print("\n" + "="*60)
        print("🛡️ ERROR HANDLING IN CHAINS - Building Robust Pipelines")
        print("="*60)
        
        print("🎯 Common chain failures and solutions:")
        
        error_scenarios = """
        🚨 Common Chain Errors:
        
        1. 🔑 API KEY ISSUES
           • Invalid or expired API keys
           • Rate limit exceeded
           • Network connectivity problems
        
        2. 📝 PARSING ERRORS
           • LLM doesn't follow expected format
           • Missing required fields in output
           • Unexpected response structure
        
        3. 🔗 CHAIN LOGIC ERRORS
           • Missing input variables
           • Type mismatches between chain steps
           • Circular dependencies
        
        4. 💰 COST OVERRUNS
           • Chains making too many API calls
           • Infinite loops in chain logic
           • Expensive models for simple tasks
        
        🛠️ Solutions:
        
        ✅ Try-Catch Blocks: Wrap chain execution
        ✅ Fallback Chains: Alternative paths when primary fails
        ✅ Input Validation: Check inputs before processing
        ✅ Output Validation: Verify outputs before next step
        ✅ Timeout Settings: Prevent infinite waits
        ✅ Logging: Track chain execution for debugging
        """
        
        print(error_scenarios)
        
        # Demonstrate error handling with a potentially failing chain
        print("\n🧪 Testing chain with error handling:")
        
        # Create a chain that might fail
        risky_prompt = PromptTemplate(
            input_variables=["user_input"],
            template="Process this input and return exactly 3 items in JSON format: {user_input}"
        )
        
        risky_chain = LLMChain(
            llm=self.chat_model,
            prompt=risky_prompt
        )
        
        # Test inputs that might cause different types of failures
        test_inputs = [
            "List programming languages",  # Should work
            "",  # Empty input
            "x" * 1000,  # Very long input
            "Generate random data that doesn't make sense for the prompt"  # Confusing input
        ]
        
        for i, test_input in enumerate(test_inputs, 1):
            print(f"\n🧪 Test {i}: {test_input[:50]}{'...' if len(test_input) > 50 else ''}")
            
            try:
                # Add timeout and error handling
                result = risky_chain.run(test_input)
                print(f"✅ Success: {result[:100]}...")
                
            except Exception as e:
                error_type = type(e).__name__
                print(f"❌ Chain failed with {error_type}: {str(e)[:100]}...")
                
                # Implement fallback strategy
                print("🔄 Trying fallback approach...")
                try:
                    fallback_result = f"Processed input: {test_input[:50]}... (fallback response)"
                    print(f"🆘 Fallback success: {fallback_result}")
                except Exception as fallback_error:
                    print(f"💥 Even fallback failed: {fallback_error}")
        
        # Best practices summary
        best_practices = """
        
        🎓 Chain Error Handling Best Practices:
        
        1. 🔍 DEFENSIVE PROGRAMMING
           • Validate all inputs before processing
           • Check intermediate outputs
           • Use type hints and validation
        
        2. 🔄 GRACEFUL DEGRADATION
           • Provide fallback responses
           • Simplify chain logic when errors occur
           • Return partial results when possible
        
        3. 📊 MONITORING & LOGGING
           • Log all chain executions
           • Monitor success/failure rates
           • Track performance metrics
        
        4. 🛡️ USER EXPERIENCE
           • Provide meaningful error messages
           • Don't expose technical details to users
           • Offer alternative actions
        
        Remember: A good chain handles errors as gracefully as it handles success! 🌟
        """
        
        print(best_practices)


def main():
    """
    Main function to run all chain demonstrations
    
    Chalo chains ki duniya mein chalte hain! 🔗
    """
    print("🔗 Welcome to Chain Mastery!")
    print("👨‍💻 Your chain-building guru from tier-2 city is here!")
    print("🎯 Let's connect AI operations like a pro!\n")
    
    # Initialize our chain master
    master = ChainMaster()
    
    try:
        # Run all demonstrations
        master.why_chains_matter()
        master.basic_llm_chain_demo()
        master.simple_sequential_chain_demo()
        master.sequential_chain_demo()
        master.transform_chain_demo()
        master.error_handling_in_chains_demo()
        
        # Final graduation message
        print("\n" + "="*60)
        print("🎊 CONGRATULATIONS! You're now a Chain Building Expert!")
        print("="*60)
        print("✅ You understand what chains are and why they matter")
        print("✅ You can build basic LLMChains")
        print("✅ You can create sequential workflows")
        print("✅ You can handle complex multi-input/output chains")
        print("✅ You can process data without LLM calls")
        print("✅ You can handle errors in chain operations")
        
        print("\n🎓 Your New Chain Superpowers:")
        print("   → Build complex AI workflows step by step")
        print("   → Create reusable chain components")
        print("   → Handle different types of inputs and outputs")
        print("   → Process data efficiently with transforms")
        print("   → Build robust, error-resistant applications")
        
        print("\n📚 What's Next?")
        print("   → 06_memory_systems.py - Give your chains memory!")
        print("   → Build your own custom chains")
        print("   → Combine chains with tools and agents")
        print("   → Create production-ready chain applications")
        
        print("\n💡 Key Insights:")
        print("   • Chains make complex tasks manageable")
        print("   • Always plan your chain workflow before coding")
        print("   • Error handling is crucial for production chains")
        print("   • Transform chains are great for data processing")
        
        print("\n🚀 You're ready to build sophisticated AI applications!")
        
    except KeyboardInterrupt:
        print("\n\n👋 Chain building interrupted! Your chains are waiting for you!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Check your internet connection and API key")


if __name__ == "__main__":
    # Let's master the art of chaining AI operations!
    # Chalo chains banate hain! 🔗
    main()
