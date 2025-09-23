#!/usr/bin/env python3
"""
📊 04_output_parsers.py - Making Sense of AI Responses!

Learning Outcomes:
After completing this tutorial, you'll understand:
- Why output parsers are essential for real applications
- How to structure AI responses in useful formats (JSON, lists, etc.)
- Different types of output parsers and when to use them
- How to handle parsing errors gracefully
- Building custom parsers for specific use cases
- Combining parsers with prompts for powerful workflows

Output parsers are like translators - they convert AI's free-form text into structured data! 🌐
Created by your dost from tier-2 city who loves clean, structured data 📊
"""

import os
import json
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

try:
    from langchain.output_parsers import (
        PydanticOutputParser, 
        CommaSeparatedListOutputParser,
        StructuredOutputParser,
        ResponseSchema,
        OutputFixingParser
    )
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema import HumanMessage
    from pydantic import BaseModel, Field
    print("✅ All imports successful! Ready to parse AI outputs like a pro!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Run: pip install -r requirements.txt")
    exit(1)


class OutputParsingMaster:
    """
    Your friendly guide to structured AI responses!
    
    Think of output parsers as your AI response organizer
    Raw AI output = Messy room, Parsed output = Organized room! 🏠
    """
    
    def __init__(self):
        """Initialize our parsing master"""
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ Please set OPENAI_API_KEY in your .env file!")
            exit(1)
        
        self.chat = ChatOpenAI(
            temperature=0.3,  # Lower temperature for more consistent structured output
            model_name="gpt-3.5-turbo"
        )
        
        print("📊 Output Parsing Master initialized! Let's structure some AI responses!")
    
    def why_output_parsers_matter(self):
        """
        Explain why output parsers are crucial for real applications
        
        Imagine ordering chai and getting a random cup vs getting exactly what you ordered!
        That's the difference parsers make! ☕
        """
        print("\n" + "="*60)
        print("🤔 WHY OUTPUT PARSERS MATTER - The Real World Problem")
        print("="*60)
        
        print("🚨 The Problem with Raw AI Output:")
        
        # Show unparsed output example
        raw_prompt = "List 5 popular programming languages with their primary use cases"
        
        try:
            raw_response = self.chat([HumanMessage(content=raw_prompt)])
            print(f"\n📝 Raw AI Response:\n{raw_response.content}")
            print("\n❌ Problems with this response:")
            print("   • Hard to extract individual languages programmatically")
            print("   • Inconsistent formatting")
            print("   • Can't easily use in databases or APIs")
            print("   • Difficult to validate completeness")
            
        except Exception as e:
            print(f"❌ Error getting raw response: {e}")
        
        benefits = """
        ✅ Benefits of Parsed Output:
        
        1. 🏗️ STRUCTURED DATA
           • Easy to store in databases
           • Simple to iterate over in loops
           • Consistent format every time
        
        2. 🔍 VALIDATION
           • Ensure all required fields are present
           • Check data types automatically
           • Catch errors early
        
        3. 🔗 INTEGRATION
           • Works seamlessly with APIs
           • Easy to pass to other functions
           • JSON for web applications
        
        4. 🎯 RELIABILITY
           • Predictable output structure
           • Handles edge cases
           • Graceful error handling
        
        Think of it like this:
        Raw output = Getting ingredients thrown at you
        Parsed output = Getting a neatly organized recipe! 👨‍🍳
        """
        
        print(benefits)
    
    def comma_separated_list_parser_demo(self):
        """
        Demonstrate the simplest parser - CommaSeparatedListOutputParser
        
        This is like asking your friend to list things separated by commas
        Perfect for simple lists! 📝
        """
        print("\n" + "="*60)
        print("📝 COMMA SEPARATED LIST PARSER - Simple Lists Made Easy")
        print("="*60)
        
        # Create the parser
        list_parser = CommaSeparatedListOutputParser()
        
        print("🔧 Parser created for comma-separated lists")
        print(f"📋 Format instructions: {list_parser.get_format_instructions()}")
        
        # Create prompt with parser instructions
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that provides accurate information."),
            ("human", """List {count} {category} that are popular in India.
            
            {format_instructions}""")
        ])
        
        test_cases = [
            {"count": "5", "category": "street foods"},
            {"count": "4", "category": "programming languages"},
            {"count": "6", "category": "Bollywood actors"}
        ]
        
        try:
            for i, test_case in enumerate(test_cases, 1):
                print(f"\n🧪 Test {i}: {test_case['count']} {test_case['category']}")
                
                # Format the prompt with parser instructions
                formatted_prompt = prompt_template.format_messages(
                    format_instructions=list_parser.get_format_instructions(),
                    **test_case
                )
                
                # Get AI response
                response = self.chat(formatted_prompt)
                print(f"🤖 Raw response: {response.content}")
                
                # Parse the response
                parsed_list = list_parser.parse(response.content)
                print(f"✅ Parsed list: {parsed_list}")
                print(f"📊 Type: {type(parsed_list)}, Length: {len(parsed_list)}")
                print("-" * 40)
                
        except Exception as e:
            print(f"❌ Error in list parser demo: {e}")
    
    def pydantic_parser_demo(self):
        """
        Demonstrate PydanticOutputParser for complex structured data
        
        Pydantic is like having a strict teacher who ensures everything is in the right format
        Perfect for complex data with validation! 🎯
        """
        print("\n" + "="*60)
        print("🏗️ PYDANTIC PARSER - Structured Data with Validation")
        print("="*60)
        
        # Define a Pydantic model for a programming language
        class ProgrammingLanguage(BaseModel):
            name: str = Field(description="Name of the programming language")
            year_created: int = Field(description="Year the language was first released")
            primary_use: str = Field(description="Main use case or domain")
            difficulty_level: str = Field(description="Beginner, Intermediate, or Advanced")
            popularity_score: int = Field(description="Popularity score from 1-10")
            
            def __str__(self):
                return f"{self.name} ({self.year_created}) - {self.primary_use} [Difficulty: {self.difficulty_level}]"
        
        # Create the parser
        pydantic_parser = PydanticOutputParser(pydantic_object=ProgrammingLanguage)
        
        print("🏗️ Pydantic model created for Programming Language data")
        print("📋 Required fields: name, year_created, primary_use, difficulty_level, popularity_score")
        
        # Create prompt with parser instructions
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a knowledgeable programming expert with accurate historical data."),
            ("human", """Provide information about the {language} programming language.
            
            {format_instructions}""")
        ])
        
        test_languages = ["Python", "JavaScript", "Go"]
        
        try:
            for language in test_languages:
                print(f"\n🔍 Analyzing: {language}")
                
                # Format prompt with parser instructions
                formatted_prompt = prompt_template.format_messages(
                    language=language,
                    format_instructions=pydantic_parser.get_format_instructions()
                )
                
                # Get AI response
                response = self.chat(formatted_prompt)
                print(f"🤖 Raw JSON response: {response.content[:150]}...")
                
                # Parse the response into Pydantic object
                parsed_language = pydantic_parser.parse(response.content)
                print(f"✅ Parsed object: {parsed_language}")
                print(f"📊 Type: {type(parsed_language)}")
                print(f"🔍 Individual fields:")
                print(f"   Name: {parsed_language.name}")
                print(f"   Year: {parsed_language.year_created}")
                print(f"   Use: {parsed_language.primary_use}")
                print(f"   Difficulty: {parsed_language.difficulty_level}")
                print(f"   Popularity: {parsed_language.popularity_score}/10")
                print("-" * 50)
                
        except Exception as e:
            print(f"❌ Error in Pydantic parser demo: {e}")
    
    def structured_output_parser_demo(self):
        """
        Demonstrate StructuredOutputParser for flexible schemas
        
        This is like having a form with specific fields you want filled
        More flexible than Pydantic but still structured! 📋
        """
        print("\n" + "="*60)
        print("📋 STRUCTURED OUTPUT PARSER - Flexible Schemas")
        print("="*60)
        
        # Define response schema
        response_schemas = [
            ResponseSchema(name="dish_name", description="Name of the Indian dish"),
            ResponseSchema(name="region", description="Which region of India it's from"),
            ResponseSchema(name="main_ingredients", description="List of main ingredients"),
            ResponseSchema(name="spice_level", description="Mild, Medium, or Spicy"),
            ResponseSchema(name="vegetarian", description="Yes or No"),
            ResponseSchema(name="preparation_time", description="Time needed to prepare in minutes")
        ]
        
        # Create the parser
        structured_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        
        print("📋 Structured parser created for Indian dish information")
        print("🏗️ Schema fields: dish_name, region, main_ingredients, spice_level, vegetarian, preparation_time")
        
        # Create prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert on Indian cuisine with detailed knowledge of regional dishes."),
            ("human", """Provide detailed information about the Indian dish: {dish_name}
            
            {format_instructions}""")
        ])
        
        test_dishes = ["Butter Chicken", "Masala Dosa", "Rajma Chawal"]
        
        try:
            for dish in test_dishes:
                print(f"\n🍽️ Analyzing dish: {dish}")
                
                # Format prompt with schema instructions
                formatted_prompt = prompt_template.format_messages(
                    dish_name=dish,
                    format_instructions=structured_parser.get_format_instructions()
                )
                
                # Get AI response
                response = self.chat(formatted_prompt)
                print(f"🤖 Raw structured response: {response.content}")
                
                # Parse the response
                parsed_dish = structured_parser.parse(response.content)
                print(f"✅ Parsed dictionary: {parsed_dish}")
                print(f"📊 Type: {type(parsed_dish)}")
                print("🔍 Easy access to fields:")
                for key, value in parsed_dish.items():
                    print(f"   {key}: {value}")
                print("-" * 50)
                
        except Exception as e:
            print(f"❌ Error in structured parser demo: {e}")
    
    def error_handling_and_fixing_parser_demo(self):
        """
        Demonstrate error handling and OutputFixingParser
        
        Sometimes AI doesn't follow instructions perfectly - that's where fixing parsers help!
        It's like having a friend who corrects your mistakes automatically 🛠️
        """
        print("\n" + "="*60)
        print("🛠️ ERROR HANDLING & FIXING PARSERS - Making AI Reliable")
        print("="*60)
        
        # Create a simple list parser for demonstration
        list_parser = CommaSeparatedListOutputParser()
        
        print("🧪 Testing parser with potentially problematic responses:")
        
        # Simulate different types of problematic AI responses
        problematic_responses = [
            "Python, JavaScript, Java, C++, Go",  # Good response
            "1. Python\n2. JavaScript\n3. Java",  # Numbered list instead of comma-separated
            "Here are the languages: Python; JavaScript; Java; C++",  # Semicolon separated
            "Python, JavaScript, and Java are great languages for beginners.",  # Extra text
        ]
        
        print("\n📝 Testing different response formats:")
        
        for i, response in enumerate(problematic_responses, 1):
            print(f"\n🔸 Test {i}: {response}")
            
            try:
                # Try to parse with regular parser
                parsed_result = list_parser.parse(response)
                print(f"✅ Parsed successfully: {parsed_result}")
                
            except Exception as e:
                print(f"❌ Regular parser failed: {e}")
                
                # Now try with OutputFixingParser
                print("🔧 Trying with OutputFixingParser...")
                
                try:
                    # Create a fixing parser that uses the chat model to fix issues
                    fixing_parser = OutputFixingParser.from_llm(
                        parser=list_parser,
                        llm=self.chat
                    )
                    
                    # The fixing parser will ask the LLM to correct the format
                    fixed_result = fixing_parser.parse(response)
                    print(f"✅ Fixed and parsed: {fixed_result}")
                    
                except Exception as fix_error:
                    print(f"❌ Even fixing parser failed: {fix_error}")
        
        # Show best practices for error handling
        print("\n🎯 Best Practices for Parser Error Handling:")
        
        best_practices = """
        1. 🔍 VALIDATE INPUT
           • Check if response looks roughly correct before parsing
           • Look for expected keywords or patterns
        
        2. 🛡️ TRY-CATCH BLOCKS
           • Always wrap parser.parse() in try-except blocks
           • Provide meaningful error messages to users
        
        3. 🔄 FALLBACK STRATEGIES
           • Have backup parsers for different formats
           • Use OutputFixingParser for automatic correction
           • Implement manual fallbacks for critical applications
        
        4. 📊 LOGGING & MONITORING
           • Log parsing failures for analysis
           • Monitor parser success rates
           • Track which prompts cause parsing issues
        
        5. 🎯 IMPROVE PROMPTS
           • Use clearer format instructions
           • Add examples in prompts
           • Lower temperature for more consistent output
        """
        
        print(best_practices)
    
    def custom_parser_demo(self):
        """
        Show how to create custom parsers for specific use cases
        
        Sometimes you need a parser that's tailored to your exact needs
        Let's build one from scratch! 🔨
        """
        print("\n" + "="*60)
        print("🔨 CUSTOM PARSERS - Building Your Own Parser")
        print("="*60)
        
        class IndianPhoneNumberParser:
            """
            Custom parser to extract and validate Indian phone numbers from AI responses
            
            This parser specifically handles Indian phone number formats
            Like a bouncer who only lets in properly formatted phone numbers! 🚪
            """
            
            def get_format_instructions(self) -> str:
                return """Please provide Indian phone numbers in this exact format:
                +91-XXXXXXXXXX (where X is a digit)
                
                Examples:
                +91-9876543210
                +91-8123456789
                
                If multiple numbers, separate with commas."""
            
            def parse(self, text: str) -> List[str]:
                """Parse and validate Indian phone numbers from text"""
                import re
                
                # Pattern for Indian phone numbers
                # +91 followed by 10 digits
                pattern = r'\+91-([6-9]\d{9})'
                
                # Find all matches
                matches = re.findall(pattern, text)
                
                if not matches:
                    # Try to find numbers without proper formatting
                    loose_pattern = r'(\+91\s*[-\s]?\s*[6-9]\d{9})'
                    loose_matches = re.findall(loose_pattern, text)
                    
                    if loose_matches:
                        # Try to fix the format
                        fixed_numbers = []
                        for match in loose_matches:
                            # Clean and reformat
                            digits_only = re.sub(r'[^\d]', '', match)[2:]  # Remove +91 and all non-digits
                            if len(digits_only) == 10 and digits_only[0] in '6789':
                                fixed_numbers.append(f"+91-{digits_only}")
                        
                        if fixed_numbers:
                            return fixed_numbers
                    
                    raise ValueError(f"No valid Indian phone numbers found in: {text}")
                
                # Return properly formatted numbers
                return [f"+91-{match}" for match in matches]
        
        # Test the custom parser
        custom_parser = IndianPhoneNumberParser()
        
        print("🏗️ Custom Indian Phone Number Parser created!")
        print(f"📋 Format instructions:\n{custom_parser.get_format_instructions()}")
        
        # Create prompt to test the parser
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that provides Indian contact information."),
            ("human", """Provide {count} sample Indian mobile phone numbers for {use_case}.
            
            {format_instructions}""")
        ])
        
        test_cases = [
            {"count": "3", "use_case": "food delivery services"},
            {"count": "2", "use_case": "customer support hotlines"}
        ]
        
        try:
            for test_case in test_cases:
                print(f"\n🧪 Test case: {test_case['count']} numbers for {test_case['use_case']}")
                
                # Format prompt
                formatted_prompt = prompt_template.format_messages(
                    format_instructions=custom_parser.get_format_instructions(),
                    **test_case
                )
                
                # Get AI response
                response = self.chat(formatted_prompt)
                print(f"🤖 AI response: {response.content}")
                
                # Parse with custom parser
                parsed_numbers = custom_parser.parse(response.content)
                print(f"✅ Parsed phone numbers: {parsed_numbers}")
                print(f"📊 Found {len(parsed_numbers)} valid numbers")
                
                # Validate each number
                for number in parsed_numbers:
                    print(f"   📞 {number} - Valid format!")
                
                print("-" * 50)
                
        except Exception as e:
            print(f"❌ Error in custom parser demo: {e}")
    
    def real_world_example(self):
        """
        A complete real-world example combining multiple concepts
        
        Let's build a restaurant review analyzer - practical and useful! 🍽️
        """
        print("\n" + "="*60)
        print("🌟 REAL-WORLD EXAMPLE - Restaurant Review Analyzer")
        print("="*60)
        
        # Define Pydantic model for restaurant review analysis
        class RestaurantReviewAnalysis(BaseModel):
            restaurant_name: str = Field(description="Name of the restaurant")
            overall_sentiment: str = Field(description="Positive, Negative, or Neutral")
            rating_prediction: int = Field(description="Predicted rating from 1-5 stars")
            food_quality: str = Field(description="Excellent, Good, Average, or Poor")
            service_quality: str = Field(description="Excellent, Good, Average, or Poor")
            key_positives: List[str] = Field(description="List of positive aspects mentioned")
            key_negatives: List[str] = Field(description="List of negative aspects mentioned")
            cuisine_type: str = Field(description="Type of cuisine (e.g., Indian, Chinese, Italian)")
            would_recommend: bool = Field(description="Whether the reviewer would recommend this restaurant")
            
        # Create parser
        review_parser = PydanticOutputParser(pydantic_object=RestaurantReviewAnalysis)
        
        print("🏗️ Restaurant Review Analyzer created!")
        print("📊 Analysis includes: sentiment, rating, food/service quality, pros/cons, recommendations")
        
        # Create comprehensive prompt
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert restaurant review analyzer with deep understanding of customer sentiments.
            Analyze reviews carefully and extract structured insights that would be valuable for restaurant owners."""),
            ("human", """Analyze this restaurant review and provide structured insights:
            
            Review: "{review_text}"
            
            {format_instructions}""")
        ])
        
        # Sample reviews to analyze
        sample_reviews = [
            """I visited Spice Garden last night with my family. The butter chicken was absolutely delicious and the naan was fresh and warm. Service was a bit slow but the staff was very polite. The ambiance was nice with good music. Overall had a great time and would definitely come back. Worth the money!""",
            
            """Terrible experience at Mumbai Express. The food took 45 minutes to arrive and when it did, the biryani was cold and tasteless. The waiter was rude when we complained. Overpriced for such poor quality. Will never visit again and wouldn't recommend to anyone."""
        ]
        
        try:
            for i, review in enumerate(sample_reviews, 1):
                print(f"\n📝 Analyzing Review {i}:")
                print(f"Review text: {review[:100]}...")
                
                # Format prompt with review
                formatted_prompt = analysis_prompt.format_messages(
                    review_text=review,
                    format_instructions=review_parser.get_format_instructions()
                )
                
                # Get AI analysis
                response = self.chat(formatted_prompt)
                print(f"\n🤖 Raw analysis: {response.content[:200]}...")
                
                # Parse the analysis
                analysis = review_parser.parse(response.content)
                print(f"\n✅ Structured Analysis:")
                print(f"   🏪 Restaurant: {analysis.restaurant_name}")
                print(f"   😊 Sentiment: {analysis.overall_sentiment}")
                print(f"   ⭐ Predicted Rating: {analysis.rating_prediction}/5")
                print(f"   🍽️ Food Quality: {analysis.food_quality}")
                print(f"   👥 Service Quality: {analysis.service_quality}")
                print(f"   ✅ Positives: {', '.join(analysis.key_positives)}")
                print(f"   ❌ Negatives: {', '.join(analysis.key_negatives)}")
                print(f"   🍜 Cuisine: {analysis.cuisine_type}")
                print(f"   👍 Would Recommend: {analysis.would_recommend}")
                
                print("-" * 60)
                
        except Exception as e:
            print(f"❌ Error in real-world example: {e}")


def main():
    """
    Main function to run all output parsing demonstrations
    
    Chalo structured data ki duniya mein chalte hain! 📊
    """
    print("📊 Welcome to Output Parsing Mastery!")
    print("👨‍💻 Your data-loving mentor from tier-2 city is here!")
    print("🎯 Let's turn messy AI responses into beautiful structured data!\n")
    
    # Initialize our parsing master
    master = OutputParsingMaster()
    
    try:
        # Run all demonstrations
        master.why_output_parsers_matter()
        master.comma_separated_list_parser_demo()
        master.pydantic_parser_demo()
        master.structured_output_parser_demo()
        master.error_handling_and_fixing_parser_demo()
        master.custom_parser_demo()
        master.real_world_example()
        
        # Final graduation message
        print("\n" + "="*60)
        print("🎊 CONGRATULATIONS! You're now an Output Parsing Expert!")
        print("="*60)
        print("✅ You understand why structured output matters")
        print("✅ You can use different types of parsers")
        print("✅ You can handle parsing errors gracefully")
        print("✅ You can create custom parsers for specific needs")
        print("✅ You've built a real-world application!")
        
        print("\n🎓 Your New Superpowers:")
        print("   → Convert AI text into usable data structures")
        print("   → Validate AI responses automatically")
        print("   → Build reliable AI applications")
        print("   → Handle edge cases and errors")
        print("   → Create custom solutions for unique problems")
        
        print("\n📚 What's Next?")
        print("   → 05_chains_basics.py - Learn to chain operations together")
        print("   → Build your own parsing solutions")
        print("   → Integrate parsers with databases")
        print("   → Create APIs that return structured AI data")
        
        print("\n💡 Key Takeaways:")
        print("   • Always use parsers in production applications")
        print("   • Choose the right parser for your use case")
        print("   • Handle errors gracefully with fallbacks")
        print("   • Custom parsers give you maximum control")
        
        print("\n🚀 You're ready for intermediate LangChain concepts!")
        
    except KeyboardInterrupt:
        print("\n\n👋 Tutorial interrupted! Your parsed data is waiting for you!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Check your internet connection and API key")


if __name__ == "__main__":
    # Let's master structured AI responses!
    # Chalo structured data banate hain! 📊
    main()
