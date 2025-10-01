#!/usr/bin/env python3
"""
🤖 Agents and Tools in LangChain - AI That Can Use Tools!

Author: Senior AI Developer from Tier-2 City, India
Purpose: Building intelligent AI agents that can use tools and make decisions

Arre yaar! Ever wished your AI could not just chat but actually DO things?
Like searching the web, calling APIs, running calculations, or managing databases?
That's the power of Agents - AI that can think, plan, and use tools! 🛠️

Think of Agents like this:
- Regular AI = Smart person who can only talk
- AI Agent = Smart person with access to tools (calculator, phone, computer)
- They can decide WHICH tool to use and WHEN to use it
- Like having a personal assistant who can actually get things done!

What we'll master today:
1. Understanding Agents - How they think and decide
2. Built-in Tools - LangChain's ready-to-use tools
3. Custom Tools - Build your own tools
4. Agent Types - Zero-shot, Conversational, ReAct, Plan-and-Execute
5. Tool Calling - How agents interact with tools
6. Multi-Agent Systems - Multiple agents working together
7. Error Handling - Making agents robust
8. Real-world Examples - Practical Indian use cases
9. Production Deployment - Scale your agent systems

Real-world analogy: Agents are like having Jarvis from Iron Man who can
control your smart home, search information, and execute tasks autonomously! 🚀
"""

print("🤖 Agents and Tools Tutorial")
print("=" * 60)
print()
print("This tutorial teaches you to build AI agents that can use tools!")
print()
print("🎯 Key Concepts:")
print("   • Agents can make decisions about which tools to use")
print("   • Tools extend AI capabilities (search, calculate, query databases)")
print("   • ReAct pattern: Reason → Act → Observe → Repeat")
print("   • Custom tools for domain-specific functionality")
print()
print("🛠️ Example Tools:")
print("   • Calculator for math operations")
print("   • Web search for real-time information")
print("   • Database queries for data retrieval")
print("   • API calls for external services")
print("   • Custom business logic tools")
print()
print("🏗️ Building Your First Agent:")
print("""
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.chains import LLMMathChain

# Initialize LLM
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# Create a calculator tool
llm_math = LLMMathChain.from_llm(llm)
calculator_tool = Tool(
    name="Calculator",
    func=llm_math.run,
    description="Useful for mathematical calculations"
)

# Create a custom tool
def get_weather(location: str) -> str:
    # Your weather API logic here
    return f"Weather in {location}: Sunny, 28°C"

weather_tool = Tool(
    name="Weather",
    func=get_weather,
    description="Get current weather for a location"
)

# Initialize agent with tools
tools = [calculator_tool, weather_tool]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Use the agent
result = agent.invoke({
    "input": "What's 25 * 4, and what's the weather in Mumbai?"
})
print(result["output"])
""")

print("\n🎨 Creating Custom Tools:")
print("=" * 60)
print("""
# Method 1: Using @tool decorator
from langchain.tools import tool

@tool
def rupee_converter(amount: float) -> str:
    '''Convert INR to USD using current exchange rate'''
    usd = amount * 0.012  # Example rate
    return f"₹{amount} = ${usd:.2f}"

# Method 2: Using Tool class
from langchain.agents import Tool

def check_pincode(pincode: str) -> str:
    # Your pincode validation logic
    return f"Pincode {pincode} is valid"

pincode_tool = Tool(
    name="PincodeChecker",
    func=check_pincode,
    description="Validate Indian PIN codes"
)

# Method 3: Using BaseTool class (for complex tools)
from langchain.tools import BaseTool
from pydantic import Field

class IndianGSTCalculator(BaseTool):
    name: str = "GSTCalculator"
    description: str = "Calculate GST for Indian businesses"
    
    def _run(self, amount: float, rate: float) -> str:
        gst = (amount * rate) / 100
        total = amount + gst
        return f"Base: ₹{amount}, GST ({rate}%): ₹{gst}, Total: ₹{total}"
    
    async def _arun(self, amount: float, rate: float) -> str:
        return self._run(amount, rate)
""")

print("\n🔄 Agent Types Comparison:")
print("=" * 60)

agent_comparison = """
┌──────────────────────┬────────────────────────────────────────┐
│ Agent Type           │ Best For                               │
├──────────────────────┼────────────────────────────────────────┤
│ Zero-Shot ReAct      │ General-purpose, flexible tasks        │
│ Conversational ReAct │ Multi-turn conversations with memory   │
│ Plan-and-Execute     │ Complex multi-step workflows           │
│ Self-Ask with Search │ Research and information gathering     │
│ Structured Chat      │ Form filling, data extraction          │
└──────────────────────┴────────────────────────────────────────┘

🎯 Selection Guide:
   • Customer Support → Conversational ReAct (remembers context)
   • Data Analysis → Plan-and-Execute (systematic approach)
   • Research Assistant → Self-Ask with Search (breaks down questions)
   • Order Processing → Structured Chat (predictable outputs)
"""
print(agent_comparison)

print("\n💡 Production Best Practices:")
print("=" * 60)

best_practices = {
    "Safety": [
        "Set tool timeouts to prevent hanging",
        "Implement rate limiting on tool usage",
        "Validate all inputs before processing",
        "Set max iterations to prevent infinite loops",
        "Add fallback mechanisms for tool failures"
    ],
    "Monitoring": [
        "Track tool usage statistics",
        "Monitor success/failure rates",
        "Log all agent actions for debugging",
        "Measure response times and costs",
        "Collect user feedback for improvement"
    ],
    "Cost Optimization": [
        "Cache frequent queries and responses",
        "Use smaller models for simple tasks",
        "Batch API calls when possible",
        "Implement smart routing to cheaper models",
        "Set usage quotas per user"
    ],
    "Indian Context": [
        "Handle Hindi + English multilingual inputs",
        "Consider network latency in tier-2/3 cities",
        "Optimize for mobile-first users",
        "Include regional payment methods",
        "Comply with Indian data protection laws"
    ]
}

for category, practices in best_practices.items():
    print(f"\n{category}:")
    for practice in practices:
        print(f"   ✓ {practice}")

print("\n" + "=" * 60)
print()
print("🇮🇳 Real-world Indian Use Cases:")
print()

use_cases = [
    {
        "name": "Agricultural Advisory Agent",
        "tools": ["Weather API", "Crop Price DB", "Farming Knowledge Base"],
        "description": "Help farmers with weather forecasts, crop prices, and farming advice"
    },
    {
        "name": "Customer Support Agent",
        "tools": ["Order DB", "Inventory Check", "Shipping Tracker", "KB Search"],
        "description": "Handle e-commerce queries about orders, products, shipping"
    },
    {
        "name": "Financial Planning Assistant",
        "tools": ["SIP Calculator", "Tax Calculator", "Investment DB", "Market Data"],
        "description": "Provide financial advice, calculate returns, suggest investments"
    },
    {
        "name": "Healthcare Appointment Agent",
        "tools": ["Doctor Schedule", "Symptom Checker", "Hospital DB", "Insurance Validator"],
        "description": "Book appointments, check symptoms, verify insurance coverage"
    },
    {
        "name": "Educational Tutor Agent",
        "tools": ["Knowledge Base", "Practice Problems", "Progress Tracker", "Video Library"],
        "description": "Answer questions, provide practice problems, track student progress"
    }
]

for i, use_case in enumerate(use_cases, 1):
    print(f"{i}. {use_case['name']}")
    print(f"   Tools: {', '.join(use_case['tools'])}")
    print(f"   Use: {use_case['description']}")
    print()

print("=" * 60)
print()
print("🚀 Getting Started:")
print()
print("1. Install required packages:")
print("   pip install langchain langchain-openai")
print()
print("2. Set your API key:")
print("   export OPENAI_API_KEY='your-key-here'")
print()
print("3. Start with simple tools (calculator, string ops)")
print()
print("4. Test each tool independently")
print()
print("5. Create agent with 2-3 tools")
print()
print("6. Test with varied queries")
print()
print("7. Add more tools gradually")
print()
print("8. Monitor performance and iterate")
print()
print("=" * 60)
print()
print("💡 Key Takeaways:")
print()
print("✓ Agents make decisions about which tools to use")
print("✓ Tools extend AI capabilities beyond text generation")
print("✓ Start simple, add complexity gradually")
print("✓ Monitor tool usage and agent decisions")
print("✓ Handle errors gracefully with fallbacks")
print("✓ For Indian apps: multilingual support is crucial")
print("✓ Production agents need safety limits and monitoring")
print()
print("🎉 You're ready to build powerful AI agents!")
print("Next: Check out custom chains tutorial!")
print()
print("=" * 60)

# Demonstration function
def demonstrate_agent_concept():
    """
    Conceptual demonstration of how agents work
    
    This shows the thinking process without requiring API keys
    """
    print("\n🎭 Agent Reasoning Demonstration")
    print("=" * 60)
    
    # Simulate agent reasoning process
    user_query = "What's 25 * 4 and convert the result to dollars if I have that many rupees?"
    
    print(f"\n👤 User Query: {user_query}")
    print("\n🤖 Agent Thinking Process:")
    print("-" * 60)
    
    steps = [
        {
            "step": 1,
            "thought": "I need to calculate 25 * 4 first",
            "action": "Use Calculator tool",
            "action_input": "25 * 4",
            "observation": "Result: 100"
        },
        {
            "step": 2,
            "thought": "Now I have 100 rupees, need to convert to dollars",
            "action": "Use RupeeConverter tool",
            "action_input": "100",
            "observation": "₹100 = $1.20"
        },
        {
            "step": 3,
            "thought": "I have all the information needed to answer",
            "action": "Final Answer",
            "action_input": None,
            "observation": "25 * 4 = 100, and ₹100 equals approximately $1.20"
        }
    ]
    
    for step in steps:
        print(f"\nStep {step['step']}:")
        print(f"  💭 Thought: {step['thought']}")
        print(f"  🎬 Action: {step['action']}")
        if step['action_input']:
            print(f"  📥 Input: {step['action_input']}")
        print(f"  👁️ Observation: {step['observation']}")
    
    print("\n" + "-" * 60)
    print("✅ Final Answer: 25 * 4 = 100, and ₹100 equals approximately $1.20")
    print()
    print("This is how agents reason through problems step-by-step!")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running conceptual demonstration...")
    print("=" * 60)
    demonstrate_agent_concept()
    print("\n" + "=" * 60)
    print("Tutorial complete! Ready to build your own agents? 🚀")
    print("=" * 60)
