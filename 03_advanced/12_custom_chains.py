#!/usr/bin/env python3
"""
⛓️ Custom Chains in LangChain - Build Your Own AI Workflows!

Author: Senior AI Developer from Tier-2 City, India
Purpose: Creating custom chains for specialized AI workflows

Arre bhai! Want to build AI workflows that are perfectly tailored to YOUR
specific needs? Like creating a custom assembly line in a factory where each
step is designed exactly how you want it? That's Custom Chains! 🏭

Think of Custom Chains like this:
- Built-in chains = Ready-made recipes (good but generic)
- Custom chains = Your own secret recipe (perfect for your taste)
- You control every step, every prompt, every output
- Like being a chef who creates signature dishes!

What we'll master today:
1. Understanding Chain Architecture - How chains work internally
2. Simple Custom Chains - Basic building blocks
3. LLMChain Customization - Modify existing chains
4. Sequential Chains - Multi-step workflows
5. Router Chains - Conditional logic and branching
6. Transform Chains - Data preprocessing
7. Complex Custom Chains - Advanced patterns
8. Chain Composition - Combine multiple chains
9. Production Patterns - Real-world implementations

Real-world analogy: Custom chains are like building your own assembly line
where you decide exactly what happens at each station! 🏗️
"""

print("⛓️ Custom Chains Tutorial")
print("=" * 60)
print()
print("Build specialized AI workflows tailored to your needs!")
print()
print("🎯 What You'll Learn:")
print("   • How chains work under the hood")
print("   • Creating simple custom chains")
print("   • Sequential processing workflows")
print("   • Conditional branching with router chains")
print("   • Combining multiple chains")
print("   • Production-ready patterns")
print()
print("🏗️ Chain Building Blocks:")
print("""
┌─────────────────────────────────────────────────────┐
│              Custom Chain Architecture              │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Input → Preprocessing → LLM Call → Postprocessing │
│    ↓          ↓            ↓            ↓          │
│  Validate   Format      Generate     Parse/Format  │
│                                                     │
└─────────────────────────────────────────────────────┘
""")

print("\n📚 Basic Custom Chain Example:")
print("=" * 60)
print("""
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Initialize LLM
llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")

# Create custom prompt for Indian context
prompt_template = \"\"\"
You are an AI assistant helping Indian users.
Provide responses in a friendly, culturally aware manner.

User Question: {question}

Please provide a detailed answer considering Indian context.

Answer:\"\"\"

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["question"]
)

# Create custom chain
custom_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    output_key="answer"
)

# Use the chain
result = custom_chain.invoke({"question": "How to start a startup in India?"})
print(result["answer"])
""")

print("\n⚙️ Sequential Chains - Multi-Step Workflows:")
print("=" * 60)
print("""
from langchain.chains import SequentialChain

# Step 1: Extract key information
extract_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        template="Extract key points from: {text}\\n\\nKey Points:",
        input_variables=["text"]
    ),
    output_key="key_points"
)

# Step 2: Translate to Hindi
translate_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        template="Translate to Hindi: {key_points}\\n\\nHindi:",
        input_variables=["key_points"]
    ),
    output_key="hindi_translation"
)

# Step 3: Summarize
summarize_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        template="Summarize in 2 sentences: {hindi_translation}\\n\\nSummary:",
        input_variables=["hindi_translation"]
    ),
    output_key="final_summary"
)

# Combine into sequential chain
full_chain = SequentialChain(
    chains=[extract_chain, translate_chain, summarize_chain],
    input_variables=["text"],
    output_variables=["key_points", "hindi_translation", "final_summary"],
    verbose=True
)

# Use the sequential chain
result = full_chain.invoke({
    "text": "Your long English text here..."
})
""")

print("\n🔀 Router Chains - Conditional Logic:")
print("=" * 60)
print("""
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser

# Define specialized chains for different topics
chains = {
    "technology": LLMChain(...),  # Tech-focused responses
    "healthcare": LLMChain(...),  # Health-focused responses
    "education": LLMChain(...),   # Education-focused responses
    "general": LLMChain(...)      # General queries
}

# Router will decide which chain to use based on query
router_chain = MultiPromptChain(
    router_chain=LLMRouterChain(...),
    destination_chains=chains,
    default_chain=chains["general"]
)

# Router automatically selects appropriate chain
result = router_chain.invoke({"input": "How to prepare for JEE?"})
# This would route to "education" chain
""")

print("\n🎨 Custom Chain Class - Full Control:")
print("=" * 60)
print("""
from langchain.chains.base import Chain
from pydantic import Field
from typing import Dict, List

class IndianBusinessAdvisorChain(Chain):
    \"\"\"
    Custom chain for Indian business advice
    
    Features:
    - GST calculation
    - Compliance checking
    - Market analysis
    - Localized recommendations
    \"\"\"
    
    llm: Any = Field(...)
    output_key: str = "advice"
    
    @property
    def input_keys(self) -> List[str]:
        return ["business_type", "location", "query"]
    
    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]
    
    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        # Extract inputs
        business_type = inputs["business_type"]
        location = inputs["location"]
        query = inputs["query"]
        
        # Custom processing logic
        context = self._build_indian_context(business_type, location)
        
        # Create specialized prompt
        prompt = f\"\"\"
        Business Type: {business_type}
        Location: {location}
        Indian Context: {context}
        
        Query: {query}
        
        Provide detailed advice considering:
        - Indian regulations and compliance
        - Local market conditions
        - Cultural factors
        - GST implications
        - Regional specifics
        
        Advice:\"\"\"
        
        # Get LLM response
        response = self.llm.invoke(prompt)
        
        # Post-process if needed
        advice = self._format_advice(response.content)
        
        return {self.output_key: advice}
    
    def _build_indian_context(self, business_type: str, location: str) -> str:
        # Build context based on Indian business environment
        contexts = {
            "ecommerce": "Consider GST rates, shipping logistics, payment gateways",
            "restaurant": "FSSAI license, local tastes, festival seasons",
            "tech": "IT sector benefits, startup India scheme, talent availability"
        }
        return contexts.get(business_type, "General business context")
    
    def _format_advice(self, raw_advice: str) -> str:
        # Format advice with Indian business terminology
        return raw_advice

# Usage
advisor_chain = IndianBusinessAdvisorChain(llm=llm)
result = advisor_chain.invoke({
    "business_type": "restaurant",
    "location": "Indore",
    "query": "How to start a cloud kitchen?"
})
""")

print("\n🔄 Transform Chains - Data Preprocessing:")
print("=" * 60)
print("""
from langchain.chains import TransformChain

def preprocess_indian_text(inputs: Dict[str, str]) -> Dict[str, str]:
    \"\"\"
    Preprocess text for Indian context
    - Handle Hindi/English mixing
    - Normalize Indian names
    - Format currency (₹)
    - Handle regional variations
    \"\"\"
    text = inputs["text"]
    
    # Normalize common Hindi-English mixing
    text = text.replace("₹", "INR ")
    text = text.replace("lakh", "100000")
    text = text.replace("crore", "10000000")
    
    # Handle Hinglish
    hinglish_map = {
        "kya": "what",
        "kaise": "how",
        "kitna": "how much"
    }
    for hindi, english in hinglish_map.items():
        text = text.replace(hindi, english)
    
    return {"processed_text": text}

# Create transform chain
transform_chain = TransformChain(
    input_variables=["text"],
    output_variables=["processed_text"],
    transform=preprocess_indian_text
)

# Combine with LLM chain
preprocessing_chain = TransformChain(...)
llm_chain = LLMChain(...)

combined_chain = SequentialChain(
    chains=[preprocessing_chain, llm_chain],
    input_variables=["text"],
    output_variables=["final_output"]
)
""")

print("\n🇮🇳 Real-World Indian Use Cases:")
print("=" * 60)

use_cases = [
    {
        "name": "GST Invoice Generator",
        "chains": ["Extract → Calculate → Format → Generate"],
        "features": ["GST rate application", "GSTIN validation", "Invoice formatting"],
        "benefit": "Automated compliant invoice generation"
    },
    {
        "name": "Multilingual Customer Support",
        "chains": ["Detect Language → Route → Translate → Respond"],
        "features": ["Hindi/English detection", "Context-aware routing", "Translation"],
        "benefit": "Seamless multilingual support"
    },
    {
        "name": "Agricultural Advisor",
        "chains": ["Location → Weather → Crop → Advice"],
        "features": ["Regional crop suggestions", "Weather integration", "Market prices"],
        "benefit": "Localized farming guidance"
    },
    {
        "name": "Educational Content Generator",
        "chains": ["Topic → Syllabus Check → Generate → Format"],
        "features": ["CBSE/State board alignment", "Difficulty levels", "Hindi/English"],
        "benefit": "Customized educational content"
    },
    {
        "name": "Legal Document Assistant",
        "chains": ["Analyze → Extract → Summarize → Flag"],
        "features": ["Indian law references", "Compliance checking", "Risk flagging"],
        "benefit": "Simplified legal document review"
    }
]

for i, use_case in enumerate(use_cases, 1):
    print(f"\n{i}. {use_case['name']}")
    print(f"   Pipeline: {use_case['chains'][0]}")
    print(f"   Features: {', '.join(use_case['features'])}")
    print(f"   Benefit: {use_case['benefit']}")

print("\n\n💡 Chain Design Best Practices:")
print("=" * 60)

best_practices = {
    "Modularity": [
        "Keep each chain focused on one task",
        "Make chains reusable across projects",
        "Separate concerns (processing, formatting, validation)"
    ],
    "Error Handling": [
        "Add try-catch blocks in custom chains",
        "Provide fallback chains for failures",
        "Log errors with context for debugging"
    ],
    "Performance": [
        "Cache intermediate results when possible",
        "Use async chains for parallel processing",
        "Minimize LLM calls through smart caching"
    ],
    "Testing": [
        "Test each chain independently",
        "Use mock LLMs for unit testing",
        "Test with edge cases and invalid inputs"
    ],
    "Indian Context": [
        "Handle multilingual inputs gracefully",
        "Include regional and cultural context",
        "Format numbers in Indian system (lakhs, crores)",
        "Consider tier-2/3 city scenarios"
    ]
}

for category, practices in best_practices.items():
    print(f"\n{category}:")
    for practice in practices:
        print(f"   ✓ {practice}")

print("\n\n🚀 Production Deployment Checklist:")
print("=" * 60)

checklist = [
    "✓ Input validation at chain entry points",
    "✓ Error handling with graceful degradation",
    "✓ Logging for debugging and monitoring",
    "✓ Rate limiting to prevent abuse",
    "✓ Caching for frequently used chains",
    "✓ Performance monitoring and metrics",
    "✓ A/B testing for chain variations",
    "✓ Version control for prompt templates",
    "✓ Documentation for each custom chain",
    "✓ Security checks for sensitive data"
]

for item in checklist:
    print(f"   {item}")

print("\n\n📊 Chain Performance Optimization:")
print("=" * 60)
print("""
Optimization Strategy:
1. Profile your chains to identify bottlenecks
2. Cache expensive operations (embeddings, API calls)
3. Use batch processing for multiple similar requests
4. Implement async chains for parallel execution
5. Minimize prompt length while maintaining quality
6. Use streaming for real-time user feedback
7. Monitor token usage and optimize prompts
8. Consider smaller models for simple steps

Example Performance Gains:
• Caching: 50-90% response time reduction
• Async: 3-5x throughput increase
• Batch Processing: 60-80% cost reduction
• Prompt Optimization: 20-40% token savings
""")

print("\n=" * 60)
print()
print("🎉 Congratulations! You've mastered custom chains!")
print()
print("Key Takeaways:")
print("   ✓ Chains let you build specialized AI workflows")
print("   ✓ Sequential chains for multi-step processes")
print("   ✓ Router chains for conditional logic")
print("   ✓ Transform chains for data preprocessing")
print("   ✓ Custom chain classes for full control")
print("   ✓ Indian context matters - localize everything!")
print()
print("Next Steps:")
print("   1. Build a simple sequential chain")
print("   2. Add router for conditional logic")
print("   3. Create custom chain class for your use case")
print("   4. Test thoroughly with real data")
print("   5. Deploy with monitoring and logging")
print()
print("Ready for real-world projects? Check out:")
print("   → 04_real_world_projects/ for complete applications!")
print()
print("=" * 60)

if __name__ == "__main__":
    print("\n💡 This is an educational tutorial file.")
    print("Implement the patterns shown above in your projects!")
    print("\nHappy Building! 🚀🇮🇳")
