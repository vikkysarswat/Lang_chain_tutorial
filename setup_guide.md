# 🛠️ Setup Guide - Let's Get You Started!

*Arre bhai, don't worry! I'll walk you through everything step by step* 😊

## 📋 Prerequisites

Before we start our LangChain journey, make sure you have these installed:

### 1. Python (3.8 or higher)
```bash
python --version  # Should show 3.8+
```

If you don't have Python, download it from [python.org](https://python.org) 

### 2. Git (for cloning the repo)
```bash
git --version
```

## 🚀 Step-by-Step Setup

### Step 1: Clone the Repository
```bash
# Clone the repo to your local machine
git clone https://github.com/vikkysarswat/Lang_chain_tutorial.git

# Navigate to the project directory
cd Lang_chain_tutorial
```

### Step 2: Create Virtual Environment
*Trust me bro, virtual environments save lives! 🛡️*

```bash
# Create a new virtual environment
python -m venv langchain_env

# Activate it
# On Windows:
langchain_env\Scripts\activate

# On macOS/Linux:
source langchain_env/bin/activate

# You should see (langchain_env) in your terminal prompt
```

### Step 3: Install Dependencies
```bash
# Make sure you're in the project directory and venv is active
pip install --upgrade pip
pip install -r requirements.txt

# This might take 2-3 minutes, grab a chai! ☕
```

### Step 4: Set Up API Keys

#### 4.1 OpenAI API Key (Most Important!)
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Create an account or log in
3. Go to API Keys section
4. Create a new API key
5. Copy the key (starts with `sk-...`)

#### 4.2 Hugging Face Token (Optional but Recommended)
1. Go to [Hugging Face](https://huggingface.co/)
2. Create account and go to Settings → Access Tokens
3. Create a new token with 'read' permission

#### 4.3 Set Environment Variables

**Option A: Using .env file (Recommended)**
```bash
# Create a .env file in the project root
echo "OPENAI_API_KEY=your-openai-key-here" > .env
echo "HUGGINGFACE_API_TOKEN=your-hf-token-here" >> .env
```

**Option B: Export in terminal**
```bash
# Windows
set OPENAI_API_KEY=your-openai-key-here
set HUGGINGFACE_API_TOKEN=your-hf-token-here

# macOS/Linux
export OPENAI_API_KEY="your-openai-key-here"
export HUGGINGFACE_API_TOKEN="your-hf-token-here"
```

### Step 5: Test Your Setup
```bash
# Run the first tutorial to test everything
cd 01_basics
python 01_introduction.py

# You should see a successful LangChain response!
```

## 🔧 IDE Setup Recommendations

### VS Code (My Personal Favorite!)
1. Install Python extension
2. Install Python Docstring Generator extension
3. Set Python interpreter to your virtual environment

### PyCharm
1. Open project in PyCharm
2. Set interpreter to your virtual environment
3. Enable code completion for better experience

## 🎯 API Key Cost Management

*Don't worry, OpenAI API is quite affordable for learning!*

- **OpenAI GPT-3.5-turbo**: ~$0.002 per 1K tokens
- **For this entire tutorial**: Should cost less than $5-10
- **Pro tip**: Use GPT-3.5-turbo instead of GPT-4 for learning (much cheaper!)

### Setting Spending Limits
1. Go to OpenAI Platform → Billing
2. Set a usage limit (e.g., $20/month)
3. Set up billing alerts

## 🐛 Common Issues & Solutions

### Issue 1: "Module not found" errors
```bash
# Make sure virtual environment is activated
# Reinstall requirements
pip install -r requirements.txt
```

### Issue 2: API Key errors
```bash
# Check if your API key is set correctly
echo $OPENAI_API_KEY  # Should not be empty
```

### Issue 3: Import errors
```bash
# Update pip and reinstall
pip install --upgrade pip
pip install --upgrade langchain
```

### Issue 4: Windows-specific issues
```bash
# If you get SSL errors on Windows
pip install --upgrade certifi
```

## 📱 Alternative: Google Colab Setup

Don't want to install anything locally? No problem!

1. Go to [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. Install packages:
```python
!pip install langchain openai python-dotenv
```
4. Set API keys in Colab:
```python
import os
os.environ["OPENAI_API_KEY"] = "your-key-here"
```

## 🎉 You're All Set!

If you've reached here, congratulations! 🎊 You're ready to start your LangChain journey.

### Next Steps:
1. Start with `01_basics/01_introduction.py`
2. Follow the learning path in README.md
3. Join our community discussions in Issues
4. Build something awesome!

## 🆘 Need Help?

- **Stuck somewhere?** Open a GitHub issue
- **Want to discuss?** Use the Discussions tab
- **Found a bug?** Create a detailed issue report

*Remember: Every expert was once a beginner. You got this! 💪*

---

**Happy Learning! 🚀**

*Made with ❤️ from a small town in India* 🇮🇳
