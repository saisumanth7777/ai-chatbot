# AI-Powered Chatbot — Learning Project

A hands-on project covering Generative AI, LLMs, NLP, ML model development,
vector databases, and prompt engineering. Built with Python + Claude API.

## Project Structure

```
ai-chatbot/
├── phase1_generative_ai/   ← GANs, text generation with Claude API
├── phase2_llm/             ← Hugging Face models, spaCy NLP pipeline
├── phase3_model_dev/       ← Intent classifier (scikit-learn / PyTorch)
├── phase4_openai_api/      ← LangChain + prompt engineering with Claude
├── phase5_vector_db/       ← ChromaDB vector store + semantic search
├── phase6_chatbot/         ← Full chatbot app (FastAPI)
└── utils/                  ← Shared helpers
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Mac/Linux
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. Add your API key:
   ```bash
   cp .env.example .env
   # then edit .env and paste your Anthropic API key
   ```

4. Run each phase in order — each folder has its own README.

## Key Concepts Covered

| Phase | Concept | Tools |
|-------|---------|-------|
| 1 | Generative AI — GANs | PyTorch |
| 1 | LLM text generation | Anthropic (Claude) |
| 2 | NLP pipeline | spaCy, Hugging Face |
| 3 | Intent classification | scikit-learn, PyTorch |
| 4 | Prompt engineering | LangChain + Claude |
| 5 | Semantic search | ChromaDB, sentence-transformers |
| 6 | Full chatbot app | FastAPI |
