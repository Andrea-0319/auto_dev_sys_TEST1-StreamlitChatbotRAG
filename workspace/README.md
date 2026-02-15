# RAG-Enabled Chatbot

A conversational AI chatbot application built with Streamlit that leverages Retrieval-Augmented Generation (RAG) to provide intelligent responses based on user-uploaded documents.

## Features

- 🤖 **Chat Interface**: Intuitive conversational UI with message history
- 📄 **Document Upload**: Support for PDF, TXT, and Markdown files
- 🔍 **RAG System**: Semantic search using FAISS vector store
- 🧠 **Reasoning Display**: Visualize model's step-by-step reasoning process
- 💾 **Persistent Storage**: Save and load vector store between sessions
- ⚙️ **Configurable**: Easy configuration via environment variables

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd workspace
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your settings
```

4. Run the application:
```bash
streamlit run ui/app.py
```

### Usage

1. **Upload Documents**: Go to the "Documents" tab and upload PDF, TXT, or MD files
2. **Ask Questions**: Switch to the "Chat" tab and ask questions about your documents
3. **View Reasoning**: Expand the reasoning section to see the model's thought process
4. **Export Chat**: Download your conversation history

## Configuration

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| MODEL_NAME | Qwen/Qwen2.5-1.5B-Instruct | Hugging Face model |
| DEVICE | cpu | Device for inference (cpu/cuda) |
| CHUNK_SIZE | 512 | Text chunk size for RAG |
| TOP_K_RETRIEVAL | 5 | Number of chunks to retrieve |

See `.env.example` for all options.

## Architecture

```
workspace/
├── config/           # Configuration management
├── core/             # Session and workflow orchestration
├── model/            # Hugging Face model integration
├── rag/              # RAG components (chunking, embeddings, vector store)
├── ui/               # Streamlit interface
└── utils/            # Logging and utilities
```

## Model Support

Tested models:
- Qwen/Qwen2.5-1.5B-Instruct (recommended)
- microsoft/Phi-3-mini-4k-instruct
- HuggingFaceTB/SmolLM-1.7B-Instruct

Models should be:
- Under 7B parameters for CPU execution
- Instruction-tuned for chat
- Compatible with Hugging Face Transformers

## System Requirements

- Python 3.9+
- 4GB+ RAM (8GB recommended)
- CPU with AVX support (for FAISS)
- ~5GB disk space for models

## License

MIT License
