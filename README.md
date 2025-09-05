# AI-powered-Multimodal-Knowledge-System-for-Skin-Diseases

## 📌 Project Overview
This project implements an **AI-powered multimodal knowledge system** designed for **skin disease understanding and support**.  
It combines multiple AI components into a unified workflow:
- 🖼️ **Image classification** (EfficientNet-based model) for analyzing skin lesion photos.  
- 💬 **Language model** (TinyLlama family) fine-tuned with **QLoRA / PEFT** for conversational medical responses.  
- 📚 **Retrieval-Augmented Generation (RAG)** using FAISS and HuggingFace embeddings for knowledge lookup from medical resources.  
- 🔗 **Agentic workflow orchestration** with `langgraph` and `langchain` to manage multimodal reasoning steps.

⚠️ **Disclaimer**: This chatbot/system is for **research and educational purposes only**. It should not be used as a substitute for professional medical consultation or diagnosis.

---

## 🚀 Quick Start

### Open in Colab
Click below to launch the notebook directly in Google Colab:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<USERNAME>/<REPO>/blob/main/gen_ai_working_code_2.ipynb)

### Run Locally (Optional)
1. Clone this repository:
   ```bash
   git clone https://github.com/<USERNAME>/AI-powered-Multimodal-Knowledge-System-for-Skin-Diseases.git
   cd AI-powered-Multimodal-Knowledge-System-for-Skin-Diseases
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the notebook or build your own Python script using the provided components.

⚙️ Dependencies
The system requires the following libraries:

torch, transformers, datasets, peft, bitsandbytes

sentence-transformers, faiss-cpu

langchain, langgraph

timm, torchvision, pillow

ipywidgets, IPython.display

See requirements.txt for details.

🏗️ System Architecture
Inputs:

Text query (medical question)

Skin lesion image (optional)

Image Pipeline:

Preprocess image (resize, normalize, convert to tensor).

Classify with EfficientNet → predicted label + confidence.

Knowledge Retrieval:

Encode query with sentence-transformers.

Query FAISS index of medical knowledge base.

LLM Response Generation:

Use fine-tuned TinyLlama with QLoRA.

Incorporate classifier output + retrieved documents into prompt.

Agentic Orchestration:

Managed via langgraph + langchain.

Handles multimodal reasoning flow.

Outputs:

Conversational medical-style response.

Context-aware recommendations.

🔍 Detailed Implementation
1. QLoRA Fine-Tuning
Base model: TinyLlama/TinyLlama-1.1B-Chat-v1.0

Fine-tuned with PEFT + QLoRA for efficient low-rank adaptation.

Training format:

css
Copy code
<human>: {instruction}
<assistant>: {response}
Tokenization length: 512

LoRA config: r=8, lora_alpha=32

2. Retrieval
Embedding model: all-MiniLM-L6-v2

Vector store: FAISS

Retrieval pipeline: query → embeddings → FAISS → top-k docs → augment LLM prompt.

3. Image Classifier
Model: Custom EfficientNet (via timm).

Pretrained backbone, custom classifier head for skin disease classes.

Input size: 224x224.

4. Chat Interface
CLI and Colab interface provided.

Agentic workflow graph: manages image analysis → retrieval → response generation.

📊 Evaluation
The system was evaluated using NLP metrics:

ROUGE-1: 0.2936

ROUGE-2: 0.0754

ROUGE-L: 0.2391

BLEU: 0.0271

Semantic Similarity: 0.7751

📂 Repository Structure
bash
Copy code
AI-powered-Multimodal-Knowledge-System-for-Skin-Diseases/
│── gen_ai_working_code_2.ipynb     # Main Colab notebook
│── requirements.txt                # Dependencies
│── README.md                       # Project documentation
│── /models                         # (optional) Model weights (if stored locally)
│── /faiss_store                    # FAISS index files
📝 Future Work
Extend dataset for broader skin disease coverage.

Add multilingual support.

Deploy system as a web app (Streamlit/FastAPI).

Explore vision-language models for tighter integration.

🙌 Acknowledgements
Hugging Face for transformers, datasets, peft.

LangChain & LangGraph for orchestration.

SentenceTransformers for embeddings.

FAISS for vector search.

TinyLlama for efficient LLMs.

⚠️ Disclaimer
This system is not a medical tool. It is a research prototype.
For any health concerns, please consult a licensed medical professional.
