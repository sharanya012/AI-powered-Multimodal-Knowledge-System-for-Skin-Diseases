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

---

## ⚙️ Implementation Details

### 1. **Skin Image Classification**
- Backbone: **EfficientNet**
- Task: Classifies uploaded skin lesion images into categories.
- Usage: Provides visual diagnostic context that can be combined with user queries.

### 2. **Language Model**
- Base Model: **TinyLlama**
- Fine-tuning: **QLoRA with PEFT (Parameter-Efficient Fine-Tuning)**
- Role: Understands patient queries, reasons over retrieved knowledge, and generates responses.

### 3. **Knowledge Retrieval (RAG)**
- Embeddings: HuggingFace sentence transformers
- Indexing: **FAISS**
- Purpose: Retrieves domain-specific knowledge (medical texts, guidelines) to ground chatbot answers.

### 4. **Agentic Workflow**
- Frameworks: **LangGraph + LangChain**
- Orchestration: Dynamically decides actions (e.g., classify image → retrieve knowledge → generate response).
- Result: Produces **explainable, multimodal answers**.

---

## Workflow Diagram:

 User Query + Skin Image
          │
          ▼
   [EfficientNet] → Image Classification
          │
          ▼
   [FAISS + Embeddings] → Knowledge Retrieval
          │
          ▼
   [TinyLlama (QLoRA)] → Conversational Response
          │
          ▼
      Final Multimodal Answer


## Requirements
Install the following dependencies before running locally:
torch
torchvision
transformers
peft
accelerate
bitsandbytes
faiss-cpu
sentence-transformers
langchain
langgraph
(List not exhaustive — check notebook for details.)

