# Local-RAG
Local RAG Pipeline with Gemma-2B and MPNet Embeddings (PyTorch)
This project implements a Retrieval-Augmented Generation (RAG) pipeline from scratch using PyTorch. It allows you to query large documents (like PDFs) and get accurate, contextual answers by combining retrieval-based and generative techniques.

The entire pipeline is designed to run locally on GPU, using:

LLM: gemma-2b-it (locally hosted)

Embedding model: all-mpnet-base-v2 (for semantic chunking)

Framework: Pure PyTorch (no LangChain/Faiss)

🧩 Project Structure
The RAG pipeline is broken into two major sections:

1. Document Preprocessing & Embedding
PDF Loading: Open and parse any PDF document.

Text Chunking: Split text into manageable, semantically meaningful chunks.

Embedding: Convert all chunks into dense vector embeddings using all-mpnet-base-v2.

Storage: Store embeddings and associated metadata for efficient retrieval.

2. Search & Answering
Retrieval: Perform vector search to find the most relevant chunks for a given query.

Prompt Creation: Construct a prompt by combining the user query with retrieved context.

Answer Generation: Use the locally running gemma-2b-it model to generate answers grounded in retrieved context.

🛠️ Dependencies
torch

transformers

sentence-transformers

PyMuPDF (fitz) or pdfplumber (for PDF parsing)

numpy

scikit-learn (for nearest neighbor search or cosine similarity)

accelerate (for optimized inference on local GPU)

All components are designed to work locally, without needing external APIs.

🚀 How to Run
Clone the repo and install dependencies

Load any PDF textbook/document

Preprocess and embed it

Ask a question and get contextual answers powered by retrieval

📌 Example Use Case
Input Query: "What are the main applications of convolutional neural networks?"

Output (Generated Answer):
"Convolutional Neural Networks (CNNs) are primarily used in image and video recognition, medical image analysis, and time-series classification, as discussed in Chapter 3 of the uploaded textbook."

📂 Files
LOCAL_RAG(2).ipynb – Main notebook


💡 Future Work
Replace MPNet with custom fine-tuned domain embeddings

Add support for conversational memory

Integrate quantized or more efficient models for edge devices

📃 License
MIT License. Feel free to fork, modify, and use locally!
