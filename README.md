<img width="871" height="316" alt="Screenshot 2025-07-15 at 2 36 33 PM" src="https://github.com/user-attachments/assets/b779d7b0-b3fe-4656-b2d1-326e68772287" />
# 🔍📚 Local-RAG: A Lightweight RAG Pipeline with Gemma-2B and MPNet (PyTorch)

Welcome to **Local-RAG**, a fully local **Retrieval-Augmented Generation (RAG)** pipeline built entirely using **PyTorch**, designed and developed by [Rushabh333](https://github.com/Rushabh333). This pipeline enables intelligent question answering over large documents like PDFs — all running on your **local GPU**.

---

## 🚀 Features

- ✅ Local inference with **Gemma-2B-IT** LLM
- ✅ Fast and accurate **semantic search** using `all-mpnet-base-v2` embeddings
- ✅ Efficient **PDF text chunking** and embedding pipeline
- ✅ Fully **PyTorch-native**, no LangChain/Faiss dependencies
- ✅ Plug-and-play for **any textbook or PDF document**
- ✅ Simple, clean, and extendable architecture

---

## 🧠 Architecture Overview

This pipeline is divided into **two major sections**:

### 1. 📄 Document Preprocessing & Embedding

- 📥 Load any PDF textbook or research paper.
- ✂️ Chunk the text into overlapping segments for better retrieval.
- 🧠 Embed chunks using `all-mpnet-base-v2`.
- 💾 Store embeddings and chunk metadata in memory or disk.

### 2. 🔎 Querying & Generation

- 🔍 Accept a user query.
- 🤖 Perform vector search to find top relevant chunks.
- 📝 Create a prompt that includes these chunks.
- 💬 Generate a final answer using **Gemma 2B (IT)** model locally.

---

## 🔧 Setup Instructions

Follow these steps to run the project on your local machine (GPU recommended):

### 1. ✅ Clone the Repository

```bash
git clone https://github.com/Rushabh333/Local-RAG.git
cd Local-RAG
2. 🐍 Create a Virtual Environment
bash
Copy
Edit
python -m venv rag-env
source rag-env/bin/activate  # Windows: rag-env\Scripts\activate
3. 📦 Install Dependencies
bash
Copy
Edit
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate
pip install sentence-transformers
pip install scikit-learn
pip install PyMuPDF  # or: pip install pdfplumber
Optional: Create a requirements.txt from your environment for reproducibility.

4. 💾 Load Models
Embedding Model: all-mpnet-base-v2
python
Copy
Edit
from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer("all-mpnet-base-v2")
LLM: gemma-2b-it
python
Copy
Edit
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it",
    device_map="auto",
    torch_dtype=torch.float16
)
5. 📘 Preprocess a PDF
python
Copy
Edit
import fitz  # PyMuPDF
doc = fitz.open("docs/my_textbook.pdf")
text = "\n".join([page.get_text() for page in doc])
Then chunk the text and embed it using your SentenceTransformer model.

6. 🔍 Ask Questions
python
Copy
Edit
query = "Explain dropout in neural networks."
# → Retrieve top-k similar chunks from the embedding index
# → Create a prompt: "Based on the following text, answer the question..."
# → Generate the answer using Gemma
📁 Project Structure
bash
Copy
Edit
Local-RAG/
├── Local_RAG(2).ipynb      # Main notebook for RAG flow
├── README.md               # This file

📌 Future Improvements
 Use disk-based vector DB (e.g., FAISS or ChromaDB)

 Add conversation memory for chat-style interaction

 Web UI with Gradio or Streamlit

 Quantized LLMs for lower GPU memory usage

🧑‍💻 Author
Made by Rushabh Lodha (Rushabh333)
