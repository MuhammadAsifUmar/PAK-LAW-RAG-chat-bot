# PakLaw AI · Legal Assistant

## What is this project?
PakLaw AI is an intelligent legal assistant tailored specifically for the **Pakistan Penal Code (PPC) 1860** and the **Code of Criminal Procedure (CrPC) 1898**. It is a web-based application built to help users quickly search, understand, and navigate Pakistani criminal laws through a conversational AI interface. 

## Why was it built?
Legal codes are often complex, dense, and difficult to navigate for the general public and even law students. Finding the exact section, punishment, or procedure requires significant time. This project was built to **democratize legal knowledge**, making it instantly accessible and easy to understand using modern AI capabilities, while ensuring the answers are strictly grounded in actual legal text.

## How does it work?
The project utilizes an advanced **Retrieval-Augmented Generation (RAG)** pipeline:
1. **Query Analysis:** When a user asks a question, the system first analyzes the intent (e.g., definition, punishment, procedure) and expands the query with relevant legal keywords.
2. **Hybrid Retrieval:** It searches for relevant legal sections using a combination of methods:
   - **FAISS (Dense Retrieval):** Understands the semantic meaning of the query.
   - **BM25 (Sparse Retrieval):** Performs exact keyword matching.
   - **Title Matching:** Directly matches query keywords against section titles.
3. **Answer Generation:** The retrieved context is passed to a Large Language Model (LLM) which generates a clear, structured response along with accurate references to the legal books, sections, and subsections.
4. **Persistence:** The application saves chat histories in a database so users can revisit previous conversations.

## When to use it?
- When you need to understand specific sections of the PPC or CrPC (e.g., "What is Section 302 of PPC?").
- When looking for the legal definition of an offense or the prescribed punishment.
- When you want to know the legal procedure for an action (e.g., "Bail rules under CrPC").
- **Disclaimer:** This tool is for informational and educational purposes and does not substitute professional legal advice.

## Technologies Used & Why
- **Python:** The core programming language used for its rich ecosystem in AI and data processing.
- **Streamlit:** Used for the frontend to rapidly build a beautiful, interactive, and responsive chat interface.
- **LangChain:** The framework used to orchestrate the LLM, memory, and retrieval chains.
- **Groq Cloud (Llama 3 Model):** Used as the primary LLM because Groq provides incredibly fast inference speeds, which is crucial for a real-time conversational assistant.
- **HuggingFace Embeddings (`all-MiniLM-L6-v2`):** Used to convert text into vector embeddings. It was chosen for its excellent balance of performance and speed for semantic search.
- **FAISS (Facebook AI Similarity Search):** Used as the vector database for rapid similarity search of embeddings.
- **BM25:** A classic information retrieval algorithm used to ensure exact keyword matches aren't missed by semantic search.
- **SQLite (via custom persistence):** Used to store and manage user chat sessions reliably without needing a heavy database setup.
- \
please check the Demo video.

  [Screencast From 2026-05-12 11-50-17.webm](https://github.com/user-attachments/assets/1f048e65-208e-4718-bc8b-9ecddb571e11)
  

