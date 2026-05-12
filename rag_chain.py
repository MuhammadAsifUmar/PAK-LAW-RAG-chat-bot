# ================================================
# RAG file - Updated for Hierarchical Chunking + FAISS + BM25 + Title Match
# ================================================

import os
import re
import json
import queue
import threading
import pickle
from dotenv import load_dotenv
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

# Classic modules (old API moved here)
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_classic.callbacks.base import BaseCallbackHandler

from pydantic import Field

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index"
BM25_INDEX_PATH = "bm25_index.pkl"
GROQ_MODEL = "llama-3.3-70b-versatile"

# ─── Stop words
STOP_WORDS = {
    'what', 'is', 'the', 'a', 'an', 'of', 'in', 'to', 'for',
    'and', 'or', 'under', 'does', 'mean', 'how', 'does', 'are',
    'by', 'with', 'from', 'that', 'this', 'which', 'when', 'where',
    'who', 'define', 'explain', 'describe', 'tell', 'me', 'about',
    'pakistan', 'ppc', 'crpc', 'code', 'section', 'cases', 'case'
}

# ─── Embeddings
print("Loading local embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# ─── FAISS Load
print("Loading FAISS vectorstore...")
vectorstore = FAISS.load_local(
    FAISS_INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

# ─── BM25 Load
print("Loading BM25 index...")
with open(BM25_INDEX_PATH, 'rb') as f:
    bm25_data = pickle.load(f)
bm25 = bm25_data["bm25"]
bm25_documents = bm25_data["documents"]

# ─── Query Analyzer LLM
llm_analyzer = ChatGroq(
    api_key=GROQ_API_KEY,
    model=GROQ_MODEL,
    temperature=0.0,
    max_tokens=200,
    streaming=False,
)

# ─── Combined Query Analysis + Expansion
COMBINED_QUERY_PROMPT = """You are an expert on Pakistan Penal Code (PPC) 1860 and Code of Criminal Procedure (CrPC) 1898.

Analyze this query and return ONLY valid JSON — no explanation, no markdown:

Query: "{query}"

Return this exact JSON structure:
{{
  "clean_query": "rewritten clear query",
  "intent": "definition|explanation|punishment|procedure|comparison|general",
  "section_number": "302" or null,
  "legal_domain": "PPC|CrPC|both|unknown",
  "expanded_keywords": "section numbers and keywords for searching — max 20 words"
}}

Rules for expanded_keywords:
- Use your PPC/CrPC knowledge to find correct section numbers
- For definition queries → always include Section 4 CrPC or Section 3 PPC
- For procedure queries → include relevant chapter sections
- For offence queries → include punishment sections too
- Example: hurt types → PPC 332 333 334 335 336 hurt injury pain
- Example: self defence → PPC 96 97 98 99 100 private defence
- Example: bail → CrPC 496 497 498 499 500 bail bailable surety
- Example: FIR → CrPC 154 155 156 157 FIR cognizable police
- Example: complaint definition → CrPC 4 2 definitions complaint cognizable
- Example: criminal courts → CrPC 6 7 8 9 courts session magistrate
- Example: summons → CrPC 61 62 63 64 68 69 summons warrant
- Example: evidence → CrPC 354 355 evidence recording inquiry
- Example: rights accused trial → CrPC 340 342 343 accused defence lawyer trial
- Example: arrest rights → CrPC 50 54 56 57 arrest rights person accused
- Example: qatl-i-amd → PPC 300 302 304 306 307 qatl amd murder intention
- Example: FIR murder procedure → CrPC 154 156 157 173 302 investigation police"""


def analyze_query(query: str) -> dict:
    """Query analyze karo — section, intent, domain, keywords nikalo"""

    # ─── Range detect karo jaise 221-223
    range_match = re.search(r'\b(\d{1,3})[–-](\d{1,3})\b', query)
    if range_match:
        start = int(range_match.group(1))
        end = int(range_match.group(2))
        sections = [str(i) for i in range(start, end+1)]
        book = None
        book_match = re.search(r'\b(PPC|CrPC)\b', query, re.IGNORECASE)
        if book_match:
            book = book_match.group(1).upper()
        return {
            "clean_query": query,
            "intent": "explanation",
            "section_number": sections[0],
            "legal_domain": book or "unknown",
            "expanded_keywords": query + " " + " ".join(sections)
        }

    # ─── Single section number detect karo
    section_match = re.search(r'\b(\d{1,3}[A-Z]?)\b', query)
    if section_match:
        book = None
        book_match = re.search(r'\b(PPC|CrPC)\b', query, re.IGNORECASE)
        if book_match:
            book = book_match.group(1).upper()
        return {
            "clean_query": query,
            "intent": "explanation",
            "section_number": section_match.group(1),
            "legal_domain": book or "unknown",
            "expanded_keywords": query
        }

    try:
        prompt = COMBINED_QUERY_PROMPT.format(query=query)
        response = llm_analyzer.invoke(prompt)
        content = response.content.strip()

        content = re.sub(r'```json|```', '', content).strip()
        result = json.loads(content)

        if not result.get("expanded_keywords"):
            result["expanded_keywords"] = query

        print(f"🔍 Query analyzed: {result}")
        return result

    except Exception as e:
        print(f"⚠️ Query analysis failed: {e}")
        return {
            "clean_query": query,
            "intent": "general",
            "section_number": None,
            "legal_domain": "unknown",
            "expanded_keywords": query
        }

# ─── Title Match Search
def title_match_search(query: str, book: str = None, top_k: int = 5) -> List[Document]:
    """Query keywords ko section titles mein dhundho"""
    query_words = set(query.lower().split()) - STOP_WORDS
    if not query_words:
        return []

    scored_docs = []
    for doc in bm25_documents:
        meta = doc.metadata
        if meta.get('level') != 1:
            continue
        if book and meta.get('book') != book:
            continue

        title = meta.get('section_title', '').lower()
        if not title:
            continue

        match_count = sum(1 for word in query_words if word in title)
        if match_count > 0:
            scored_docs.append((match_count, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]

# ─── Inject Metadata into Answer
def inject_metadata_into_answer(answer: str, source_docs: list) -> str:
    """Section aur Book — metadata se lo"""

    if answer is None:
        return ""
    if not isinstance(answer, str):
        answer = str(answer)
    if not source_docs:
        return answer

    answer_book = None
    if 'CrPC' in answer or 'Code of Criminal Procedure' in answer:
        answer_book = 'CrPC'
    elif 'PPC' in answer or 'Penal Code' in answer:
        answer_book = 'PPC'

    if not answer_book:
        if '• CrPC' in answer:
            answer_book = 'CrPC'
        elif '• PPC' in answer:
            answer_book = 'PPC'

    primary_doc = source_docs[0]
    for doc in source_docs:
        meta = doc.metadata
        if meta.get('level') == 1:
            if answer_book is None or meta.get('book') == answer_book:
                primary_doc = doc
                break

    meta = primary_doc.metadata

    book          = meta.get("book", "")
    section       = meta.get("section", "")
    section_title = meta.get("section_title", "")
    subsection    = meta.get("subsection", "")

    answer = answer.replace("Pakistan Penal Code", "PPC")
    answer = answer.replace("Code of Criminal Procedure", "CrPC")

    if section:
        section_str = f"Section {section}"
        if section_title:
            section_str += f" — {section_title}"
        if subsection:
            section_str += f" ({subsection})"
        answer = re.sub(
            r'\*\*Section.*?\*\*',
            f'**{section_str}**',
            answer,
            count=1
        )

    if book:
        answer = re.sub(
            r'\*\*Book:\*\*[^\n]*',
            f'**Book:** {book}',
            answer
        )

    return answer


# ─── Smart Legal Retriever — FAISS + BM25 + Title Match
class SmartLegalRetriever(BaseRetriever):
    vectorstore: any = Field()
    k: int = Field(default=7)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:

        analysis = analyze_query(query)

        sec_num  = analysis.get("section_number")
        book     = analysis.get("legal_domain")
        expanded = analysis.get("expanded_keywords") or query

        if book and book.upper() not in ["PPC", "CRPC"]:
            book = None
        elif book:
            book = book.upper()
            if book == "CRPC":
                book = "CrPC"

        exact_docs = []

        if sec_num:
            for doc in bm25_documents:
                meta = doc.metadata
                if meta.get('section') == sec_num and meta.get('level') == 1:
                    if book is None or meta.get('book') == book:
                        exact_docs.append(doc)

        faiss_docs = self.vectorstore.similarity_search(expanded, k=self.k)

        tokenized_query = expanded.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        top_bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:self.k]
        bm25_docs = [bm25_documents[i] for i in top_bm25_indices]

        title_docs = title_match_search(query, book=book, top_k=5)

        if book:
            same_book_faiss = [d for d in faiss_docs if d.metadata.get('book') == book]
            other_book_faiss = [d for d in faiss_docs if d.metadata.get('book') != book]
            faiss_docs = same_book_faiss + other_book_faiss

            same_book_bm25 = [d for d in bm25_docs if d.metadata.get('book') == book]
            other_book_bm25 = [d for d in bm25_docs if d.metadata.get('book') != book]
            bm25_docs = same_book_bm25 + other_book_bm25

        if exact_docs:
            combined = exact_docs + title_docs + faiss_docs + bm25_docs
        else:
            combined = title_docs + bm25_docs + faiss_docs

        seen = set()
        final_docs = []

        for doc in combined:
            key = doc.metadata.get('section', '') + doc.metadata.get('book', '')
            if key not in seen:
                seen.add(key)
                final_docs.append(doc)

        return final_docs[:self.k + 3]

# ─── Prompt
PROMPT_TEMPLATE = """
You are a precise legal assistant specialized in the Pakistan Penal Code (PPC) 1860
and the Code of Criminal Procedure (CrPC) 1898.

Strict Rules:

1. Answer ONLY using the provided context.
2. Do NOT generate information not present in the context.
3. If question mentions a specific section number like "302", "375", "154" etc —
   search that exact section in context and answer from it directly.
4. If the question is unrelated to PPC or CrPC respond ONLY with:
"I can only answer questions related to the Pakistan Penal Code (PPC) 1860 or the Code of Criminal Procedure (CrPC) 1898."
5. If the information is not found in the context respond ONLY with:
"This specific information is not found in the indexed sections."
6. If the answer exists, structure your response EXACTLY like this:

**Section [number] — [Section Title]**

[Clear explanation of the section in simple words]

**Key Points:**
- [Point 1]
- [Point 2]
- [Point 3 if applicable]

**Book:**

7. If the context shows '[* * *]' or omitted content for a section,
   respond with: "Section [X] has been repealed/omitted from the
   Pakistan Penal Code / Code of Criminal Procedure."
8. NEVER repeat the question as the answer.
9. Always use "PPC" not "Pakistan Penal Code" and "CrPC" not "Code of Criminal Procedure".
10. NEVER say "Section X of CrPC" for PPC sections and vice versa.
    Always check Book metadata before mentioning book name with section.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)

# ─── Streaming Handler
class QueueStreamHandler(BaseCallbackHandler):
    def __init__(self, token_queue: queue.Queue):
        self.q = token_queue

    def on_llm_new_token(self, token: str, **kwargs):
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs):
        self.q.put(None)

    def on_llm_error(self, error, **kwargs):
        self.q.put(None)

# ─── Non-Streaming LLM
print("Initializing Groq LLM...")
llm_batch = ChatGroq(
    api_key=GROQ_API_KEY,
    model=GROQ_MODEL,
    temperature=0.1,
    max_tokens=1800,
    streaming=False,
)

# ✅ Memory off
memory_batch = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer",
    k=0
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm_batch,
    retriever=SmartLegalRetriever(vectorstore=vectorstore),
    memory=memory_batch,
    combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents=True,
    output_key="answer",
)

# ✅ Stream memory off
stream_memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer",
    k=0
)

print("✅ RAG chain ready with FAISS + BM25 + Title Match!")

# ─── Citation Builder
_NOT_FOUND_PHRASES = (
    "i can only answer questions related",
    "this specific information is not found",
)

def build_citations(answer: str, source_docs: list, max_refs: int = 4):
    if not answer:
        return ""
    if any(p in answer.lower() for p in _NOT_FOUND_PHRASES):
        return ""

    lines = []
    seen = set()

    for doc in source_docs:
        meta = doc.metadata

        book          = meta.get("book", "")
        section       = meta.get("section", "")
        section_title = meta.get("section_title", "")
        subsection    = meta.get("subsection", "")
        page = meta.get("page_start", meta.get("page", ""))

        if book and section:
            key = (book, section, subsection if subsection else "")
        else:
            src = meta.get("filename", meta.get("source", ""))
            key = (src, "")

        if key in seen:
            continue
        seen.add(key)

        citation_parts = []

        if book and section:
            if subsection:
                citation_parts.append(f"{book} Section {section}({subsection})")
            else:
                citation_parts.append(f"{book} Section {section}")
            if section_title:
                citation_parts.append(f"({section_title})")

        if citation_parts:
            if page:
                citation_parts.append(f"Page {page}")
            lines.append(f"• {' — '.join(citation_parts)}")

    if lines:
        return "\n\n**References:**\n" + "\n".join(lines[:max_refs])

    return ""

# ─── Standard Ask (non-streaming)
def ask_question(question: str):
    result = qa_chain.invoke({"question": question})
    answer = result.get("answer") or ""
    if not answer:
        return "I can only answer questions related to the Pakistan Penal Code (PPC) 1860 or the Code of Criminal Procedure (CrPC) 1898."
    try:
        answer = inject_metadata_into_answer(answer, result["source_documents"])
        citations = build_citations(answer, result["source_documents"])
        return answer + citations
    except Exception:
        return answer

# ─── Streaming Ask
def stream_question(question: str):
    token_q = queue.Queue()
    stream_handler = QueueStreamHandler(token_q)

    llm_stream = ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0.1,
        max_tokens=1800,
        streaming=True,
        callbacks=[stream_handler],
    )

    stream_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_stream,
        retriever=SmartLegalRetriever(vectorstore=vectorstore),
        memory=stream_memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        output_key="answer",
    )

    result_holder = {}
    error_holder = {}

    def run_chain():
        try:
            result_holder["data"] = stream_chain.invoke({"question": question})
        except Exception as e:
            error_holder["err"] = e
            token_q.put(None)

    thread = threading.Thread(target=run_chain, daemon=True)
    thread.start()

    while True:
        token = token_q.get()
        if token is None:
            break
        yield token

    thread.join()
    if "err" in error_holder:
        raise error_holder["err"]

    if "data" in result_holder:
        result = result_holder["data"]
        answer = result.get("answer") or ""
        if answer:
            try:
                answer = inject_metadata_into_answer(answer, result["source_documents"])
                citations = build_citations(answer, result["source_documents"])
                if citations:
                    yield citations
            except Exception:
                pass

# ─── CLI Test
if __name__ == "__main__":
    test_questions = [
        "hi",
        "What are rights of accused during trial?",
        "What is complaint under CrPC?",
        "What is Section 302 of PPC?",
    ]

    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print('='*60)
        print("\nAnswer:\n")
        for token in stream_question(question):
            print(token, end="", flush=True)
        print("\n")