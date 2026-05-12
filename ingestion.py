import os
import re
import fitz
import pdfplumber
import hashlib
import json
import time
import pickle
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index"
BM25_INDEX_PATH = "bm25_index.pkl"
HASH_FILE = ".corpus_hash.json"

# ─── PPC Patterns
PPC_SECTION_PATTERNS = [
    re.compile(r'^\s*([1-9]\d{0,2}[A-Z]?(?:\([a-z]\))?)\.\s*[—\-]?\s*(.*)'),
    re.compile(r'^\s*\d+\[([1-9]\d{0,2}[A-Z]?(?:\([a-z]\))?)\.\s*[—\-]?\s*(.*)'),
    re.compile(r'^\s*\*([1-9]\d{0,2}[A-Z]?)\.\s*[—\-]?\s*(.*)'),
    re.compile(r'^\s*\d+\s*\[\d+\[([1-9]\d{0,2}[A-Z]?(?:\([a-z]\))?)\.\s*[—\-]?\s*(.*)'),
    re.compile(r'^\s*\d+([1-5]\d{2}[A-Z]?)\.\s*[—\-]?\s*(.*)'),
]
PPC_IGNORE_PATTERNS = [
    re.compile(r'^\s*\d+\*\['),
    re.compile(r'^\s*\d+\[\d+\[\d+\['),
    re.compile(r'^\s*[1-9]\d{3,}(?!\.)'),
    re.compile(r'^\s*[1-9][S-Z]\.'),
    re.compile(r'^\s*\d+[A-Z][a-z]+\.\s+by\s+'),
]

# ─── CrPC Patterns
CRPC_SECTION_PATTERNS = [
    re.compile(r'^\s*([1-9]\d{0,2}[A-Z]?(?:\([a-z]\))?)\.\s*[—\-]?\s*(.*)'),
    re.compile(r'^\s*\d+\[([1-9]\d{0,2}[A-Z]?(?:\([a-z]\))?)\.\s*[—\-]?\s*(.*)'),
    re.compile(r'^\s*\d+\[([1-9]\d{0,2}-[A-Z])\.\s*[—\-]?\s*(.*)'),
    re.compile(r'^\s*\d+([1-5]\d{2}[A-Z]?)\.\s*[—\-]?\s*(.*)'),
]
CRPC_IGNORE_PATTERNS = [
    re.compile(r'^\s*\d+,\d+\['),
    re.compile(r'^\s*[1-9]\d{3,}(?!\.)'),
    re.compile(r'^\s*[1-9][S-Z]\.'),
    re.compile(r'^\s*\d+[A-Z][a-z]+\.\s+by\s+'),
]

PPC_CHAPTER_REGEX = r'^CHAPTER\s+([IVX]+[A-Z]?)'
CRPC_CHAPTER_REGEX = r'^CHAPTE[R]?\s+([IVX]+[A-Z]?)'
CRPC_PART_REGEX = r'^PART\s+([IVX]+)'

# ─── Helper Function
def match_section(line, patterns, ignore_patterns):
    for ignore in ignore_patterns:
        if ignore.match(line):
            return None
    for pattern in patterns:
        m = pattern.match(line)
        if m:
            return m.group(1), m.group(2).strip()
    return None

CATEGORY_HEADINGS = {
    "Of the Right of Private Defence",
    "Of Criminal Force and Assault",
    "Of Kidnapping, Abduction, Slavery and Forced Labour",
    "Of Rape",
    "Of Unnatural Offences",
    "Of Theft",
    "Of Robbery and Dacoity",
    "OF HIJACKING",
    "Of Hijacking",
    "Of Criminal Misappropriation of Property",
    "Of Criminal Breach of Trust",
    "Of the Receiving of Stolen Property",
    "Of Receiving Stolen Property",
    "Of Cheating",
    "Of Fraudulent Deeds and Dispositions of Property",
    "Of Mischief",
    "Of Criminal Trespass",
    "Of Trade, Property and Other Marks",
    "Of Currency-Notes and Bank-Notes",
    "Of Murder",
    "Of Culpable Homicide Not Amounting to Murder",
    "Of Causing Miscarriage, Injuries to Unborn Children, Exposure of Infants, and Concealment of Births",
    "Of Hurt",
    "Of Wrongful Restraint and Wrongful Confinement",
    "Of Extortion",
    "Of Forgery",
    "Of Using as Genuine a Forged Document",
    "Of Counterfeiting Coin",
    "Of Counterfeiting Currency-Notes and Bank-Notes",
    "Of Defamation",
    "Of Criminal Intimidation, Insult and Annoyance",
    "Of Offences Relating to Marriage",
    "Of Offences Relating to Religion",
    "Of Offences Against the State",
    "Of Contempts of the Lawful Authority of Public Servants",
    "Of False Evidence and Offences Against Public Justice",
    "Of Offences Relating to Elections",
    "Of Public Health, Safety, Convenience, Decency and Morals",
    "Of Offences Against the Public Tranquility",
}


def calculate_corpus_hash():
    corpus_dir = Path("corpus")
    pdf_files = sorted(corpus_dir.glob("*.pdf"))
    hash_data = {}
    for pdf_path in pdf_files:
        with open(pdf_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        hash_data[pdf_path.name] = {
            "hash": file_hash,
            "size": os.path.getsize(pdf_path),
            "modified": os.path.getmtime(pdf_path)
        }
    return hash_data


def load_previous_hash():
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, 'r') as f:
            return json.load(f)
    return None


def save_current_hash(hash_data):
    with open(HASH_FILE, 'w') as f:
        json.dump(hash_data, f, indent=2)


def has_corpus_changed():
    current_hash = calculate_corpus_hash()
    previous_hash = load_previous_hash()

    if previous_hash is None:
        print("ℹ️  No previous ingestion found. Will create new embeddings.")
        return True, current_hash

    if current_hash == previous_hash:
        print("✅ Corpus unchanged. No need to rebuild embeddings.")
        return False, current_hash

    added = set(current_hash.keys()) - set(previous_hash.keys())
    removed = set(previous_hash.keys()) - set(current_hash.keys())
    modified = [f for f in current_hash.keys()
                if f in previous_hash and current_hash[f]['hash'] != previous_hash[f]['hash']]

    if added:
        print(f"📄 New files: {', '.join(added)}")
    if removed:
        print(f"🗑️  Removed files: {', '.join(removed)}")
    if modified:
        print(f"✏️  Modified files: {', '.join(modified)}")

    return True, current_hash


def faiss_index_exists():
    index_path = Path(FAISS_INDEX_PATH)
    exists = index_path.exists() and (index_path / "index.faiss").exists()
    if exists:
        print(f"📊 Existing FAISS index found at: {FAISS_INDEX_PATH}")
    return exists


def extract_text_with_pages(pdf_path):
    pages_content = []
    filename = os.path.basename(pdf_path).lower()

    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            text = doc[page_num].get_text("text").strip()
            page_number = page_num + 1

            if text:
                if "penal code" in filename and page_number <= 29:
                    continue
                if "criminal procedure" in filename:
                    if page_number <= 26 or (192 <= page_number <= 276):
                        continue
                pages_content.append((page_number, text))

        doc.close()
        if pages_content:
            return pages_content

    except Exception as e:
        print(f"PyMuPDF failed: {e}")

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text and text.strip():
                    if "penal code" in filename and page_num <= 29:
                        continue
                    if "criminal procedure" in filename:
                        if page_num <= 26 or (192 <= page_num <= 276):
                            continue
                    pages_content.append((page_num, text))
        if pages_content:
            return pages_content
    except:
        pass

    return []


def process_sections(page_texts, source_name, book_type):
    section_documents = []

    if book_type == "PPC":
        section_patterns = PPC_SECTION_PATTERNS
        ignore_patterns = PPC_IGNORE_PATTERNS
        chapter_regex = PPC_CHAPTER_REGEX
        part_regex = None
    else:
        section_patterns = CRPC_SECTION_PATTERNS
        ignore_patterns = CRPC_IGNORE_PATTERNS
        chapter_regex = CRPC_CHAPTER_REGEX
        part_regex = CRPC_PART_REGEX

    current_section = None
    current_section_title = None
    current_chapter = None
    current_chapter_title = None
    current_part = None
    buffer = ""
    page_start = None

    for page_num, text in page_texts:
        lines = text.split("\n")

        for idx, line in enumerate(lines):
            line = line.strip()

            if part_regex and re.match(part_regex, line):
                current_part = line
                continue

            chapter_match = re.match(chapter_regex, line)
            if chapter_match:
                current_chapter = line
                current_chapter_title = None
                continue

            if line.isupper() and line.startswith("OF") and len(line) > 5:
                if line not in CATEGORY_HEADINGS:
                    current_chapter_title = line
                continue

            if line in CATEGORY_HEADINGS:
                continue

            sec_result = match_section(line, section_patterns, ignore_patterns)

            if sec_result:
                if current_section and buffer.strip():
                    section_documents.append(
                        Document(
                            page_content=buffer.strip(),
                            metadata={
                                "source": source_name,
                                "book": book_type,
                                "section": current_section,
                                "section_title": current_section_title,
                                "chapter": current_chapter,
                                "chapter_title": current_chapter_title,
                                "part": current_part,
                                "page_start": page_start
                            }
                        )
                    )

                current_section = sec_result[0]
                current_section_title = sec_result[1]

                # ✅ FIX: Agar title empty hai toh next line se lo
                if not current_section_title:
                    next_idx = idx + 1
                    if next_idx < len(lines):
                        next_line = lines[next_idx].strip()
                        if not match_section(next_line, section_patterns, ignore_patterns):
                            current_section_title = next_line

                buffer = line + "\n"
                page_start = page_num
                continue

            if current_section:
                buffer += line + "\n"

    if current_section and buffer.strip():
        section_documents.append(
            Document(
                page_content=buffer.strip(),
                metadata={
                    "source": source_name,
                    "book": book_type,
                    "section": current_section,
                    "section_title": current_section_title,
                    "chapter": current_chapter,
                    "chapter_title": current_chapter_title,
                    "part": current_part,
                    "page_start": page_start
                }
            )
        )

    return section_documents


def hierarchical_chunk(section_doc):
    chunks = []

    section = section_doc.metadata.get("section")
    section_title = section_doc.metadata.get("section_title", "")
    chapter = section_doc.metadata.get("chapter")
    chapter_title = section_doc.metadata.get("chapter_title", "")
    part = section_doc.metadata.get("part")
    page = section_doc.metadata.get("page_start")
    source = section_doc.metadata.get("source")
    book = section_doc.metadata.get("book")
    text = section_doc.page_content

    # Level 1
    chunks.append(
        Document(
            page_content=f"[{book} Section {section}] {text}",
            metadata={
                "book": book,
                "section": section,
                "section_title": section_title,
                "chapter": chapter,
                "chapter_title": chapter_title,
                "part": part,
                "page_start": page,
                "source": source,
                "level": 1,
                "chunk_type": "full_section"
            }
        )
    )

    # Level 2
    subsection_pattern = r'\(([a-z]|\d+)\)'
    subsection_matches = re.findall(subsection_pattern, text)

    if len(subsection_matches) > 0:
        parts = re.split(f'({subsection_pattern})', text)

        current_subsection = None
        subsection_buffer = ""

        for i, part_text in enumerate(parts):
            match = re.match(subsection_pattern, part_text)
            if match:
                if current_subsection and subsection_buffer.strip():
                    chunks.append(
                        Document(
                            page_content=f"[{book} Section {section}({current_subsection})] {subsection_buffer.strip()}",
                            metadata={
                                "book": book,
                                "section": section,
                                "subsection": current_subsection,
                                "section_title": section_title,
                                "chapter": chapter,
                                "chapter_title": chapter_title,
                                "part": part,
                                "page_start": page,
                                "source": source,
                                "level": 2,
                                "chunk_type": "subsection"
                            }
                        )
                    )

                current_subsection = match.group(1)
                subsection_buffer = part_text
            else:
                subsection_buffer += part_text

        if current_subsection and subsection_buffer.strip():
            chunks.append(
                Document(
                    page_content=f"[{book} Section {section}({current_subsection})] {subsection_buffer.strip()}",
                    metadata={
                        "book": book,
                        "section": section,
                        "subsection": current_subsection,
                        "section_title": section_title,
                        "chapter": chapter,
                        "chapter_title": chapter_title,
                        "part": part,
                        "page_start": page,
                        "source": source,
                        "level": 2,
                        "chunk_type": "subsection"
                    }
                )
            )

    # Level 3
    if len(text) > 1500:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
        )

        small_chunks = splitter.split_text(text)

        if len(small_chunks) > 1:
            for idx, chunk_text in enumerate(small_chunks):
                chunks.append(
                    Document(
                        page_content=f"[{book} Section {section} Part {idx+1}] {chunk_text}",
                        metadata={
                            "book": book,
                            "section": section,
                            "section_title": section_title,
                            "chapter": chapter,
                            "chapter_title": chapter_title,
                            "part": part,
                            "page_start": page,
                            "source": source,
                            "level": 3,
                            "chunk_type": "small_chunk",
                            "part_number": idx + 1
                        }
                    )
                )

    return chunks


def load_corpus():
    corpus_dir = Path("corpus")
    pdf_files = list(corpus_dir.glob("*.pdf"))

    all_sections = []

    for pdf_path in pdf_files:
        print(f"\n📖 Processing: {pdf_path.name}")

        page_texts = extract_text_with_pages(pdf_path)

        if "penal code" in pdf_path.name.lower():
            book_type = "PPC"
        elif "criminal procedure" in pdf_path.name.lower():
            book_type = "CrPC"
        else:
            continue

        section_docs = process_sections(page_texts, pdf_path.name, book_type)
        all_sections.extend(section_docs)

        print(f"   ✅ Extracted {len(section_docs)} sections")

    print(f"\n📊 Total sections: {len(all_sections)}")
    return all_sections


def build_vectorstore(force_rebuild=False):

    if not force_rebuild:
        corpus_changed, current_hash = has_corpus_changed()

        if not corpus_changed and faiss_index_exists():
            print("\n🎉 Embeddings already exist and corpus unchanged!")
            print("💡 Tip: Use 'python ingestion.py --rebuild' to force rebuild")
            return
    else:
        print("🔨 Force rebuild mode")
        current_hash = calculate_corpus_hash()

    section_docs = load_corpus()
    if not section_docs: 
        raise ValueError("No sections loaded from corpus/")

    print("\n✂️  Applying hierarchical chunking...")
    all_chunks = []
    for section_doc in section_docs:
        hierarchical_chunks = hierarchical_chunk(section_doc)
        all_chunks.extend(hierarchical_chunks)

    level1 = sum(1 for c in all_chunks if c.metadata.get('level') == 1)
    level2 = sum(1 for c in all_chunks if c.metadata.get('level') == 2)
    level3 = sum(1 for c in all_chunks if c.metadata.get('level') == 3)

    print(f"✅ Created {len(all_chunks)} hierarchical chunks")
    print(f"   Level 1 (Full sections): {level1}")
    print(f"   Level 2 (Subsections): {level2}")
    print(f"   Level 3 (Small chunks): {level3}")

    print("\n🔧 Initializing embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # ─── FAISS index banao
    print("\n💾 Creating FAISS index locally...")
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"✅ FAISS index saved → {FAISS_INDEX_PATH}/")

    # ─── BM25 index banao
    print("\n📝 Creating BM25 index...")
    corpus_texts = [doc.page_content.lower().split() for doc in all_chunks]
    bm25 = BM25Okapi(corpus_texts)

    # BM25 + documents save karo
    bm25_data = {
        "bm25": bm25,
        "documents": all_chunks
    }
    with open(BM25_INDEX_PATH, 'wb') as f:
        pickle.dump(bm25_data, f)
    print(f"✅ BM25 index saved → {BM25_INDEX_PATH}")

    print(f"\n✅ Total chunks indexed: {len(all_chunks)}")

    save_current_hash(current_hash)
    print(f"💾 Saved corpus hash to {HASH_FILE}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--rebuild":
        build_vectorstore(force_rebuild=True)
    else:
        build_vectorstore(force_rebuild=False)
