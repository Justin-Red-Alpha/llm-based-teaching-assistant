from sentence_transformers import SentenceTransformer
import pdfplumber
import re
import numpy as np
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import bitsandbytes

def rag(pdf_path,query,k_chunks):
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    def clean_text(text: str) -> str:
        """Normalize newlines and remove control characters."""
        print("Cleaning text...")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", "", text)  # strip control chars
        text = re.sub(r"\n{3,}", "\n\n", text)  # collapse large blank gaps
        return text.strip()

    def extract_pdf(path: str) -> str:
        """Extract text from a text-based PDF using pdfplumber (no OCR fallback)."""
        print("Extracting PDF...")
        text_pages = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                text_pages.append(txt)
        return clean_text("\n\n".join(text_pages))

    def chunk_text(text: str, max_chunk_size: int = 1000, overlap: int = 200) -> list[str]:
        """Chunk text using LangChain's splitter (no other LangChain needed!)."""
        print("Chunking text...")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ",""],
            length_function=len,
        )
        
        chunks = splitter.split_text(text)
        print(f"Created {len(chunks)} chunks")
        return chunks

    def embed_text_chunks(chunks: list[str]):
        print("Embedding text chunks...")
        chunk_embeddings = model.encode(chunks, normalize_embeddings=True)
        return chunk_embeddings

    def create_faiss_index(chunks: list[str], embeddings: np.ndarray):
        print("Creating FAISS index...")
        embeddings = embeddings.astype("float32")
        embedding_dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(embedding_dim)
        index.add(embeddings)

        return index, chunks

    def retrieve(query: str, index, chunk_mapping, k: int = 5):
        """Retrieve top-k most similar chunks."""
        print("Retrieving relevant chunks...")
        query_embedding = model.encode(
            [query],
            normalize_embeddings=True
        )
        
        distances, indices = index.search(query_embedding.astype('float32'), k)
        
        results = [
            {
                'index': int(idx),
                'chunk': chunk_mapping[int(idx)],
                'score': float(distances[0][i])
            }
            for i, idx in enumerate(indices[0])
        ]
        return results
    pdf_text = extract_pdf(pdf_path)
    chunks = chunk_text(pdf_text)
    embeddings = embed_text_chunks(chunks)
    index, chunk_mapping = create_faiss_index(chunks, embeddings)
    results = retrieve(query, index, chunk_mapping, k=k_chunks)
    return results

def concat_chunks(docs):
    return '\n\n'.join(docs[i]['chunk'] for i in range(len(docs)))



def build_model():
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"


    tokenizer = AutoTokenizer.from_pretrained(model_id)
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load model in 4-bit quantization (efficient for Colab)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantization_config
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256
    )
    return generator

def generate_answer(generator, context, question):
    qn = question
    cxt = context
    prompt = f"""
    You are a helpful assistant that reads the context provided to generate answers for exercises, you must provide the essential details and supporting arguments,
    Essential Details are mandatory facts that must appear in a correct answer.  
    Supporting Arguments are additional facts that strengthen or enrich the answer but are not mandatory.  
    Provide the Essential Details and Supporting Arguments for the following question using the provided context.
    Do not create follow-up questions.\n\nContext: {cxt}\n\nQuestion: {qn}\n\nAnswer:
    """
    answer = generator(prompt, do_sample=False)[0]['generated_text']
    return answer

def extract_qa_fields(question_text: str, llm_output: str):
    """
    Extracts question number, question, answer, essential details, and supporting arguments
    from an LLM's text output.
    """

    # --- Step 1: Extract question number and text from the input question ---
    q_match = re.match(r"(\d+)\.\s*(.*)", question_text.strip())
    if q_match:
        question_number = q_match.group(1)
        question = q_match.group(2).strip()
    else:
        question_number = None
        question = question_text.strip()

    # --- Step 2: Extract answer sections from LLM output ---
    # Match sections labeled "Answer", "Essential Details", and "Supporting Arguments"
    patterns = {
        "Answer": r"Answer:\s*(.*?)(?=(Essential Details:|Supporting Arguments:|$))",
        "Essential Details": r"Essential Details:\s*(.*?)(?=(Supporting Arguments:|$))",
        "Supporting Arguments": r"Supporting Arguments:\s*(.*)",
    }

    extracted = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, llm_output, re.DOTALL | re.IGNORECASE)
        extracted[key] = match.group(1).strip() if match else None

    # --- Step 3: Return structured JSON-like dict ---
    result = {
        "Question Number": question_number,
        "Question": question,
        "Answer": extracted.get("Answer"),
        "Essential Details": extracted.get("Essential Details"),
        "Supporting Arguments": extracted.get("Supporting Arguments"),
    }

    return result

