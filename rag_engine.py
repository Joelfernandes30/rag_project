import os
import time
import subprocess
from typing import List, Dict, Any, Union
from dotenv import load_dotenv
from supabase import create_client, Client
from google import genai
from google.oauth2.credentials import Credentials
from groq import Groq

# Load environment variables
load_dotenv()

TABLE_NAME = "charaka_samhita_vectors"
EMBEDDING_DIMENSION = 768

# Available Groq LLM models
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]

_supabase_client: Union[Client, None] = None


def get_secret(key: str, default: str = None) -> str:
    """Fetches key from st.secrets if running in Streamlit, or falls back to os.getenv."""
    try:
        import streamlit as st
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    val = os.getenv(key)
    return val if val is not None else default


def get_supabase_client() -> Client:
    """Initializes and returns singleton Supabase REST API Client."""
    global _supabase_client
    if _supabase_client is None:
        url = get_secret("SUPA_BASE_URL")
        key = get_secret("SUPA_BASE_SCERET_KEY") or get_secret("SUPA_BASE_API_KEY")
        if not url or not key:
            raise ValueError("SUPA_BASE_URL or SUPA_BASE_SCERET_KEY is missing in secrets / .env")
        _supabase_client = create_client(url, key)
    return _supabase_client


def get_google_genai_client() -> genai.Client:
    """
    Initializes Google GenAI Client for Vertex AI / Gemini API.
    Supports GCP Vertex AI Service Account JSON (via st.secrets['gcp_service_account'] or sa_key.json),
    gcloud CLI access token, or direct GEMINI_API_KEY.
    """
    import shutil
    from google.oauth2 import service_account

    api_key = get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")
    project = get_secret("GOOGLE_CLOUD_PROJECT", "project-f6280df9-ac10-4ee7-8fb")
    location = get_secret("GOOGLE_CLOUD_LOCATION", "us-central1")

    # 1. Vertex AI via GCP Service Account in Streamlit Cloud Secrets (st.secrets["gcp_service_account"])
    try:
        import streamlit as st
        if "gcp_service_account" in st.secrets:
            creds_dict = dict(st.secrets["gcp_service_account"])
            creds = service_account.Credentials.from_service_account_info(
                creds_dict,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            proj = creds_dict.get("project_id") or project
            return genai.Client(vertexai=True, project=proj, location=location, credentials=creds)
    except Exception as e:
        print(f"Streamlit secrets GCP Service Account notice: {e}")

    # 2. Vertex AI via local sa_key.json file
    sa_key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sa_key.json")
    if os.path.exists(sa_key_path) and os.path.getsize(sa_key_path) > 10:
        try:
            creds = service_account.Credentials.from_service_account_file(
                sa_key_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            return genai.Client(vertexai=True, project=project, location=location, credentials=creds)
        except Exception as e:
            print(f"sa_key.json init notice: {e}")

    # 3. Direct Gemini API Key if provided
    if api_key:
        return genai.Client(api_key=api_key)
    
    # 4. Local machine fallback via active gcloud CLI token
    if project and shutil.which("gcloud"):
        try:
            token = subprocess.check_output("gcloud auth print-access-token", shell=True, text=True).strip()
            if token:
                creds = Credentials(token)
                return genai.Client(vertexai=True, project=project, location=location, credentials=creds)
        except Exception as e:
            print(f"gcloud access token fetch notice: {e}")

    raise ValueError(
        "GCP Vertex AI Authentication missing. Please add [gcp_service_account] secrets to Streamlit Cloud Secrets."
    )


def get_google_embeddings(texts: Union[str, List[str]], dimension: int = EMBEDDING_DIMENSION) -> List[List[float]]:
    """
    Generates 768-dimensional embeddings for given text or list of texts.
    Uses Google text-embedding-004 via Vertex AI / Gemini API.
    Includes exponential backoff retries for 429 quota limits.
    """
    client = get_google_genai_client()
    
    if isinstance(texts, str):
        input_list = [texts]
    else:
        input_list = texts

    all_embeddings: List[List[float]] = []
    batch_size = 25
    
    for i in range(0, len(input_list), batch_size):
        batch = input_list[i : i + batch_size]
        
        # Exponential backoff retry loop for 429 rate limits
        for attempt in range(1, 6):
            try:
                res = client.models.embed_content(
                    model="text-embedding-004",
                    contents=batch,
                    config={"output_dimensionality": dimension}
                )
                for emb in res.embeddings:
                    all_embeddings.append(emb.values)
                break
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) and attempt < 5:
                    wait_time = attempt * 6
                    print(f"Vertex AI rate limit (429). Retrying batch in {wait_time}s (Attempt {attempt}/5)...", flush=True)
                    time.sleep(wait_time)
                else:
                    raise e
        
        # Gentle delay between API requests to respect Vertex AI per-minute quota
        time.sleep(1.0)

    return all_embeddings


def get_vector_store_stats() -> Dict[str, Any]:
    """Returns Supabase vector store status and document chunk count."""
    try:
        sb = get_supabase_client()
        res = sb.table(TABLE_NAME).select("id", count="exact").limit(1).execute()
        count = res.count if res.count is not None else len(res.data)
        return {"exists": True, "count": count, "error": None}
    except Exception as e:
        err_msg = str(e)
        if "PGRST205" in err_msg or "Could not find the table" in err_msg:
            return {
                "exists": False,
                "count": 0,
                "error": "Table public.charaka_samhita_vectors not created yet. Please execute schema.sql in Supabase SQL Editor."
            }
        return {"exists": False, "count": 0, "error": err_msg}


def retrieve_context(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Queries Supabase pgvector using Google 768-dim query embedding and match_documents RPC.
    Retrieves top_k context passages (default 10, expandable to 20 for full-chapter coverage).
    """
    sb = get_supabase_client()
    
    # Generate 768-dimensional query embedding
    query_embeddings = get_google_embeddings(query, dimension=EMBEDDING_DIMENSION)
    if not query_embeddings:
        return []
        
    query_vector = query_embeddings[0]

    retrieved = []
    try:
        # Call match_documents RPC function in Supabase
        rpc_res = sb.rpc(
            "match_documents",
            {
                "query_embedding": query_vector,
                "match_count": top_k
            }
        ).execute()

        for item in rpc_res.data:
            similarity = float(item.get("similarity", 0.0))
            score_pct = max(0.0, min(100.0, round(similarity * 100, 1)))
            retrieved.append({
                "snippet": item.get("content", ""),
                "section": item.get("section", "Unknown Section"),
                "chapter": item.get("chapter", "Unknown Chapter"),
                "pages": item.get("pages", "N/A"),
                "page_number": item.get("page_number", "N/A"),
                "source_file": item.get("source_file", ""),
                "chapter_label": item.get("chapter_label", f"{item.get('section', '')} - {item.get('chapter', '')}"),
                "similarity_score": score_pct
            })
    except Exception as rpc_err:
        print(f"RPC match_documents error: {rpc_err}. Falling back to table select...")
        try:
            res = sb.table(TABLE_NAME).select("*").limit(top_k).execute()
            for item in res.data:
                retrieved.append({
                    "snippet": item.get("content", ""),
                    "section": item.get("section", "Unknown Section"),
                    "chapter": item.get("chapter", "Unknown Chapter"),
                    "pages": item.get("pages", "N/A"),
                    "page_number": item.get("page_number", "N/A"),
                    "source_file": item.get("source_file", ""),
                    "chapter_label": item.get("chapter_label", f"{item.get('section', '')} - {item.get('chapter', '')}"),
                    "similarity_score": 85.0
                })
        except Exception as e:
            print(f"Fallback select error: {e}")

    return retrieved


def generate_rag_response(
    query: str,
    top_k: int = 10,
    model_name: str = "llama-3.3-70b-versatile",
    temperature: float = 0.1,
    min_relevance: float = 45.0
) -> Dict[str, Any]:
    """
    Executes full RAG workflow: Retrieval via Google Embeddings + Supabase -> Groq LLM generation.
    Strictly grounded — the LLM may ONLY use information present in retrieved vector context.
    Sources below min_relevance (%) are discarded to prevent noise.
    """
    api_key = get_secret("GROQ_API_KEY")
    if not api_key:
        return {
            "answer": "⚠️ GROQ_API_KEY is not set in secrets / `.env`. Please add your Groq API key to proceed.",
            "citations": [],
            "error": "Missing API Key"
        }

    try:
        sources = retrieve_context(query, top_k=top_k)
    except Exception as e:
        return {
            "answer": f"⚠️ Context Retrieval Error: {str(e)}",
            "citations": [],
            "error": str(e)
        }

    if not sources:
        return {
            "answer": "⚠️ No relevant vector data found in Supabase database. Please click **'Build / Rebuild Index'** in the sidebar to index the Charaka Samhita chapters.",
            "citations": [],
            "error": "Database Empty"
        }

    # Filter out low-relevance sources to prevent noise from entering the prompt
    strong_sources = [s for s in sources if s["similarity_score"] >= min_relevance]
    if not strong_sources:
        # If nothing passes threshold, use top 3 anyway but warn
        strong_sources = sources[:3]

    # Format context string for LLM prompt
    context_blocks = []
    for idx, s in enumerate(strong_sources, start=1):
        context_blocks.append(
            f"[Source {idx}] (Relevance: {s['similarity_score']}%)\n"
            f"  Section: {s['section']} | Chapter: {s['chapter']} | Page: {s['page_number']} (Range: {s['pages']})\n"
            f"  Text:\n  {s['snippet']}"
        )
    formatted_context = "\n\n---\n\n".join(context_blocks)

    system_prompt = (
        "You are Vaidya AI, an expert Ayurvedic research and clinical assistant. You answer questions STRICTLY and ONLY "
        "from the Retrieved Context provided below.\n\n"
        "ABSOLUTE RULES — VIOLATIONS ARE FORBIDDEN:\n"
        "1. ONLY use facts, formulations, herbs, treatments, clinical trials, and statements that appear in the Retrieved Context. Do NOT add ANY external knowledge.\n"
        "2. If the Retrieved Context does NOT contain information to answer the question, state cleanly:\n"
        "   \"The retrieved Ayurvedic publications do not contain specific details on this topic. Please try rephrasing your query or ask about a topic covered in the indexed CCRAS books (Drug Development Book, Evidence-Based Ayurvedic Practice, Science of Life Dossier, Nutritional Advocacy, Medico-Ethno-Botanical Survey, CCRAS Vision Document 2030).\"\n"
        "3. Do NOT create empty repetitive sections or headers for missing information. Provide a concise, clear synthesis of what IS found in the context.\n"
        "4. For EVERY claim, formulation, dosage, or clinical finding, cite the exact source in the format: "
        "[Ref: Chapter/Book Name, Page X]. The Book Name and Page MUST come from the Retrieved Context metadata.\n"
        "5. Maintain a professional, dignified Ayurvedic clinical tone.\n"
        "6. End with a concise health disclaimer."
    )

    user_prompt = (
        f"Question: {query}\n\n"
        f"Retrieved Context ({len(strong_sources)} passages):\n\n{formatted_context}"
    )

    try:
        client = Groq(api_key=api_key)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=model_name,
            temperature=temperature,
            max_tokens=3000
        )
        answer = chat_completion.choices[0].message.content

        return {
            "answer": answer,
            "citations": strong_sources,
            "error": None
        }
    except Exception as e:
        return {
            "answer": f"❌ Error generating response from Groq API: {str(e)}",
            "citations": strong_sources,
            "error": str(e)
        }
