import os
import streamlit as st
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Import RAG modules
from ingest import build_vector_database
from rag_engine import (
    generate_rag_response,
    get_vector_store_stats,
    GROQ_MODELS
)

# Standard Streamlit Page Configuration
st.set_page_config(
    page_title="Charaka Samhita AI Vaidya",
    page_icon="🌿",
    layout="wide"
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Controls
with st.sidebar:
    st.title("📖 Knowledge Base")
    
    stats = get_vector_store_stats()
    
    if stats["exists"] and stats["count"] > 0:
        st.success(f"⚡ **Supabase Vector Store Online**\n\nIndexed Chunks: **{stats['count']}** (1536-dim)")
    elif stats.get("error"):
        st.error(f"⚠️ **Supabase Setup Required**\n\n{stats['error']}")
    else:
        st.warning("⚡ **Supabase Store Empty**\n\nPlease click **Build Index** below to process the target Charaka Samhita chapters into Supabase pgvector.")

    if st.button("⚡ Build / Rebuild Database Index", use_container_width=True):
        with st.spinner("Processing Pharmaceuticals, Rasayana & Siddhisthanam chapters with Google Embeddings (1536-dim)..."):
            try:
                res = build_vector_database(force_rebuild=True)
                st.success(f"Successfully indexed {res['document_count']} document chunks into Supabase pgvector!")
                st.rerun()
            except Exception as e:
                st.error(f"Ingestion failed: {str(e)}")

    st.divider()
    st.subheader("⚙️ RAG Settings")
    
    selected_model = st.selectbox(
        "Groq LLM Model",
        options=GROQ_MODELS,
        index=0,
        help="Select Groq accelerated LLM"
    )
    
    top_k_val = st.slider(
        "Top-K Context Chunks",
        min_value=1,
        max_value=20,
        value=10,
        help="Number of relevant chapter passages retrieved from Supabase pgvector"
    )
    
    temperature_val = st.slider(
        "Response Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Lower values yield more factual, text-grounded responses"
    )

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main Application Title & Subtitle
st.title("🌿 Charaka Samhita AI Vaidya")
st.caption("Authentic RAG Chatbot with Google Embeddings (1536-dim), Supabase pgvector & Groq AI")

# Sample Prompt Suggestions (if chat is empty)
if not st.session_state.messages:
    st.write("##### 💡 Example Questions:")
    col1, col2 = st.columns(2)
    
    sample_queries = [
        "What are the medicinal properties of Kṛtavedhana in Pharmaceuticals?",
        "Explain Rasayana therapy and its benefits in Charaka Samhita",
        "What formulations are described for Vata and Pitta disorders in Siddhisthanam?",
        "How is Jvara (fever) treated according to classical Ayurvedic text?"
    ]
    
    with col1:
        if st.button(sample_queries[0], use_container_width=True):
            st.session_state.selected_prompt = sample_queries[0]
        if st.button(sample_queries[1], use_container_width=True):
            st.session_state.selected_prompt = sample_queries[1]
            
    with col2:
        if st.button(sample_queries[2], use_container_width=True):
            st.session_state.selected_prompt = sample_queries[2]
        if st.button(sample_queries[3], use_container_width=True):
            st.session_state.selected_prompt = sample_queries[3]

# Display Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Render verification sources block
        if msg["role"] == "assistant" and msg.get("citations"):
            with st.expander("📚 **Retrieved Context & Verification Sources (Click to verify)**"):
                for idx, cite in enumerate(msg["citations"], start=1):
                    st.markdown(f"**Source #{idx}: {cite['section']} — {cite['chapter']}**")
                    st.markdown(f"- **Page Number:** `{cite['page_number']}` (Page Range: `{cite['pages']}`)")
                    st.markdown(f"- **Source File:** `{cite['source_file']}`")
                    st.markdown(f"- **Relevance Score:** `{cite['similarity_score']}%`")
                    st.text_area(
                        f"Retrieved Excerpt #{idx}",
                        value=cite["snippet"],
                        height=100,
                        disabled=True,
                        key=f"hist_{msg.get('id', 0)}_{idx}"
                    )
                    st.divider()

# Handle prompt input from quick buttons or text input
user_input = st.chat_input("Ask Vaidya Charaka about Ayurvedic treatments, formulations, or chapters...")

if "selected_prompt" in st.session_state and st.session_state.selected_prompt:
    prompt_to_process = st.session_state.selected_prompt
    st.session_state.selected_prompt = None
else:
    prompt_to_process = user_input

if prompt_to_process:
    # Append User Message
    st.session_state.messages.append({"role": "user", "content": prompt_to_process})
    with st.chat_message("user"):
        st.markdown(prompt_to_process)
        
    # Generate RAG response
    with st.chat_message("assistant"):
        with st.spinner("Searching Supabase pgvector & consulting Groq AI..."):
            rag_output = generate_rag_response(
                query=prompt_to_process,
                top_k=top_k_val,
                model_name=selected_model,
                temperature=temperature_val
            )
            
            st.markdown(rag_output["answer"])
            
            # Display Verification Citations
            citations = rag_output.get("citations", [])
            if citations:
                with st.expander("📚 **Retrieved Context & Verification Sources (Click to verify)**"):
                    for idx, cite in enumerate(citations, start=1):
                        st.markdown(f"**Source #{idx}: {cite['section']} — {cite['chapter']}**")
                        st.markdown(f"- **Page Number:** `{cite['page_number']}` (Page Range: `{cite['pages']}`)")
                        st.markdown(f"- **Source File:** `{cite['source_file']}`")
                        st.markdown(f"- **Relevance Score:** `{cite['similarity_score']}%`")
                        st.text_area(
                            f"Retrieved Excerpt #{idx}",
                            value=cite["snippet"],
                            height=100,
                            disabled=True,
                            key=f"curr_{len(st.session_state.messages)}_{idx}"
                        )
                        st.divider()

    # Store in message history
    st.session_state.messages.append({
        "role": "assistant",
        "content": rag_output["answer"],
        "citations": citations,
        "id": len(st.session_state.messages)
    })
