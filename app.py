
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
import networkx as nx
import json
import re

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title = "Policy GraphRAG",
    page_icon  = "🏛️",
    layout     = "wide"
)

st.title("🏛️ Bangladesh Education Policy — GraphRAG Assistant")
st.caption("Multi-agent system powered by LLaMA 3.1 + ChromaDB + Knowledge Graph")

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Groq API Key", type="password")
    st.markdown("Get a free key at [console.groq.com](https://console.groq.com)")
    st.divider()
    st.markdown("### 🤖 Active Agents")
    st.markdown("🕐 **TimeAgent** — tracks changes over time")
    st.markdown("⚠️ **ContradictionAgent** — finds conflicts")
    st.markdown("💡 **ImpactAgent** — traces cause & effect")
    st.markdown("🛡️ **GroundingAgent** — verifies claims")

# ── Policy Documents ──────────────────────────────────────
policy_documents = [
    {"id":"doc_2015","year":2015,"title":"National Education Policy 2015",
     "content":"The Ministry of Education of Bangladesh approved the Girls Stipend Program expansion in 2015. The program allocated 500 crore BDT from the national budget to increase female enrollment. The Finance Ministry approved the budget under World Bank loan WB-2015-EDU. Target: increase girls enrollment by 20% in rural areas by 2018. Primary schools in Sylhet and Rajshahi divisions received priority funding. The policy was signed by Education Minister Abdul Momen on March 15, 2015."},
    {"id":"doc_2018","year":2018,"title":"Education Progress Report 2018",
     "content":"Girls enrollment increased by 18% in rural areas, slightly below the 20% target set in 2015. The Girls Stipend Program was extended until 2022 with increased funding of 750 crore BDT. UNICEF partnership was established to provide teacher training in 500 schools. Dropout rate decreased from 35% to 22% between 2015 and 2018. The Finance Ministry noted budget overrun of 50 crore BDT due to inflation. Secondary school enrollment for girls rose by 25% in Dhaka division."},
    {"id":"doc_2019","year":2019,"title":"Annual Budget Review 2019",
     "content":"The Finance Ministry allocated 600 crore BDT for education, which is 20% less than 2018. However the Education Ministry set a target to increase enrollment by 30% by 2022. The reduced budget conflicts with the expanded enrollment target. Digital classroom initiative launched in 1000 schools using ICT funds of 100 crore BDT. Teacher salary increase of 10% approved to reduce teacher shortage in rural areas. World Bank loan WB-2015-EDU was fully disbursed by December 2019."},
    {"id":"doc_2022","year":2022,"title":"Decade Review 2012-2022",
     "content":"Overall girls enrollment increased by 31% between 2015 and 2022, exceeding original targets. The Girls Stipend Program is credited as the primary driver of enrollment growth. Literacy rate for women aged 15-24 rose from 65% to 84% over the decade. Budget constraints in 2019 caused a temporary slowdown in school construction. 36000 new classrooms were built across Bangladesh using JICA funding. Net enrollment rate reached 98% for primary school aged children."},
    {"id":"doc_2024","year":2024,"title":"National Education Strategy 2024-2030",
     "content":"Building on the Girls Stipend Program success a new Gender Equity in STEM initiative launched. Target: 40% female enrollment in engineering and technology programs by 2030. Budget allocation of 1200 crore BDT approved for the next 6 years. Digital literacy program expanded to cover all 64 districts of Bangladesh. UNICEF and UNESCO partnership renewed for teacher quality improvement. New metric: track not just enrollment but graduation and employment outcomes."}
]

# ── Load models (cached so it only runs once) ─────────────
@st.cache_resource
def load_models():
    embed_model  = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.Client()
    collection   = chroma_client.create_collection(
        name     = "policy_docs",
        metadata = {"hnsw:space": "cosine"}
    )
    all_chunks = []
    for doc in policy_documents:
        words  = doc["content"].split()
        start  = 0
        while start < len(words):
            end   = min(start + 100, len(words))
            chunk = " ".join(words[start:end])
            all_chunks.append({
                "chunk_id": f"{doc['id']}_{start}",
                "doc_id"  : doc["id"],
                "year"    : doc["year"],
                "title"   : doc["title"],
                "text"    : chunk
            })
            start += 80
    texts      = [c["text"]     for c in all_chunks]
    ids        = [c["chunk_id"] for c in all_chunks]
    metadatas  = [{"doc_id": c["doc_id"], "year": c["year"], "title": c["title"]} for c in all_chunks]
    embeddings = embed_model.encode(texts).tolist()
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    G = nx.DiGraph()
    return embed_model, collection, G

embed_model, collection, G = load_models()

# ── Helpers ───────────────────────────────────────────────
def retrieve(query, n=4):
    emb  = embed_model.encode([query]).tolist()
    res  = collection.query(query_embeddings=emb, n_results=n)
    return [{"text": res["documents"][0][i],
             "year": res["metadatas"][0][i]["year"],
             "title": res["metadatas"][0][i]["title"]}
            for i in range(len(res["documents"][0]))]

def ask_llm(client, system, user, max_tokens=250):
    r = client.chat.completions.create(
        model    = "llama-3.1-8b-instant",
        messages = [{"role":"system","content":system},
                    {"role":"user","content":user}],
        temperature = 0.3,
        max_tokens  = max_tokens
    )
    return r.choices[0].message.content

# ── Main UI ───────────────────────────────────────────────
st.divider()
question = st.text_input(
    "💬 Ask a question about Bangladesh education policy:",
    placeholder="e.g. How did girls enrollment change between 2015 and 2024?"
)

example_questions = [
    "How did girls enrollment change between 2015 and 2024?",
    "Are there any budget vs enrollment target conflicts?",
    "What was the impact of the Girls Stipend Program?"
]
st.caption("💡 Try: " + " | ".join(example_questions))

if st.button("🔍 Analyze", type="primary", disabled=not api_key):
    if not question:
        st.warning("Please enter a question!")
    else:
        client = Groq(api_key=api_key)

        col1, col2 = st.columns(2)

        with col1:
            with st.spinner("🕐 TimeAgent thinking..."):
                ctx  = retrieve(question + " change over time year")
                ctx_text = "\n\n".join([f"[{c['year']}] {c['text']}" for c in ctx])
                time_ans = ask_llm(client,
                    "You are a TimeAgent. Analyze changes over time. Mention specific years. Under 120 words.",
                    f"Question: {question}\n\nContext:\n{ctx_text}")
            st.subheader("🕐 Timeline Analysis")
            st.write(time_ans)

            with st.spinner("⚠️ ContradictionAgent thinking..."):
                ctx2  = retrieve(question + " conflict contradiction inconsistency")
                ctx_text2 = "\n\n".join([f"[{c['year']}] {c['text']}" for c in ctx2])
                contra_ans = ask_llm(client,
                    "You are a ContradictionAgent. Find contradictions between documents. Say CONTRADICTION: if found. Under 120 words.",
                    f"Question: {question}\n\nContext:\n{ctx_text2}")
            st.subheader("⚠️ Contradictions")
            st.write(contra_ans)

        with col2:
            with st.spinner("💡 ImpactAgent thinking..."):
                ctx3  = retrieve(question + " impact outcome result caused")
                ctx_text3 = "\n\n".join([f"[{c['year']}] {c['text']}" for c in ctx3])
                impact_ans = ask_llm(client,
                    "You are an ImpactAgent. Trace cause-and-effect chains. Under 120 words.",
                    f"Question: {question}\n\nContext:\n{ctx_text3}")
            st.subheader("💡 Impact Chain")
            st.write(impact_ans)

            with st.spinner("🛡️ GroundingAgent verifying..."):
                combined = time_ans + " " + contra_ans + " " + impact_ans
                ctx4  = retrieve(question)
                ctx_text4 = "\n\n".join([f"[{c['year']}] {c['text']}" for c in ctx4])
                verify_ans = ask_llm(client,
                    "You are a GroundingAgent. Verify claims with ✅ VERIFIED, ❌ UNVERIFIED, or ⚠️ PARTIAL. Under 150 words.",
                    f"Answer:\n{combined}\n\nSources:\n{ctx_text4}")
            st.subheader("🛡️ Verification Report")
            st.write(verify_ans)

        st.divider()
        with st.spinner("📝 Writing final answer..."):
            final = ask_llm(client,
                "Synthesize the multi-agent analysis into one clean answer. Sections: 📅 Timeline, ⚠️ Contradictions, 💡 Impact, 🛡️ Confidence. Under 300 words.",
                f"Question: {question}\n\nTime: {time_ans}\n\nContradictions: {contra_ans}\n\nImpact: {impact_ans}\n\nVerification: {verify_ans}",
                max_tokens=400)
        st.subheader("📊 Final Synthesized Answer")
        st.success(final)

elif not api_key:
    st.info("👈 Enter your Groq API key in the sidebar to get started.")
