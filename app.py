import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import re
import PyPDF2
import tempfile
import os
from fpdf import FPDF
from datetime import datetime
import base64

st.set_page_config(page_title="Policy GraphRAG", page_icon="🏛️", layout="wide")

st.title("🏛️ Bangladesh Education Policy — GraphRAG Assistant")
st.caption("Multi-agent system powered by LLaMA 3.1 + ChromaDB + Knowledge Graph")

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

# ── Entity colors ─────────────────────────────────────────
entity_colors = {
    "PROGRAM":      "#4CAF50",
    "PERSON":       "#2196F3",
    "ORGANIZATION": "#FF9800",
    "BUDGET":       "#F44336",
    "POLICY":       "#9C27B0",
    "METRIC":       "#00BCD4",
    "OTHER":        "#9E9E9E"
}

# ── Load models ───────────────────────────────────────────
@st.cache_resource
def load_models():
    embed_model   = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.Client()
    collection    = chroma_client.create_collection(
        name="policy_docs",
        metadata={"hnsw:space": "cosine"}
    )
    all_chunks = []
    for doc in policy_documents:
        words = doc["content"].split()
        start = 0
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
    return embed_model, collection

# ✅ MUST be before sidebar
embed_model, collection = load_models()

# ── PDF Report Generator ──────────────────────────────────
def clean_text(text):
    """Sanitize text for FPDF latin-1 font — handles all common Unicode chars."""
    import unicodedata
    replacements = [
        ("—", "--"), ("–", "-"), ("‒", "-"), ("‑", "-"),
        ("‘", "'"),  ("’", "'"), ("‚", "'"), ("‛", "'"),
        ("“", '"'),  ("”", '"'), ("„", '"'), ("‟", '"'),
        ("…", "..."),("•", "*"), ("●", "*"),
        ("→", "->"), ("←", "<-"),
        ("·", "-"),
        ("✅", "[OK]"),("❌", "[X]"),("⚠", "[!]"),
        ("✓", "[v]"),("✔", "[v]"),("✗", "[x]"),
        ("é", "e"),  ("è", "e"), ("ê", "e"), ("ë", "e"),
        ("à", "a"),  ("â", "a"), ("ä", "a"), ("á", "a"),
        ("ô", "o"),  ("ö", "o"), ("ó", "o"),
        ("û", "u"),  ("ü", "u"), ("ú", "u"),
        ("î", "i"),  ("ï", "i"), ("í", "i"),
        ("ç", "c"),  ("ñ", "n"),
        (" ", " "),  ("​", ""),
        ("°", " deg"),("±", "+/-"),
    ]
    for src, dst in replacements:
        text = text.replace(src, dst)
    text = unicodedata.normalize("NFKD", text)
    return text.encode("latin-1", errors="ignore").decode("latin-1")

def generate_pdf_report(question, time_ans, contra_ans, impact_ans, verify_ans, final_ans):
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_fill_color(30, 30, 60)
    pdf.rect(0, 0, 210, 28, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 16)
    pdf.set_xy(10, 8)
    pdf.cell(0, 10, "Bangladesh Education Policy -- GraphRAG Report", ln=True)
    pdf.set_font("Arial", "", 9)
    pdf.set_xy(10, 19)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Powered by LLaMA 3.1 + Multi-Agent RAG", ln=True)

    pdf.set_text_color(0, 0, 0)
    pdf.ln(6)

    # Question box
    pdf.set_fill_color(240, 244, 255)
    pdf.set_draw_color(100, 120, 200)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 8, "Question", ln=True, fill=True, border="B")
    pdf.set_font("Arial", "", 11)
    pdf.set_fill_color(248, 250, 255)
    pdf.multi_cell(0, 7, clean_text(question), border=1, fill=True)
    pdf.ln(5)

    # Section helper
    section_colors = {
        "Timeline Analysis":    (230, 245, 230),
        "Contradictions Found": (255, 243, 230),
        "Impact Chain":         (230, 240, 255),
        "Verification Report":  (245, 230, 245),
        "Final Synthesized Answer": (220, 255, 220),
    }

    def add_section(title, content, emoji=""):
        r, g, b = section_colors.get(title, (245, 245, 245))
        pdf.set_fill_color(r, g, b)
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, f"{emoji}  {title}", ln=True, fill=True, border="B")
        pdf.set_font("Arial", "", 10)
        pdf.set_fill_color(250, 250, 250)
        pdf.multi_cell(0, 6, clean_text(content), border=1, fill=True)
        pdf.ln(4)

    add_section("Timeline Analysis",       time_ans,    "")
    add_section("Contradictions Found",    contra_ans,  "")
    add_section("Impact Chain",            impact_ans,  "")
    add_section("Verification Report",     verify_ans,  "")
    add_section("Final Synthesized Answer", final_ans,  "")

    # Footer
    pdf.set_y(-15)
    pdf.set_font("Arial", "I", 8)
    pdf.set_text_color(130, 130, 130)
    pdf.cell(0, 10, "Generated by Bangladesh Policy GraphRAG Assistant -- Multi-Agent AI System", align="C")

    return bytes(pdf.output(dest="S"))

def get_pdf_download_link(pdf_bytes, filename):
    b64 = base64.b64encode(pdf_bytes).decode()
    return f'<a href="data:application/pdf;base64,{b64}" download="{filename}" style="display:inline-block;padding:10px 20px;background:#1f6aa5;color:white;border-radius:6px;text-decoration:none;font-weight:bold;font-size:15px;">📥 Download PDF Report</a>'

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    secret_key = st.secrets.get("GROQ_API_KEY", "")
    if secret_key:
        api_key = secret_key
        st.success("🔑 API key loaded!")
    else:
        api_key = st.text_input("Groq API Key", type="password")
        st.markdown("Get a free key at [console.groq.com](https://console.groq.com)")

    st.divider()

    st.markdown("### 📄 Upload Your Own PDFs")
    uploaded_files = st.file_uploader(
        "Upload policy PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("⚙️ Process PDFs", type="secondary"):
            with st.spinner("Extracting text from PDFs..."):
                new_chunks = []
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name
                    try:
                        reader    = PyPDF2.PdfReader(tmp_path)
                        full_text = ""
                        for page in reader.pages:
                            full_text += page.extract_text() + " "
                    except Exception as e:
                        st.error(f"Could not read {uploaded_file.name}: {e}")
                        continue
                    finally:
                        os.unlink(tmp_path)

                    words   = full_text.split()
                    start   = 0
                    chunk_i = 0
                    while start < len(words):
                        end   = min(start + 100, len(words))
                        chunk = " ".join(words[start:end])
                        new_chunks.append({
                            "chunk_id": f"upload_{uploaded_file.name}_{chunk_i}",
                            "doc_id"  : f"upload_{uploaded_file.name}",
                            "year"    : 0,
                            "title"   : uploaded_file.name,
                            "text"    : chunk
                        })
                        start   += 80
                        chunk_i += 1

                if new_chunks:
                    texts      = [c["text"]     for c in new_chunks]
                    ids        = [c["chunk_id"] for c in new_chunks]
                    metadatas  = [{"doc_id": c["doc_id"],
                                   "year"  : c["year"],
                                   "title" : c["title"]} for c in new_chunks]
                    embeddings = embed_model.encode(texts).tolist()
                    collection.add(
                        ids        = ids,
                        embeddings = embeddings,
                        documents  = texts,
                        metadatas  = metadatas
                    )
                    st.success(f"✅ Added {len(new_chunks)} chunks from {len(uploaded_files)} PDF(s)!")
                    st.caption("Now ask questions — your PDFs are included!")

    st.divider()
    st.markdown("### 🤖 Active Agents")
    st.markdown("🕐 **TimeAgent** — tracks changes over time")
    st.markdown("⚠️ **ContradictionAgent** — finds conflicts")
    st.markdown("💡 **ImpactAgent** — traces cause & effect")
    st.markdown("🛡️ **GroundingAgent** — verifies claims")

# ── Graph helpers ─────────────────────────────────────────
def build_graph(extracted_data):
    G = nx.DiGraph()
    for doc_id, data in extracted_data.items():
        for entity in data["entities"]:
            name  = entity["name"].strip()
            etype = entity.get("type", "OTHER").upper()
            if name and len(name) > 1:
                G.add_node(name,
                           entity_type=etype,
                           color=entity_colors.get(etype, "#9E9E9E"),
                           doc_year=data["year"])
        for rel in data["relationships"]:
            src = rel.get("source", "").strip()
            tgt = rel.get("target", "").strip()
            if src and tgt and src in G.nodes and tgt in G.nodes:
                G.add_edge(src, tgt,
                           relation=rel.get("relation", "related_to"),
                           year=data["year"])
    return G

def draw_graph(G):
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_title("Knowledge Graph — Policy Entities & Relationships",
                 fontsize=14, fontweight="bold", pad=15)
    pos         = nx.spring_layout(G, k=3, iterations=50, seed=42)
    node_colors = [G.nodes[n].get("color", "#9E9E9E") for n in G.nodes()]
    node_sizes  = [400 + G.degree(n) * 300 for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=node_sizes, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="#AAAAAA",
                           arrows=True, arrowsize=15,
                           alpha=0.6, width=1.5, ax=ax)
    labels = {n: n for n, d in G.degree() if d >= 2}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight="bold", ax=ax)
    patches = [mpatches.Patch(color=c, label=t)
               for t, c in entity_colors.items() if t != "OTHER"]
    ax.legend(handles=patches, loc="upper left", fontsize=8)
    ax.axis("off")
    plt.tight_layout()
    return fig

# ── Helpers ───────────────────────────────────────────────
def retrieve(query, n=4):
    emb = embed_model.encode([query]).tolist()
    res = collection.query(query_embeddings=emb, n_results=n)
    return [{"text" : res["documents"][0][i],
             "year" : res["metadatas"][0][i]["year"],
             "title": res["metadatas"][0][i]["title"]}
            for i in range(len(res["documents"][0]))]

def ask_llm(client, system, user, max_tokens=250):
    r = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        temperature=0.3,
        max_tokens=max_tokens
    )
    return r.choices[0].message.content

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2 = st.tabs(["💬 Ask Agents", "🕸️ Knowledge Graph"])

# ══════════════════ TAB 1 ════════════════════════════════
with tab1:
    st.divider()
    question = st.text_input(
        "💬 Ask a question about Bangladesh education policy:",
        placeholder="e.g. How did girls enrollment change between 2015 and 2024?"
    )
    st.caption("💡 Try: How did girls enrollment change between 2015 and 2024? | Are there any budget vs enrollment target conflicts? | What was the impact of the Girls Stipend Program?")

    if st.button("🔍 Analyze", type="primary", disabled=not api_key):
        if not question:
            st.warning("Please enter a question!")
        else:
            client = Groq(api_key=api_key)
            col1, col2 = st.columns(2)

            with col1:
                with st.spinner("🕐 TimeAgent thinking..."):
                    ctx      = retrieve(question + " change over time year")
                    ctx_text = "\n\n".join([f"[{c['year']}] {c['text']}" for c in ctx])
                    time_ans = ask_llm(client,
                        "You are a TimeAgent. Analyze changes over time. Mention specific years. Under 120 words.",
                        f"Question: {question}\n\nContext:\n{ctx_text}")
                st.subheader("🕐 Timeline Analysis")
                st.write(time_ans)

                with st.spinner("⚠️ ContradictionAgent thinking..."):
                    ctx2      = retrieve(question + " conflict contradiction inconsistency")
                    ctx_text2 = "\n\n".join([f"[{c['year']}] {c['text']}" for c in ctx2])
                    contra_ans = ask_llm(client,
                        "You are a ContradictionAgent. Find contradictions between documents. Say CONTRADICTION: if found. Under 120 words.",
                        f"Question: {question}\n\nContext:\n{ctx_text2}")
                st.subheader("⚠️ Contradictions")
                st.write(contra_ans)

            with col2:
                with st.spinner("💡 ImpactAgent thinking..."):
                    ctx3      = retrieve(question + " impact outcome result caused")
                    ctx_text3 = "\n\n".join([f"[{c['year']}] {c['text']}" for c in ctx3])
                    impact_ans = ask_llm(client,
                        "You are an ImpactAgent. Trace cause-and-effect chains. Under 120 words.",
                        f"Question: {question}\n\nContext:\n{ctx_text3}")
                st.subheader("💡 Impact Chain")
                st.write(impact_ans)

                with st.spinner("🛡️ GroundingAgent verifying..."):
                    combined  = time_ans + " " + contra_ans + " " + impact_ans
                    ctx4      = retrieve(question)
                    ctx_text4 = "\n\n".join([f"[{c['year']}] {c['text']}" for c in ctx4])
                    verify_ans = ask_llm(client,
                        "You are a GroundingAgent. Verify claims with VERIFIED, UNVERIFIED, or PARTIAL. Under 150 words.",
                        f"Answer:\n{combined}\n\nSources:\n{ctx_text4}")
                st.subheader("🛡️ Verification Report")
                st.write(verify_ans)

            st.divider()
            with st.spinner("📝 Writing final answer..."):
                final = ask_llm(client,
                    "Synthesize the multi-agent analysis into one clean answer. Sections: Timeline, Contradictions, Impact, Confidence. Under 300 words.",
                    f"Question: {question}\n\nTime: {time_ans}\n\nContradictions: {contra_ans}\n\nImpact: {impact_ans}\n\nVerification: {verify_ans}",
                    max_tokens=400)
            st.subheader("📊 Final Synthesized Answer")
            st.success(final)

            # ── Option E: PDF Export ──────────────────────
            st.divider()
            st.subheader("📥 Export Report")
            with st.spinner("Generating PDF report..."):
                try:
                    pdf_bytes = generate_pdf_report(
                        question, time_ans, contra_ans,
                        impact_ans, verify_ans, final
                    )
                    filename  = f"policy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    st.markdown(
                        get_pdf_download_link(pdf_bytes, filename),
                        unsafe_allow_html=True
                    )
                    st.caption("Click the button above to download your full analysis as a PDF.")
                except Exception as e:
                    st.error(f"PDF generation failed: {e}. Make sure fpdf2 is in requirements.txt")

    elif not api_key:
        st.info("👈 Enter your Groq API key in the sidebar to get started.")

# ══════════════════ TAB 2 ════════════════════════════════
with tab2:
    st.subheader("🕸️ Entity Knowledge Graph")
    st.caption("Extracted from all 5 policy documents — nodes sized by number of connections")

    if not api_key:
        st.info("👈 Enter your Groq API key in the sidebar to build the graph.")
    else:
        if st.button("🔨 Build Knowledge Graph", type="primary"):
            client = Groq(api_key=api_key)

            with st.spinner("Extracting entities from all 5 documents... (~30 seconds)"):
                extracted_data = {}
                progress = st.progress(0)

                for i, doc in enumerate(policy_documents):
                    prompt = f"""Extract entities and relationships from this policy document (Year: {doc['year']}).
Return ONLY valid JSON:
{{
    "entities": [{{"name": "entity name", "type": "PROGRAM/PERSON/ORGANIZATION/BUDGET/POLICY/METRIC"}}],
    "relationships": [{{"source": "name1", "relation": "relation_type", "target": "name2"}}]
}}
Document: {doc['content']}
Return only JSON, nothing else."""

                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=400
                    )
                    raw = re.sub(r'```json|```', '', response.choices[0].message.content).strip()
                    try:
                        result = json.loads(raw)
                    except Exception:
                        result = {"entities": [], "relationships": []}

                    extracted_data[doc["id"]] = {
                        "year"         : doc["year"],
                        "title"        : doc["title"],
                        "entities"     : result.get("entities", []),
                        "relationships": result.get("relationships", [])
                    }
                    progress.progress((i + 1) / len(policy_documents))

            G = build_graph(extracted_data)

            if G.number_of_nodes() == 0:
                st.error("⚠️ No entities extracted. Try clicking Build again.")
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Entities",      G.number_of_nodes())
                col2.metric("Total Relationships", G.number_of_edges())
                top_node = sorted(G.degree(), key=lambda x: x[1], reverse=True)[0]
                col3.metric("Most Connected Node", f"{top_node[0]} ({top_node[1]} links)")

                fig = draw_graph(G)
                st.pyplot(fig)

                st.subheader("🏆 Top 10 Most Connected Entities")
                top10      = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:10]
                table_data = [{"Entity"     : n,
                               "Type"       : G.nodes[n].get("entity_type", "UNKNOWN"),
                               "Connections": d} for n, d in top10]
                st.table(table_data)