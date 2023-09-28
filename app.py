"""
It connects to the Weaviate demo databse containing 10M wikipedia vectors
Wikipedia articles get chunked by paragraph, and each chunk gets assigned an embedding vector
"""
import streamlit as st
import wikipedia

st.set_page_config(
    page_title="Wikipedia Semantic Engine",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Built by @dcarpintero with Streamlit, Cohere and Weaviate"},
)

wikisearch = wikipedia.SearchEngine()

@st.cache_data
def query_bm25(query, lang='en', top_n=10):
   return wikisearch.with_bm25(query, lang=lang, top_n=top_n)

@st.cache_data
def query_neartext(query, lang='en', top_n=10):
   return wikisearch.with_neartext(query, lang=lang, top_n=top_n)

@st.cache_data
def query_hybrid(query, lang='en', top_n=10):
   return wikisearch.with_hybrid(query, lang=lang, top_n=top_n)

def query_with_llm(context, query, temperature, model):
   response = wikisearch.with_llm(context=context, query=query, temperature=temperature, model=model)
   return response.generations[0].text

def onchange_with_near_text():
    if st.session_state.with_near_text:
        st.session_state.with_bm25 = False
        st.session_state.with_hybrid = False

def onchange_with_bm25():
    if st.session_state.with_bm25:
        st.session_state.with_near_text = False
        st.session_state.with_hybrid = False

def onchange_with_hybrid():
    if st.session_state.with_hybrid:
        st.session_state.with_near_text = False
        st.session_state.with_bm25 = False

languages = {
    'Arabic': 'ar',
    'Chinese': 'zh',
    'English': 'en',
    'French': 'fr',
    'German': 'de',
    'Hindi': 'hi',
    'Italian': 'it',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Spanish': 'es'
}

with st.sidebar.expander("🤖 COHERE-SETTINGS", expanded=True):
    lang = st.selectbox("Language", list(languages.keys()), index=2)
    lang_code = languages.get(lang)
    gen_model = st.selectbox("Generation Model", ["command", "command-light", "command-nightly"], key="gen-model", index=0)
    rank_model = st.selectbox("Rank Model", ["rerank-english-v2.0", "rerank-multilingual-v2.0"], key="rank-model", index=0)
    temperature = st.slider('Temperature', min_value=0.0, max_value=1.0, value=0.25, step=0.05)
    max_results = st.slider('Max Results', min_value=0,
                            max_value=15, value=7, step=1)

with st.sidebar.expander("🔧 WEAVIATE-SETTINGS", expanded=True):
    st.toggle('Near Text Search', key="with_near_text",
              on_change=onchange_with_near_text)
    st.toggle('BM25 Search', key="with_bm25", on_change=onchange_with_bm25)
    st.toggle('Hybrid Search',  key="with_hybrid",
              on_change=onchange_with_hybrid)

with st.expander(" ABOUT-THIS-APP", expanded=True):
    st.write("""
             - This Retrieval Augmented Generation App (RAG) uses the *Weaviate* database containing 10M Wikipedia embedding vectors.
             - Step 1: Pre-Search on *Weaviate* with Sparse Retrival (bm25), Dense Retrieval (neartext), or Hybrid Mode (bm25 + neartext).
             - Step 2: *Cohere Rank Model* re-organizes the Pre-Search by assigning a relevance score to each Pre-Search result given the query.
             - Step 3: *Cohere Generation Model* composes a response based on the ranked results.
             - Try your language and experiment with the settings!
             """)
    
query = st.text_input("Ask 'Wikipedia'", '')

if query:
    if st.session_state.with_near_text:
        data = query_neartext(query, lang=lang_code, top_n=max_results)
    elif st.session_state.with_bm25:
        data = query_bm25(query, lang=lang_code, top_n=max_results)
    elif st.session_state.with_hybrid:
        data = query_hybrid(query, lang=lang_code, top_n=max_results)
    else:
        st.info("ℹ️ Select your preferred Search Mode (Near Text, BM25 or Hybrid)!")
        st.stop()

    col1, col2, col3 = st.columns([1,1,1])

    with col1:
        st.subheader("🔎 Pre-Search")

        for doc in data:
            st.markdown(f'[{doc["title"]}]({doc["url"]}) "{doc["text"][:1000]}"')
            st.divider()

    with col2:
        st.subheader("🏆 Ranking")

        data_ranked = wikisearch.rerank(query=query, documents=data, top_n=max_results, model=rank_model)
        for idx, r in enumerate(data_ranked):
            doc = r.document
            st.write(f"[Document Rank: {idx+1}, Document Index: {r.index + 1}, Relevance Score: {r.relevance_score:.3f}]")
            st.markdown(f'[{doc["title"]}]({doc["url"]}) "{doc["text"][:1000]}"')
            st.divider()

    with col3:
        st.subheader("📝 Generation")

        with st.spinner("Deep Diving..."):
            r = query_with_llm(context=data_ranked, query=query, temperature=temperature, model=gen_model)
        st.write(r)
        
        with st.expander("📚 WIKIPEDIA-REFERENCES", expanded=True):
            st.info("Some references might appear duplicated while referring to different paragraphs of the same article.")
            for r in data_ranked:
                doc = r.document
                st.markdown(f'[{doc["title"]}]({doc["url"]}) [Score:{r.relevance_score:.3f}]')
            

