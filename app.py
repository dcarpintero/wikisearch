"""
Multilingual Retrieval Augmented Generation demo built with Cohere, Weaviate and Streamlit.
It implements Semantic Search on Wikipedia Articles using 10 million vector embeddings.

This demo illustrates a three step approach (Pre-Search, Rank, Generation):
- Step 1: Pre-Search on Weaviate with Sparse Retrival (bm25), Dense Retrieval (neartext), or Hybrid Mode (bm25 + neartext).
- Step 2: Cohere Rank Model re-organizes the Pre-Search by assigning a relevance score to each Pre-Search result given the query.
- Step 3: Cohere Generation Model composes a response based on the ranked results.

Author:
    @dcarpintero : https://github.com/dcarpintero
"""
import streamlit as st
import wikipedia

st.set_page_config(
    page_title="Wikipedia Semantic Search",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Built by @dcarpintero with Streamlit, Cohere and Weaviate"},
)

@st.cache_resource(show_spinner=False)
def load_semantic_engine():
    return wikipedia.SearchEngine()

wikisearch = load_semantic_engine()

@st.cache_data
def query_bm25(query, lang='en', top_n=10):
   return wikisearch.with_bm25(query, lang=lang, top_n=top_n)

@st.cache_data
def query_neartext(query, lang='en', top_n=10):
   return wikisearch.with_neartext(query, lang=lang, top_n=top_n)

@st.cache_data
def query_hybrid(query, lang='en', top_n=10):
   return wikisearch.with_hybrid(query, lang=lang, top_n=top_n)

def query_llm(context, query, temperature, model, lang="english"):
   response = wikisearch.with_llm(context=context, query=query, temperature=temperature, model=model, lang=lang)
   text = response.generations[0].text
   return text

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

def onclick_sample_query(query):
    st.write("onclick_sample_query: " + query)
    st.session_state.user_query_txt = query


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

samples = {
    'q1': 'Who invented the printing press, what was the key development for this?',
    'q2': 'What are the top 3 highest mountains in the world?', 
    'q3': 'Who was the first person to win two Nobel prizes?',
    'q4': 'When and in which year were celebrated the first olimpic games?',
}

# -----------------------------------------------------------------------------
# Sidebar Section
# -----------------------------------------------------------------------------
with st.sidebar.expander("🤖 COHERE-SETTINGS", expanded=True):
    lang = st.selectbox("Language", list(languages.keys()), index=2)
    lang_code = languages.get(lang)
    gen_model = st.selectbox("Generation Model", ["command", "command-light", "command-nightly"], key="gen-model", index=0)
    rank_model = st.selectbox("Rank Model", ["rerank-multilingual-v2.0", "rerank-english-v2.0"], key="rank-model", index=0)
    temperature = st.slider('Temperature', min_value=0.0, max_value=1.0, value=0.30, step=0.05)
    max_results = st.slider('Max Results', min_value=0,
                            max_value=15, value=10, step=1)

with st.sidebar.expander("🔧 WEAVIATE-SETTINGS", expanded=True):
    st.toggle('Dense Retrieval', key="with_near_text",
              on_change=onchange_with_near_text)
    st.toggle('Keyword Search', key="with_bm25", on_change=onchange_with_bm25)
    st.toggle('Hybrid Mode',  key="with_hybrid",
              on_change=onchange_with_hybrid)
    st.info("ℹ️ Note that *Dense Retrieval* and *Hybrid* outperform *Keyword Search* on complex queries!")

with st.expander("ℹ️ ABOUT-THIS-APP", expanded=False):
    st.write("""
             - Multilingual RAG built with *Cohere* and the *Weaviate* demo database containing 10M Wikipedia embedding vectors (October 9th, 2021).
             - Step 1: Pre-Search on *Weaviate* with Sparse Retrival (bm25), Dense Retrieval (neartext), or Hybrid Mode (bm25 + neartext).
             - Step 2: *Cohere Rank Model* re-organizes the Pre-Search by assigning a relevance score to each Pre-Search result given the query.
             - Step 3: *Cohere Generation Model* composes a response based on the ranked results.
             - Ask in your preferred language, and experiment with the settings!
             """)
    
with st.sidebar:
    col_gh, col_co, col_we = st.columns([1,1,1])
    with col_gh:
        "[![Github](https://img.shields.io/badge/Github%20Repo-gray?logo=Github)](https://github.com/dcarpintero/wise)"
    with col_co:
        "[![Cohere](https://img.shields.io/badge/Cohere%20LLMs-purple)](https://cohere.com/?ref=https://github.com/dcarpintero)"
    with col_we:
        "[![Weaviate](https://img.shields.io/badge/Weaviate-green)](https://weaviate.io/?ref=https://github.com/dcarpintero)"
        
    
# -----------------------------------------------------------------------------
# Ask Wikipedia Section
# -----------------------------------------------------------------------------
st.subheader("🪄 Wikipedia Semantic Search with Cohere Rerank")
query = st.text_input(label="Ask 'Wikipedia'", placeholder='Ask your question here, or select one from the examples below',  key="user_query_txt", label_visibility="hidden")

btn_printing = st.session_state.get("btn_printing", False)
btn_nobel = st.session_state.get("btn_nobel", False)
btn_internet = st.session_state.get("btn_internet", False)
btn_ai = st.session_state.get("btn_ai", False)

col1, col2, col3, col4 = st.columns([1,1,1,1])
with col1:
    if st.button(label=samples["q1"], type="primary", disabled=btn_printing, on_click=onclick_sample_query, args=[samples["q1"]]):
        st.session_state.btn_printing = True
with col2:
    if st.button(label=samples["q2"], type="primary", disabled=btn_nobel,  on_click=onclick_sample_query, args=[samples["q2"]]):
        st.session_state.btn_nobel = True
with col3:
    if st.button(label=samples["q3"], type="primary", disabled=btn_internet,  on_click=onclick_sample_query, args=[samples["q3"]]):
        st.session_state.btn_internet = True
with col4:
    if st.button(label=samples["q4"], type="primary", disabled=btn_ai,  on_click=onclick_sample_query, args=[samples["q4"]]):
        st.session_state.btn_ai = True

if query:
    if st.session_state.with_near_text:
        data = query_neartext(query, lang=lang_code, top_n=max_results)
    elif st.session_state.with_bm25:
        data = query_bm25(query, lang=lang_code, top_n=max_results)
    elif st.session_state.with_hybrid:
        data = query_hybrid(query, lang=lang_code, top_n=max_results)
    else:
        st.info("ℹ️ Select your preferred Search Mode (Dense Retrieval, Keyword Search, or Hybrid)!")
        st.stop()

    st.divider()
    col1, col2, col3 = st.columns([1,1,1])

    with col1:
        st.subheader("🔎 1. Pre-Search")

        for idx, doc in enumerate(data):
            with st.expander(f'**{doc["title"]} [Rank: {idx+1}**]', expanded=False):
                st.markdown(f'"*{doc["text"][:800]} [...]*" [Source]({doc["url"]})')

    with col2:
        st.subheader("🏆 2. Ranking")

        data_ranked = wikisearch.rerank(query=query, documents=data, top_n=max_results, model=rank_model)
        for idx, r in enumerate(data_ranked):
            doc = r.document
            expanded = False
            if idx == 0:
                expanded = True
            with st.expander(f'**{doc["title"]} [Previous Rank: {r.index + 1} - Relevance: {r.relevance_score:.3f}**]', expanded=expanded):
                st.markdown(f'"*{doc["text"][:800]} [...]*" [Source]({doc["url"]})')

    with col3:
        st.subheader("📝 3. LLM Generation")

        with st.spinner("Deep Diving..."):
            r = query_llm(context=data_ranked[:5], query=query, temperature=temperature, model=gen_model, lang=lang)
        st.success(f"🪄 {r}")
        st.info("ℹ️ Some references might appear to be duplicated while referring to different paragraphs of the same article.")