"""
It connects to the Weaviate demo databse containing 10M wikipedia vectors
Wikipedia articles get chunked by paragraph, and each chunk gets assigned an embedding vector
"""
import streamlit as st
import wikipedia

st.set_page_config(
    page_title="Wikipedia Semantic Engine",
    page_icon="🦙",
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

def query_with_llm(context, query):
   response = wikisearch.with_llm(context=context, query=query)
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

with st.sidebar.expander(" WIKIPEDIA-SETTINGS", expanded=True):
    lang = st.selectbox("language", list(languages.keys()), index=2, label_visibility="hidden")
    lang_code = languages.get(lang)
    st.write(f"Selected language: {lang_code}")
    max_results = st.slider('Max Results', min_value=0,
                            max_value=100, value=10, step=1)

with st.sidebar.expander("🔧 WEAVIATE-SETTINGS", expanded=True):
    st.toggle('Near Text Search', key="with_near_text",
              on_change=onchange_with_near_text)
    st.toggle('BM25 Search', key="with_bm25", on_change=onchange_with_bm25)
    st.toggle('Hybrid Search',  key="with_hybrid",
              on_change=onchange_with_hybrid)

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
        st.subheader("Step 1: Search Results")

        for doc in data:
            st.markdown(f'[{doc["title"]}]({doc["url"]}) "{doc["text"][:1000]}"')
            st.divider()

    with col2:
        st.subheader("Step 2: Ranked Results")

        data_ranked = wikisearch.rerank(query=query, documents=data)
        for idx, r in enumerate(data_ranked):
            doc = r.document
            st.write(f"[Document Rank: {idx+1}, Document Index: {r.index + 1}, Relevance Score: {r.relevance_score:.2f}]")
            st.markdown(f'[{doc["title"]}]({doc["url"]}) "{doc["text"][:1000]}"')
            st.divider()

    with col3:
        st.subheader("Step 3: LLM Answer")

        with st.spinner("Querying LLM..."):
            r = query_with_llm(context=data_ranked, query=query)
        st.write(r)
        st.divider()
        
        st.write("Wikipedia References:")
        for r in data_ranked:
            doc = r.document
            st.markdown(f'[{doc["title"]}]({doc["url"]}) [Score:{r.relevance_score}]')
            

