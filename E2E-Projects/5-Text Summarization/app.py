import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

## streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website")
st.title("LangChain: Summarize Text From YT or Website")

st.subheader('Summarize URL')

## Get the Groq API
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")
    
generic_url = st.text_input("URL", label_visibility="collapsed")

## Gemma Model Using Groq API
llm = ChatGroq(groq_api_key=groq_api_key, model="Gemma-7b-It")

prompt_template = """
Provide a summary of following content in 300 words:
Content: {text}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["text"]
)

if st.button("Summarize the Content from YT or Website"):
    ## Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information")
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url. It can may be a YT video or website url")
        
    else:
        try:
            with st.spinner("Waiting..."):
                ## loading the website or yt video data
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url], 
                        ssl_verify=False, 
                        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"})
                    docs = loader.load()
                    
                    ##Chain for Summarization
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary=chain.run(docs)
                    
                    st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception: {e}")