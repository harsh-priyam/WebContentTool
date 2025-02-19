import streamlit as st
from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI,AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Page configuration
st.set_page_config(
    page_title="URL Content Q&A Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
 <style>
/* Button styling */
.stButton > button {
    width: 100%;
    border-radius: 20px;
    height: 2.5rem;
}

/* Container styling */
.st-emotion-cache-18ni7ap {
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
}

/* Main container padding */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Headings for both modes */
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    color: inherit;
}

/* URL count styling */
.url-count {
    padding: 10px;
    border-radius: 10px;
    text-align: center;
    margin: 10px 0;
    background-color: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(128, 128, 128, 0.2);
}

/* Response container styling */
.response-container {
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
    background-color: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(128, 128, 128, 0.2);
}

/* Answer title styling */
.answer-title {
    color: inherit !important;
    margin-bottom: 1rem;
    font-size: 1.1rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vectordbs' not in st.session_state:
    st.session_state.vectordbs = []

def create_embeddings(url: str):
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    for script in soup(["script", "style", "meta", "link"]):
        script.decompose()
    
    text = soup.get_text()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    vectordb = FAISS.from_texts(
        chunks,
        embeddings
    )
    return vectordb

def get_context(query: str, vectordbs):
    context = []
    for vectordb in vectordbs:
        retriever = vectordb.as_retriever(search_type='mmr', search_kwargs={'k': 4})
        docs = retriever.invoke(query)
        for doc in docs:
            context.append(doc.page_content)
    return str(context)

def chat(query: str, context: str):
    prompt = """You are a professional chatbot that can answer any queries related to the context provided. Make sure your 
    response should be highly professional and upto the mark, you are not supposed to generate anything which is harmful and 
    should not hurt anyone's sentiments and also dont answer anything which is out of context,and just responds with "Please ask question related to the urls provided"
    Context: {context}
    Query: {query}
    """
    
    llm = ChatOpenAI(temperature=0.2, model_name="gpt-4o-mini", api_key=OPENAI_API_KEY)
    
    PROMPT = PromptTemplate(
        template=prompt,
        input_variables=["context", "query"],
    )
    
    llm_chain = PROMPT | llm | StrOutputParser()
    llm_response = llm_chain.invoke({"query": query, "context": context})
    return llm_response

# Main UI
st.title("🤖 URL Content Q&A Bot")
st.markdown("---")

# Create two main columns for better layout
left_col, right_col = st.columns([2, 1])

with left_col:
    # URL Input Section
    st.header("📚 Add Content Sources")
    
    # URL input and button in the same row
    url_col1, url_col2 = st.columns([3, 1])
    
    with url_col1:
        url_input = st.text_input("Enter a URL", placeholder="https://example.com", key="url_input")
    
    with url_col2:
        add_url = st.button("Add URL", key="add_url", use_container_width=True)

    # Process URL
    if add_url and url_input:
        with st.spinner("🔄 Creating embeddings for the URL..."):
            try:
                vectordb = create_embeddings(url_input)
                st.session_state.vectordbs.append(vectordb)
                st.success("✅ Successfully added URL!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    elif add_url:
        st.warning("⚠️ Please enter a URL first")

    # Display added URLs count with nice formatting
    if st.session_state.vectordbs:
        st.markdown(f"""
            <div class="url-count">
                📑 Number of URLs processed: {len(st.session_state.vectordbs)}
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Query Section
    st.header("❓ Ask Questions")
    
    query_col1, query_col2 = st.columns([3, 1])
    
    with query_col1:
        query = st.text_input("Enter your question", placeholder="What would you like to know?")
    
    with query_col2:
        send_query = st.button("Send Query", use_container_width=True)

    # Process Query
    if send_query:
        if query and st.session_state.vectordbs:
            with st.spinner("🤔 Thinking..."):
                try:
                    context = get_context(query, st.session_state.vectordbs)
                    response = chat(query, context)
                    st.markdown("""
                        <div class="response-container">
                            <h3>💡 Answer:</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    st.write(response)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        elif not query:
            st.warning("⚠️ Please enter a question first")
        else:
            st.warning("⚠️ Please add at least one URL before asking questions")

with right_col:
    st.header("🛠️ Controls")
    
    # Clear URLs button with confirmation
    if st.button("🗑️ Clear All URLs", use_container_width=True):
        st.session_state.vectordbs = []
        st.success("🧹 All URLs cleared!")
        st.rerun()
    
    # Help section
    st.markdown("### 📖 How to Use")
    st.markdown("""
    1. 🔗 Enter a URL in the input box
    2. ➕ Click 'Add URL' to process it
    3. 🔄 Add multiple URLs as needed
    4. ❓ Type your question
    5. 🚀 Click 'Send Query' for answers
    6. 🗑️ Use 'Clear All URLs' to reset
    
    > **Note**: Ensure your OpenAI API key is properly configured in the `.env` file
    """)
    
    # Additional information
    st.markdown("### 🎯 Features")
    st.markdown("""
    - 📊 Process multiple URLs
    - 🔍 Smart context retrieval
    - 🤖 AI-powered responses
    - 📱 Mobile-friendly interface
    """)