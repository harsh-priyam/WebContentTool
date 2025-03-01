from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)
vectordbs = []

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
    
    vectordb = FAISS.from_texts(chunks, embeddings)
    return vectordb


def get_context(query: str):
    context = []
    for vectordb in vectordbs:
        retriever = vectordb.as_retriever(search_type='mmr', search_kwargs={'k': 4})
        docs = retriever.invoke(query)
        for doc in docs:
            context.append(doc.page_content)
    return str(context)


def chat(query: str, context: str):
    prompt = """You are a professional chatbot that can answer any queries related to the context provided. 
    Make sure your response is highly professional and accurate.
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


@app.route('/add_urls', methods=['POST'])
def add_urls():
    global vectordbs
    data = request.json
    urls = data.get('urls', [])
    
    if not urls:
        return jsonify({"error": "No URLs provided."}), 400
    
    for url in urls:
        vectordb = create_embeddings(url)
        vectordbs.append(vectordb)
    
    return jsonify({"message": "Embeddings created successfully."})


@app.route('/query', methods=['POST'])
def query_chatbot():
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({"error": "Query is required."}), 400
    
    context = get_context(query)
    response = chat(query, context)
    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(debug=True)
