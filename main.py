from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def create_embeddings(url:str):
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


def get_context(query:str):
    context = []
    for vectordb in vectordbs:
        retriever = vectordb.as_retriever(search_type='mmr',search_kwargs={'k': 4})
        docs = retriever.invoke(query)
        for doc in docs:
            context.append(doc.page_content)
    return str(context)

def chat(query: str , context:str):

    prompt = """You are a professional chatbot that can answer any queries related to the context provided. Make sure your
    response should be highly professional and upto the mark,you are not supposed to generate anything which is harmful and
    should not hurt anyone's sentiments.
    Context : {context}
    Query: {query}
    """
    llm = ChatOpenAI(temperature=0.2, model_name="gpt-4o-mini",api_key=OPENAI_API_KEY)

    PROMPT = PromptTemplate(
        template=prompt,
        input_variables=["context","query"],
    )

    llm_chain = PROMPT | llm | StrOutputParser()

    llm_response = llm_chain.invoke({"query": query, "context":context})

    return llm_response


if __name__ == '__main__':
    urls = []
    
    while True:
        url = input("Enter a web URL for scraping (or type 'done' to finish): ").strip()
        if url.lower() == 'done':
            break
        urls.append(url)

    if not urls:
        print("No URLs were provided. Exiting.")
        exit()

    print("Please wait! Creating embeddings for the given URLs...")

    vectordbs = []
    for url in urls:
        vectordb = create_embeddings(url)
        vectordbs.append(vectordb)

    print("Embeddings have been created successfully!")

    while True:
        query = input("Please enter your query from the URLs you have provided 😊: ")
        if query == "exit":
             exit()
        context = get_context(query)  
        response = chat(query, context)
        print(response)
        
             

