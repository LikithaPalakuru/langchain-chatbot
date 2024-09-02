import streamlit as st
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os


load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

DOCUMENT_PATH = "documents/python.pdf"

@st.cache_resource
def load_and_process_document():
    loader = PyPDFLoader(DOCUMENT_PATH)
    data = loader.load()

    if not data:
        st.error("No data found in the document. Please check the document path and content.")
        return None, None, None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    
    texts = []
    metadata = []
    for page_num, page in enumerate(data):
        content = page.page_content
        chunks = text_splitter.split_text(content)
        for chunk in chunks:
            texts.append(chunk)
            metadata.append({'page': page_num, 'pdf': DOCUMENT_PATH, 'content': chunk})

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(texts, embedding=embeddings)
    vector_store.save_local("faiss_index")

    return vector_store, metadata, texts

vector_store, metadata, texts = load_and_process_document()

if not vector_store:
    st.error("Failed to initialize vector store.")

prompt_template = """
You are an expert document analyser. Answer the question as detailed as possible from the provided context,
make sure to provide all the details, if the answer is not in the context
just say 'answer is not available in the context', don't provide the wrong answer
\n\n
Context: \n{context}\n
Question: \n{question}\n
Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

if not chain:
    st.error("Failed to load QA chain.")

st.title("Document Question Answering System")

question = st.text_input("Enter your question:")

if question:
    st.write(f"Processing question: {question}")

    docs = vector_store.similarity_search(question)

    if not docs:
        st.write("No relevant documents found.")
    else:
        relevant_metadata = []
        for doc in docs:
            
            try:
                index = texts.index(doc.page_content)
                relevant_metadata.append(metadata[index])
            except ValueError:
                st.error("Chunk not found in the texts list.")

        sorted_metadata = sorted(relevant_metadata, key=lambda x: x['page'])

        
        context = "\n\n".join([doc.page_content for doc in docs])
        response = chain.invoke(
            {"input_documents": docs, "question": question},
            return_only_outputs=True
        )

        st.write("Answer:")
        st.write(response)

        st.write("Document Metadata:")
        for info in sorted_metadata:
            st.write(f"PDF: {info['pdf']}, Page: {info['page']}")