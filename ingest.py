from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import OllamaEmbeddings

DATA_PATH = 'data/'
DB_CHROMA_PATH = 'vectorstore/db_chroma'

# Create vector database
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    
    texts = text_splitter.split_documents(documents)

    # Create a Chroma vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text",  
                                  model_kwargs={'device': 'cpu'})

    db = Chroma.from_documents(texts, embeddings, persist_directory=DB_CHROMA_PATH)

if __name__ == "__main__":
    create_vector_db()