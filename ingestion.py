import os
from dotenv import load_dotenv

# to load any sort of unstructured text into a document
from langchain_unstructured import UnstructuredLoader
# to split document into chunks
from langchain_text_splitters import CharacterTextSplitter
# to create embeddings
from langchain_openai import OpenAIEmbeddings
# to store embedding into vector stores
from langchain_pinecone import PineconeVectorStore


load_dotenv()

if __name__ == '__main__':

    print("Ingesting...")
    # create loader object with path to text file
    loader = UnstructuredLoader(file_path="/Users/cosminmanolescu/Desktop/Work/Digital_Futures/df-frontier/LangChain_Udemy/langchain-course/mediumblog1.txt",
                                chunking_strategy='basic',
                                max_characters=1000000)
    # create langchain document
    document = loader.load()

    print("splitting...")
    # create splitter object
    text_splitter = CharacterTextSplitter(chunk_size=1000,
                                          chunk_overlap=0)
    # split document
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    print("embedding...")
    # create embedding object
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPEN_API_KEY"))
    # ingest into Pinecone vector store
    PineconeVectorStore.from_documents(texts,
                                       embeddings,
                                       index_name=os.environ.get("INDEX_NAME") )

    print('finish')