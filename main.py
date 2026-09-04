import os
from dotenv import load_dotenv

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Initialising components...")

embeddings = OpenAIEmbeddings(api_key=os.environ.get("OPEN_API_KEY"))
llm = ChatOpenAI(api_key=os.environ.get("OPEN_API_KEY"))
vectorstore = PineconeVectorStore(
    index_name=os.environ.get("INDEX_NAME"),
    embedding=embeddings
)
# this allows us to retrieve elements from the vector store
retriever = vectorstore.as_retriever(search_kwargs={'k':3}) # k = 3 limits retrieval to top 3 most relevant chunks
# create prompt which uses grounding
prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
    
    {context}

    Question: {question}

    Provide detailed answer:
    """
)

# helper function to receive langchain docs from vector store and 
# combine into single string to be inputted into prompt as {context}
def format_docs(docs):
    """Format retreived documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

### IMPLEMENTATION 1 - Without LangChain Expression Language
# helper function to receive a string and return an LLM response
# done without LangChain Expression Language (to show how it works)
def retrieval_chain_without_lcel(query:str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves docs, formats them, and generates response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """

    # Step 1: Retrieve relevant docs
    docs = retriever.invoke(query) # retriever is a runnable objects, meaning it can be use using .invoke
    # Step 2: Format into 1 string
    context =  format_docs(docs)
    # Step 3: Format the prompt with the question and context
    messages = prompt_template.format_messages(context=context, question=query)
    # Step 4: Obtain LLM response
    response = llm.invoke(messages)
    return response.content

### IMPLEMENTATION 2 - With full LangChain implementation
# create chain function
def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """

    retrieval_chain = (
        RunnablePassthrough.assign( # this creates a dict that combines the original input with the new computed field, that will be explicitly mentioned here
            context=itemgetter("question") | retriever | format_docs  # this is an inside chain that returns the relevant vector store chunks, based on the input question
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain

if __name__ == "__main__":
    print("Retrieving...")

    # set query
    query = "what is Pinecone in machine learning?"

    ### IMPLEMENTATION 1
    # print("\n" + "=" * 70)
    # print("IMPLEMENTATION 1: Without LCEL")
    # print("=" * 70)
    # result_without_lcel = retrieval_chain_without_lcel(query)
    # print("\nAnswer:")
    # print(result_without_lcel)

    ### IMPLEMENTATION 2
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 70)
    print("Why LCEL is better:")
    print("- More concise and declarative")
    print("- Built-in streaming: chain.stream()")
    print("- Built-in async: chain.ainvoke()")
    print("- Easy to compose with other chains")
    print("- Better for production use")
    print("=" * 70)
    # create chain
    chain_with_lcel = create_retrieval_chain_with_lcel()
    # run chain
    result_with_lcel = chain_with_lcel.invoke({"question":query})
    print("\nAnswer:")
    print(result_with_lcel)