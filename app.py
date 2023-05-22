# Import os to set API key
import os
# Import OpenAI as main LLM service
from langchain.llms import OpenAI
# Bring in streamlit for UI/app interface
import streamlit as st

# Import PDF document loaders...there's other ones as well!
from langchain.document_loaders import PyPDFLoader
# Import chroma as the vector store 
from langchain.vectorstores import Chroma

# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

st.header("LangChain : ")
st.image("home.png" , width=300)
st.write("""
LangChain is a framework for developing applications powered by language models. We believe that the most powerful and differentiated
 applications will not only call out to a language model, but will also be:

Data-aware: connect a language model to other sources of data

Agentic: allow a language model to interact with its environment

https://python.langchain.com/en/latest/index.html
""")

with st.sidebar: 
    st.image("icon.jpg", width=200)
    # Set APIkey for OpenAI Service
    # Can sub this out for other LLM providers
    try : 
        os.environ['OPENAI_API_KEY'] = st.text_input("Entre Your OPENAI_API_KEY here : ")
    except : 
        pass

    pdf_file = st.file_uploader("Select Your PDF file here : " , type=["pdf"])

    start = st.button("Click To start")

if pdf_file :
    if start :     
        # Create instance of OpenAI LLM
        llm = OpenAI(temperature=0.1, verbose=True)

        # Create and load PDF Loader
        loader = PyPDFLoader(pdf_file.names)
        # Split pages from pdf 
        pages = loader.load_and_split()
        # Load documents into vector database aka ChromaDB
        store = Chroma.from_documents(pages, collection_name='annualreport')

        # Create vectorstore info object - metadata repo?
        vectorstore_info = VectorStoreInfo(
            name="annual_report",
            description="a banking annual report as a pdf",
            vectorstore=store
        )
        # Convert the document store into a langchain toolkit
        toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

        # Add the toolkit to an end-to-end LC
        agent_executor = create_vectorstore_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True
        )
        st.title('🦜🔗 GPT Investment Banker')
        # Create a text input box for the user
        prompt = st.text_input('Input your prompt here')

        # If the user hits enter
        if prompt:
            # Then pass the prompt to the LLM
            response = agent_executor.run(prompt)
            # ...and write it out to the screen
            st.write(response)

            # With a streamlit expander  
            with st.expander('Document Similarity Search'):
                # Find the relevant pages
                search = store.similarity_search_with_score(prompt) 
                # Write out the first 
                st.write(search[0][0].page_content) 