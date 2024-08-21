from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set environment variables for LangChain
os.environ['LANGCHAIN_TRACING_V2'] = ''
api_key = os.getenv('LANGCHAIN_API_KEY')

if not api_key:
    st.error("LANGCHAIN_API_KEY is not set. Please check your .env file.")
else:
    os.environ['LANGCHAIN_API_KEY'] = api_key

    # Prompt template
    prompt = ChatPromptTemplate(
        [
            ("system", "You are a helpful assistant for making Flutter apps. Please respond to the user's queries."),
            ("user", "Question: {question}")
        ]
    )

    # Streamlit framework
    st.title('Chat Bot using Ollama')
    input_text = st.text_input('Ask about Flutter code')

    # Initialize Ollama model
    llm = Ollama(model='gemma:2b')

    # Output parser
    output_parser = StrOutputParser()

    # Create the chain
    chain = prompt | llm | output_parser

    # Process input and display output
    if input_text:
        try:
            response = chain.invoke({"question": input_text})
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
