i make four file in which split the data by using 4 diffrenent method




import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
import streamlit as st

# Load environment variables
load_dotenv()

# Access the token from environment variables
token = os.getenv('HF_TOKEN')
if token is None:
    raise ValueError("HF_TOKEN not found in environment variables.")
else:
    print("Token retrieved successfully")

# Set up the Hugging Face LLM directly with parameters
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.7,         # Explicitly set parameters
    max_length=128,          # Explicitly set parameters
    huggingface_token=token  # Explicitly set token
)

# Create the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant for making Flutter apps. Please respond to the user's queries."),
        ("user", "Question: {question}")
    ]
)

# Create the chain
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Streamlit app title
st.title('Huggingface Gen AI App')

# Get user input
input_text = st.text_input('Ask a question here')

# Process input and display output
if input_text:
    try:
        # Format the input using the prompt template
        formatted_input = prompt.format(question=input_text)
        # Pass the formatted input to the chain
        response = chain.invoke(formatted_input)
        st.write(response)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
