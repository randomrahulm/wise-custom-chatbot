from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.agents import create_csv_agent
from tempfile import NamedTemporaryFile
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent





def main():
    load_dotenv(os.getenv("OPEN_AI_KEY"))
    st.set_page_config(page_title="wise")
    st.header("ASK your QUESTIONS")
    pdf =st.sidebar.file_uploader("UPLOAD YOUR PDF",type='pdf')
    file=st.sidebar.file_uploader("UPLOAD YOUR CSV",type='csv')
    
    
    if file is not None:
        df=pd.read_csv(file)
        user_questions=st.text_input("Ask your csv")
        agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)
        if user_questions is not None:
            response=agent.run(user_questions)
            st.write(response)
        
            



    if pdf is not None:
        pdf_reader=PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        
    #print("Hello World")
        text_splitter=CharacterTextSplitter(
            separator='\n',
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len 
        )
        chunks=text_splitter.split_text(text)
        
        embeddings=OpenAIEmbeddings()
        knowleddge_base=FAISS.from_texts(chunks,embeddings)

        user_questions=st.text_input("Ask Questins to your Data")

        if user_questions:
            docs=knowleddge_base.similarity_search(user_questions)

            llms=OpenAI()
            chain=load_qa_chain(llms,chain_type="stuff")
            response=chain.run(input_documents=docs,question=user_questions)

            st.write(response)
            

if __name__=="__main__":
    main()