# import os
# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.schema import SystemMessage, HumanMessage, Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from pypdf import PdfReader
# from langchain.prompts import (
#     ChatPromptTemplate,
#     HumanMessagePromptTemplate,
#     SystemMessagePromptTemplate,
# )
# from langchain_community.vectorstores import FAISS
#
# load_dotenv(override=True)
# groq_api_key = os.getenv("GROQ_API_KEY")
# if not groq_api_key:
#     raise RuntimeError("Missing GROQ_API_KEY in environment")
#
#
# # 1. Load resumes
# def load_resumes(file_paths):
#     resume_texts = {}
#     for path in file_paths:
#         reader = PdfReader(path)
#         content = ""
#         for page in reader.pages:
#             content += page.extract_text()
#         resume_texts[os.path.basename(path)] = content
#     return resume_texts
#
#
# # def create_chunks(text):
# #     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# #     return [Document(page_content=chunk) for chunk in splitter.split_text(text)]
#
#
# # # 3. Embed and save separately
# # def create_vector_db_per_resume(resume_dict):
# #     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# #     return embeddings
#
# def get_conversation_chain(resume_texts) -> ConversationalRetrievalChain:
#     llm = ChatGroq(
#         api_key=groq_api_key,
#         model="llama3-70b-8192",
#         temperature=0.2,
#         max_tokens=1024
#     )
#
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#     role = input('Enter the hiring role')
#
#     system_prompt = SystemMessagePromptTemplate.from_template(
#         """
#         You are a hiring assistant for a company, you are hiring for the role of a \"\"\"{role}\"\"\"
#         Review the reume, look at previous experience if any and technical skills if appropriate for the role.
#         You should return result as percentage, as to how appropriate the person is for the role.
#         """
#     )
#
#     prompt_template = ChatPromptTemplate.from_messages([
#         system_prompt
#     ])
#
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vector_store.as_retriever(),
#         memory=memory,
#         combine_docs_chain_kwargs={"prompt": prompt_template},
#     )
#     return conversation_chain
#
#
# def main():
#     path = input("Upload your Resume: ")
#     file_path = path.strip()
#     resume_contents = load_resumes(file_path)
#     get_conversation_chain()
#
#
#
#
#
# if __name__ == "__main__":
#     main()


import streamlit as st
from langchain_groq import ChatGroq
from pypdf import PdfReader
import os

# Load PDF and extract text
def load_resume_text(file):
    reader = PdfReader(file)
    content = ""
    for page in reader.pages:
        content += page.extract_text()
    return content

# Streamlit UI
def main():
    st.title("Resume Evaluator")
    uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
    role = st.text_input("Enter the hiring role:")
    groq_api_key = st.text_input("Enter your GROQ_API_KEY:", type='password')

    if st.button("Evaluate") and uploaded_file and role and groq_api_key:
        # Load and process resume
        resume_text = load_resume_text(uploaded_file)

        # Initialize LLM
        llm = ChatGroq(
            api_key=groq_api_key,
            model="llama3-70b-8192",
            temperature=0.2,
            max_tokens=1024
        )

        # Prompt template
        prompt = f"""
        You are a hiring assistant for a company, hiring for the role of "{role}".
        Review the following resume:
        ---
        {resume_text}
        ---
        Evaluate how appropriate the person is for the role based on previous experience and technical skills.
        Return:
        - A score as a percentage (0-100) for candidate fit.
        - Candidate level: Beginner, Intermediate, or Advanced.
        """

        # Get result
        result = llm.invoke(prompt)
        st.subheader("📊 Evaluation Result")
        st.write(result)

if __name__ == "__main__":
    main()

