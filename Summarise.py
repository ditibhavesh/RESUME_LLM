from langchain_groq import ChatGroq
from pypdf import PdfReader
import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import os

def load_resume_text(file_path):
    reader = PdfReader(file_path)
    content = ""
    for page in reader.pages:
        content += page.extract_text()
    return content


import re

def judge_candidates_fit(groq_api_key, role, resume_text):
    llm = ChatGroq(
        api_key=groq_api_key,
        model="llama3-70b-8192",
        temperature=0.2,
        max_tokens=1024
    )

    template = f"""
    You are a hiring assistant for a company, hiring for the role of "{role}".
    Review the following resume:
    ---
    {resume_text}
    ---
    Evaluate how appropriate the person is for the role.
    Return only this format:
    "Score: <number>%\nLevel: <Beginner/Intermediate/Advanced/Not a good fit for the role>"
    """

    result = llm.invoke(template).content

    # Extract score and level
    match = re.search(r'Score:\s*(\d+)%\s*Level:\s*(.*)', result)
    if match:
        score = int(match.group(1))
        level = match.group(2).strip()
    else:
        score = 0
        level = "Not a good fit for the role"

    return {
        "raw": result,
        "score": score,
        "level": level
    }



def suggest_interview_question(api_key, role, evaluation_summary, count):
    """Generate a follow-up interview question based on candidate evaluation."""
    llm = ChatGroq(api_key=api_key, model="llama3-70b-8192", temperature=0.2, max_tokens=1024)

    prompt = f"""
    You are a hiring assistant for a company hiring for the role of "{role}".
    Based on the evaluation below:
    ---
    {evaluation_summary}
    ---
    If the score is below 40 and candidate is not a fit for the role - just return 'Sorry, we would not like to proceed with your candidature, please try again for a role that appeals to you'
    if score is above 40 , based on the candidate's level - Ask interview questions, structured as:
    If level is beginner ask 5 questions easy questions, if intermediate then ask 4 medium difficulty questions and if advanced then all 3 difficult technical questions
    Just return the next question, with no explanations.
    """
    return llm.invoke(prompt).content

    # - 70% technical
    # - 20% problem-solving/brain-stimulating
    # - 10% soft skills

def assess_answers_and_skills(groq_api_key, role, question, answer):
    llm = ChatGroq(
        api_key=groq_api_key,
        model="llama3-70b-8192",
        temperature=0.2,
        max_tokens=1024
    )

    prompt_text = f"""
    You are a hiring assistant for a company, hiring for the role of "{role}".
    Here is the interview question asked:
    ---
    {question}
    ---
    And here is the candidate's answer:
    ---
    {answer}
    ---
    Now assess the candidate's answering skill.
    Re-evaluate their level (Beginner, Intermediate, or Advanced) and provide a new fit score out of 100.
    While recalculation take into account the previous fit score and then provide the new score.
    """

    return llm.invoke(prompt_text).content


def main():
    resume_path = st.file_uploader("Upload your Resume (PDF path): ")
    role = st.text_input("Enter the hiring role: ")
    groq_api_key = st.text_input("GROQ_API_KEY", type='password')

    # Setup session state
    if "evaluation_result" not in st.session_state:
        st.session_state.evaluation_result = None
    if "questions" not in st.session_state:
        st.session_state.questions = []
    if "answers" not in st.session_state:
        st.session_state.answers = []
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0

    if st.button("Evaluate Resume") and resume_path and role and groq_api_key:
        resume_text = load_resume_text(resume_path)
        evaluation = judge_candidates_fit(groq_api_key, role, resume_text)
        st.session_state.evaluation_result = evaluation
        st.session_state.questions = []
        st.session_state.answers = []
        st.session_state.current_index = 0

        st.write("### 📊 Evaluation Result")
        st.write(evaluation["raw"])

        if evaluation["score"] < 40:
            st.warning("Candidate score is below 40. Not a good fit for the role.")
            st.stop()

        # First question after evaluation
        first_question = suggest_interview_question(
            groq_api_key, role, evaluation["raw"], 1
        )
        st.session_state.questions.append(first_question)
        st.session_state.answers.append("")

    # Continue if evaluation already happened
    if st.session_state.evaluation_result and st.session_state.questions:
        idx = st.session_state.current_index
        question = st.session_state.questions[idx]
        st.write(f"### 💬 Question {idx + 1}")
        st.write(question)

        # Text input for answer
        answer = st.text_area(f"✍️ Candidate's Answer to Question {idx + 1}",
                              value=st.session_state.answers[idx], key=f"answer_{idx}")

        # On submission
        if st.button("Submit Answer"):
            st.session_state.answers[idx] = answer
            if answer.strip():
                result = assess_answers_and_skills(
                    groq_api_key, role, question, answer
                )
                st.write(f"🧠 Updated Evaluation for Answer {idx + 1}")
                st.write(result)

                # Next question
                next_question = suggest_interview_question(
                    groq_api_key, role, st.session_state.evaluation_result["raw"], idx + 2
                )
                if "Sorry" in next_question:
                    st.warning(next_question)
                else:
                    st.session_state.questions.append(next_question)
                    st.session_state.answers.append("")
                    st.session_state.current_index += 1



if __name__ == "__main__":
    main()


