from langchain_groq import ChatGroq
from pypdf import PdfReader
import streamlit as st
import re


def load_resume_text(file_path):
    reader = PdfReader(file_path)
    content = ""
    for page in reader.pages:
        content += page.extract_text()
    return content


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


def suggest_interview_question(api_key, role, evaluation_summary, count, asked_questions):
    llm = ChatGroq(api_key=api_key, model="llama3-70b-8192", temperature=0.7, max_tokens=1024)

    asked_questions_str = "\n".join(f"- {q}" for q in asked_questions)

    prompt = f"""
    You are a hiring assistant for a company hiring for the role of "{role}".
    Based on the candidate's evaluation:
    ---
    {evaluation_summary}
    ---
    The candidate has already been asked these questions:
    {asked_questions_str}

    If score is above 40, generate the next interview question:
    - Ensure this question is NOT one of the previous questions.
    - If level is Beginner, ask an easy question.
    - If Intermediate, ask a medium difficulty question.
    - If Advanced, ask a difficult technical question.
    - Questions should include 70% technical, 20% problem-solving, 10% soft skills.

    Return only the question text, no explanations.
    """
    return llm.invoke(prompt).content


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
    While recalculating, take into account the previous fit score and then provide the new score.
    """

    return llm.invoke(prompt_text).content


def final_evaluation_summary(groq_api_key, role, questions, answers, initial_evaluation):
    llm = ChatGroq(
        api_key=groq_api_key,
        model="llama3-70b-8192",
        temperature=0.3,
        max_tokens=1024
    )
    qa_pairs = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in zip(questions, answers)])

    prompt = f"""
    You are a hiring assistant for a company, hiring for the role of "{role}".
    Here is the initial evaluation summary:
    ---
    {initial_evaluation}
    ---
    Here are all the interview questions and candidate's answers:
    ---
    {qa_pairs}
    ---
    Please provide a final overall evaluation of the candidate's fit for the role.
    Return the result in the format:
    "Final Score: <number>%\nFinal Level: <Beginner/Intermediate/Advanced/Not a good fit>"
    Include a brief reason summary.
    """

    return llm.invoke(prompt).content


def main():
    resume_path = st.file_uploader("Upload your Resume (PDF): ")
    role = st.text_input("Enter the hiring role: ")
    groq_api_key = st.text_input("GROQ_API_KEY", type='password')

    if "evaluation_result" not in st.session_state:
        st.session_state.evaluation_result = None
    if "questions" not in st.session_state:
        st.session_state.questions = []
    if "answers" not in st.session_state:
        st.session_state.answers = []
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "finished" not in st.session_state:
        st.session_state.finished = False
    if "final_summary" not in st.session_state:
        st.session_state.final_summary = ""

    if st.button("Evaluate Resume") and resume_path and role and groq_api_key:
        resume_text = load_resume_text(resume_path)
        evaluation = judge_candidates_fit(groq_api_key, role, resume_text)
        st.session_state.evaluation_result = evaluation
        st.session_state.questions = []
        st.session_state.answers = []
        st.session_state.current_index = 0
        st.session_state.finished = False
        st.session_state.final_summary = ""

        st.write("### 📊 Evaluation Result")
        st.write(evaluation)

        if evaluation["score"] < 40:
            st.warning("Candidate score is below 40. Not a good fit for the role.")
            st.stop()

        # First question
        first_question = suggest_interview_question(
            groq_api_key, role, evaluation["raw"], 1, []
        )
        # if "Sorry" in first_question:
        #     st.warning(first_question)
        #     st.stop()

        st.session_state.questions.append(first_question)
        st.session_state.answers.append("")

    if st.session_state.evaluation_result and st.session_state.questions and not st.session_state.finished:
        idx = st.session_state.current_index
        question = st.session_state.questions[idx]

        st.write(f"💬 Question {idx + 1}")
        st.write(question)

        answer = st.text_area(f"✍️ Candidate's Answer to Question {idx + 1}",
                              value=st.session_state.answers[idx], key=f"answer_{idx}")

        if st.button("Submit Answer"):
            if not answer.strip():
                st.warning("Please provide an answer before submitting.")
                st.stop()
            st.session_state.answers[idx] = answer

            # Assess answer & update evaluation
            # result = assess_answers_and_skills(
            #     groq_api_key, role, question, answer
            # )
            # st.write(f"🧠 Updated Evaluation for Answer {idx + 1}")
            # st.write(result)

            # Max questions per level
            level = st.session_state.evaluation_result["level"].lower()
            max_questions_map = {"beginner": 5, "intermediate": 4, "advanced": 3}
            max_questions = max_questions_map.get(level, 3)

            if idx + 1 >= max_questions:
                st.session_state.finished = True
                st.session_state.final_summary = final_evaluation_summary(
                    groq_api_key, role, st.session_state.questions, st.session_state.answers,
                    st.session_state.evaluation_result["raw"]
                )
                st.success("✅ All questions answered. Final evaluation:")
                st.write(st.session_state.final_summary)

            else:
                next_question = suggest_interview_question(
                    groq_api_key, role, st.session_state.evaluation_result["raw"], idx + 2, st.session_state.questions
                )
                if "Sorry" in next_question:
                    st.warning(next_question)
                    st.session_state.finished = True
                    st.session_state.final_summary = final_evaluation_summary(
                        groq_api_key, role, st.session_state.questions, st.session_state.answers,
                        st.session_state.evaluation_result["raw"]
                    )
                    st.success("✅ Final evaluation:")
                    st.write(st.session_state.final_summary)
                else:
                    st.session_state.questions.append(next_question)
                    st.session_state.answers.append("")
                    st.session_state.current_index += 1

    elif st.session_state.finished:
        st.success("✅ Interview complete. Final evaluation:")
        st.write(st.session_state.final_summary)


if __name__ == "__main__":
    main()
