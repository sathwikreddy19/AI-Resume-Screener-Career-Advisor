from flask import Flask, render_template, request
import pymupdf
import os
import json
import faiss
import markdown

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer, util
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate


app = Flask(__name__)


# ==========================================
# ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

client = InferenceClient(
    token=hf_token
)


# ==========================================
# EMBEDDING MODEL
# ==========================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==========================================
# CAREER KNOWLEDGE BASE
# ==========================================

with open(
    "career_knowledge.txt",
    "r",
    encoding="utf-8"
) as file:
    career_knowledge = file.read()


# ==========================================
# TEXT CHUNKING
# ==========================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100
)

chunks = text_splitter.split_text(
    career_knowledge
)


# ==========================================
# CHUNK EMBEDDINGS
# ==========================================

chunk_embeddings = embedding_model.encode(
    chunks
)

chunk_embeddings = chunk_embeddings.astype(
    "float32"
)


# ==========================================
# FAISS VECTOR INDEX
# ==========================================

dimension = chunk_embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(
    chunk_embeddings
)


# ==========================================
# RAG PROMPT TEMPLATE
# ==========================================

rag_template = PromptTemplate.from_template(
"""
You are a helpful AI career advisor.

Answer the user's career question using the information
provided in the context.

Use the context as your primary source of information.

Do not invent technologies, facts, or career requirements
that are not supported by the context.

If the context does not contain enough information to answer
the question, clearly say that the available knowledge does
not contain enough information.

Give the answer in a clear and beginner-friendly format.

Context:
{context}

Question:
{question}

Answer:
"""
)


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==========================================
# RESUME ANALYZER
# ==========================================

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    # --------------------------------------
    # GET FORM DATA
    # --------------------------------------

    resume = request.files.get(
        "resume"
    )

    job_description = request.form.get(
        "job_description"
    )


    # --------------------------------------
    # BASIC VALIDATION
    # --------------------------------------

    if not resume or not resume.filename:

        return render_template(
            "index.html",
            error="Please upload a resume."
        )


    if not resume.filename.lower().endswith(".pdf"):

        return render_template(
            "index.html",
            error="Please upload only a PDF file."
        )


    if not job_description or not job_description.strip():

        return render_template(
            "index.html",
            error="Please enter a job description."
        )


    # ======================================
    # PDF TEXT EXTRACTION
    # ======================================

    try:

        pdf_bytes = resume.read()

        document = pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        resume_text = ""

        for page in document:

            resume_text = (
                resume_text
                +
                page.get_text()
            )

        document.close()

    except Exception:

        return render_template(
            "index.html",
            error="Unable to read the PDF. Please upload a valid resume PDF."
        )


    # --------------------------------------
    # EMPTY PDF TEXT CHECK
    # --------------------------------------

    if not resume_text.strip():

        return render_template(
            "index.html",
            error="Could not extract text from the PDF. Please upload a text-based resume PDF."
        )


    # ======================================
    # RESUME SKILL EXTRACTION PROMPT
    # ======================================

    resume_prompt = f"""
You are an expert technical recruiter.

Extract only the technical skills from the resume below.

Do not include soft skills.

Do not add skills that are not mentioned.

Return only valid JSON in this exact format:

{{
    "skills": [
        "Python",
        "SQL",
        "Flask"
    ]
}}

Resume:

{resume_text}
"""


    # ======================================
    # RESUME LLM API CALL
    # ======================================

    try:

        resume_response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": resume_prompt
                }
            ],
            max_tokens=2000
        )

    except Exception:

        return render_template(
            "index.html",
            error="AI service is temporarily unavailable. Please try again."
        )


    resume_answer = (
        resume_response
        .choices[0]
        .message
        .content
    )


    # ======================================
    # RESUME JSON PARSING
    # ======================================

    try:

        resume_data = json.loads(
            resume_answer
        )

        resume_skills = resume_data[
            "skills"
        ]

    except (json.JSONDecodeError, KeyError, TypeError):

        return render_template(
            "index.html",
            error="Unable to process the resume. Please try again."
        )


    # ======================================
    # JOB DESCRIPTION SKILL EXTRACTION
    # ======================================

    jd_prompt = f"""
You are an expert technical recruiter.

Extract only the technical skills from the
job description below.

Do not include soft skills.

Do not add skills that are not mentioned.

Return only valid JSON in this exact format:

{{
    "skills": [
        "Python",
        "SQL",
        "Flask"
    ]
}}

Job Description:

{job_description}
"""


    # ======================================
    # JD LLM API CALL
    # ======================================

    try:

        jd_response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": jd_prompt
                }
            ],
            max_tokens=2000
        )

    except Exception:

        return render_template(
            "index.html",
            error="AI service is temporarily unavailable. Please try again."
        )


    jd_answer = (
        jd_response
        .choices[0]
        .message
        .content
    )


    # ======================================
    # JD JSON PARSING
    # ======================================

    try:

        jd_data = json.loads(
            jd_answer
        )

        jd_skills = jd_data[
            "skills"
        ]

    except (json.JSONDecodeError, KeyError, TypeError):

        return render_template(
            "index.html",
            error="Unable to process the job description. Please try again."
        )


    # ======================================
    # NORMALIZE RESUME SKILLS
    # ======================================

    normalized_resume_skills = []

    for skill in resume_skills:

        normalized_skill = (
            skill
            .lower()
            .strip()
        )

        normalized_resume_skills.append(
            normalized_skill
        )


    # ======================================
    # NORMALIZE JD SKILLS
    # ======================================

    normalized_jd_skills = []

    for skill in jd_skills:

        normalized_skill = (
            skill
            .lower()
            .strip()
        )

        normalized_jd_skills.append(
            normalized_skill
        )


    # ======================================
    # SEMANTIC SKILL MATCHING
    # ======================================

    matched_skills = []
    missing_skills = []

    threshold = 0.70


    for jd_skill in normalized_jd_skills:

        jd_vector = embedding_model.encode(
            jd_skill
        )

        best_similarity = 0


        for resume_skill in normalized_resume_skills:

            resume_vector = embedding_model.encode(
                resume_skill
            )

            similarity = util.cos_sim(
                jd_vector,
                resume_vector
            ).item()


            if similarity > best_similarity:

                best_similarity = similarity


        if best_similarity >= threshold:

            matched_skills.append(
                jd_skill
            )

        else:

            missing_skills.append(
                jd_skill
            )


    # ======================================
    # MATCH SCORE
    # ======================================

    if len(normalized_jd_skills) > 0:

        match_score = (
            len(matched_skills)
            /
            len(normalized_jd_skills)
        ) * 100

    else:

        match_score = 0


    match_score = round(
        match_score,
        2
    )


    # ======================================
    # RESUME IMPROVEMENT PROMPT
    # ======================================

    suggestion_prompt = f"""
You are an expert resume reviewer and career advisor.

Analyze the resume against the job description and
provide practical suggestions to improve the resume
for this role.

Follow these rules strictly:

1. Do not suggest adding skills, experience, projects,
certifications, or achievements that the candidate
does not actually have.

2. Do not invent numbers, percentages, metrics,
achievements, technologies, responsibilities,
or experience.

3. Use only information explicitly present in the
resume and job description.

4. If a required skill is missing from the resume,
clearly identify it as a skill gap.

Recommend learning or practicing that skill before
adding it to the resume.

5. If measurable impact is not provided in the resume,
suggest quantifying the impact only if the candidate
has verified data.

Do not create example numbers or percentages.

6. Never rewrite a resume bullet using unsupported
information.

7. Clearly separate:

- strengths already present in the resume
- missing technical skills
- safe resume improvement suggestions

Return only valid JSON in this exact format:

{{
    "strengths": [
        "Strength 1",
        "Strength 2"
    ],

    "missing_skills": [
        "Missing skill 1",
        "Missing skill 2"
    ],

    "suggestions": [
        "Suggestion 1",
        "Suggestion 2"
    ]
}}

Resume:

{resume_text}

Job Description:

{job_description}

Matched Skills:

{matched_skills}

Missing Skills:

{missing_skills}
"""


    # ======================================
    # SUGGESTION LLM API CALL
    # ======================================

    try:

        suggestion_response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": suggestion_prompt
                }
            ],
            max_tokens=2000
        )

    except Exception:

        return render_template(
            "index.html",
            error="AI service is temporarily unavailable. Please try again."
        )


    suggestion_answer = (
        suggestion_response
        .choices[0]
        .message
        .content
    )


    # ======================================
    # SUGGESTION JSON PARSING
    # ======================================

    try:

        suggestion_data = json.loads(
            suggestion_answer
        )

        strengths = suggestion_data[
            "strengths"
        ]

        suggestion_missing_skills = suggestion_data[
            "missing_skills"
        ]

        improvement_suggestions = suggestion_data[
            "suggestions"
        ]

    except (json.JSONDecodeError, KeyError, TypeError):

        return render_template(
            "index.html",
            error="Unable to generate resume suggestions. Please try again."
        )


    # ======================================
    # RESULTS PAGE
    # ======================================

    return render_template(
        "results.html",
        resume_skills=resume_skills,
        jd_skills=jd_skills,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        match_score=match_score,
        strengths=strengths,
        suggestion_missing_skills=suggestion_missing_skills,
        improvement_suggestions=improvement_suggestions
    )


# ==========================================
# CAREER ADVISOR
# ==========================================

@app.route(
    "/career-advisor",
    methods=["POST"]
)
def career_advisor():

    question = request.form.get(
        "question"
    )


    # --------------------------------------
    # QUESTION VALIDATION
    # --------------------------------------

    if not question or not question.strip():

        return render_template(
            "index.html",
            error="Please enter a career question."
        )


    # ======================================
    # QUESTION EMBEDDING
    # ======================================

    question_embedding = embedding_model.encode(
        [question]
    )

    question_embedding = question_embedding.astype(
        "float32"
    )


    # ======================================
    # FAISS SEARCH
    # ======================================

    distances, indices = index.search(
        question_embedding,
        k=2
    )


    # ======================================
    # RETRIEVE CHUNKS
    # ======================================

    retrieved_chunks = []

    for i in indices[0]:

        retrieved_chunks.append(
            chunks[i]
        )


    context = "\n\n".join(
        retrieved_chunks
    )


    # ======================================
    # CREATE RAG PROMPT
    # ======================================

    rag_prompt = rag_template.format(
        context=context,
        question=question
    )


    # ======================================
    # RAG LLM API CALL
    # ======================================

    try:

        rag_response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": rag_prompt
                }
            ],
            max_tokens=1800
        )

    except Exception:

        return render_template(
            "index.html",
            error="AI Career Advisor is temporarily unavailable. Please try again."
        )


    rag_answer = (
        rag_response
        .choices[0]
        .message
        .content
    )


    # ======================================
    # MARKDOWN TO HTML
    # ======================================

    rag_answer_html = markdown.markdown(
        rag_answer,
        extensions=["tables"]
    )


    # ======================================
    # CAREER ADVISOR RESULT PAGE
    # ======================================

    return render_template(
        "career_results.html",
        question=question,
        rag_answer=rag_answer_html
    )


# ==========================================
# RUN FLASK APP
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )