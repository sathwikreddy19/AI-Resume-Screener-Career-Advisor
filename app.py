# ============================================================
# IMPORTS
# ============================================================

from flask import Flask, render_template, request
import pymupdf
import os
import faiss
import markdown
import numpy as np

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise RuntimeError(
        "GEMINI_API_KEY was not found in .env file."
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=gemini_api_key
)


# ============================================================
# GEMINI MODELS
# ============================================================

LLM_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)

EMBEDDING_MODEL = "gemini-embedding-001"


# ============================================================
# PYDANTIC STRUCTURED OUTPUT MODELS
# ============================================================

class SkillsOutput(BaseModel):

    skills: list[str]


class SkillMatchOutput(BaseModel):

    matched_skills: list[str]

    missing_skills: list[str]


class SuggestionsOutput(BaseModel):

    strengths: list[str] = Field(
        description=(
            "Evidence-based strengths of the candidate "
            "supported by the resume."
        )
    )

    skill_gap_recommendations: list[str] = Field(
        min_length=1,
        description=(
            "Practical recommendations addressing the "
            "candidate's missing job-description skills. "
            "This list must not be empty."
        )
    )

    improvement_suggestions: list[str] = Field(
        description=(
            "Practical suggestions for improving the resume "
            "for the supplied job description."
        )
    )


# ============================================================
# GEMINI EMBEDDING FUNCTION
# ============================================================

def get_embedding(
    text,
    task_type="RETRIEVAL_DOCUMENT"
):

    result = client.models.embed_content(

        model=EMBEDDING_MODEL,

        contents=text,

        config=types.EmbedContentConfig(
            task_type=task_type
        )
    )

    embedding = np.array(
        result.embeddings[0].values,
        dtype="float32"
    )

    return embedding


# ============================================================
# LOAD CAREER KNOWLEDGE BASE
# ============================================================

with open(
    "career_knowledge.txt",
    "r",
    encoding="utf-8"
) as file:

    career_knowledge = file.read()


# ============================================================
# LANGCHAIN TEXT SPLITTING
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(

    chunk_size=600,

    chunk_overlap=100
)

chunks = text_splitter.split_text(
    career_knowledge
)


# ============================================================
# EMBEDDING CACHE
# ============================================================

EMBEDDING_CACHE_FILE = (
    "career_embeddings_gemini.npy"
)

chunk_embeddings = None


# ============================================================
# LOAD EMBEDDINGS FROM CACHE
# ============================================================

if os.path.exists(
    EMBEDDING_CACHE_FILE
):

    try:

        cached_embeddings = np.load(
            EMBEDDING_CACHE_FILE
        )

        if len(cached_embeddings) == len(chunks):

            chunk_embeddings = (
                cached_embeddings.astype(
                    "float32"
                )
            )

            print(
                "Loaded career embeddings from cache."
            )

        else:

            print(
                "Knowledge base changed."
            )

            print(
                "Rebuilding embedding cache..."
            )

    except Exception as error:

        print(
            "Could not load embedding cache:",
            error
        )


# ============================================================
# CREATE EMBEDDINGS IF CACHE DOES NOT EXIST
# ============================================================

if chunk_embeddings is None:

    print(
        "Creating Gemini embeddings "
        "for career knowledge..."
    )

    embedding_list = []

    for number, chunk in enumerate(
        chunks,
        start=1
    ):

        vector = get_embedding(

            chunk,

            task_type="RETRIEVAL_DOCUMENT"
        )

        embedding_list.append(
            vector
        )

        print(
            f"Embedded chunk "
            f"{number}/{len(chunks)}"
        )

    chunk_embeddings = np.array(
        embedding_list,
        dtype="float32"
    )

    np.save(
        EMBEDDING_CACHE_FILE,
        chunk_embeddings
    )

    print(
        "Saved career embeddings to cache."
    )


# ============================================================
# CREATE FAISS VECTOR DATABASE
# ============================================================

dimension = chunk_embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(
    chunk_embeddings
)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# RESUME ANALYSIS ROUTE
# ============================================================

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    # ========================================================
    # GET RESUME + JOB DESCRIPTION
    # ========================================================

    resume = request.files.get(
        "resume"
    )

    job_description = request.form.get(
        "job_description"
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    if not resume or not resume.filename:

        return render_template(
            "index.html",
            error="Please upload a resume PDF."
        )


    if (
        not job_description
        or not job_description.strip()
    ):

        return render_template(
            "index.html",
            error="Please enter the job description."
        )


    if not resume.filename.lower().endswith(
        ".pdf"
    ):

        return render_template(
            "index.html",
            error="Please upload only a PDF file."
        )


    # ========================================================
    # PDF TEXT EXTRACTION
    # ========================================================

    try:

        pdf_bytes = resume.read()

        document = pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        resume_text = ""

        for page in document:

            resume_text += page.get_text()

        document.close()

    except Exception as error:

        print(
            "PDF reading error:",
            error
        )

        return render_template(
            "index.html",
            error=(
                "Unable to read the PDF. "
                "Please upload a valid PDF."
            )
        )


    # ========================================================
    # CHECK EMPTY PDF
    # ========================================================

    if not resume_text.strip():

        return render_template(
            "index.html",
            error=(
                "No readable text was found in the PDF. "
                "Please upload a text-based resume PDF."
            )
        )


    # ========================================================
    # STEP 1
    # DYNAMIC RESUME SKILL EXTRACTION
    # ========================================================

    resume_skills_prompt = f"""
You are an AI Resume Skill Extractor.

Read the COMPLETE resume carefully.

Your task is to dynamically identify the candidate's
technical and professional skills.

IMPORTANT RULES:

1. Extract skills only when explicitly mentioned
   or clearly demonstrated by the resume.

2. Do NOT use a predefined or hardcoded skill list.

3. Discover skills dynamically from the supplied resume.

4. Do NOT invent skills.

5. Do NOT assume a candidate knows a technology simply
   because it is related to another technology.

6. Consider skills appearing in:
   - Skills
   - Projects
   - Internship
   - Experience
   - Certifications
   - Technical descriptions

7. Include relevant:
   - Programming languages
   - Frameworks
   - Libraries
   - Databases
   - Developer tools
   - Cloud technologies
   - AI/ML technologies
   - Data technologies
   - Web technologies
   - Technical concepts

8. Remove unnecessary duplicates.

9. Return structured output only.


RESUME:

{resume_text}
"""


    try:

        response = client.models.generate_content(

            model=LLM_MODEL,

            contents=resume_skills_prompt,

            config=types.GenerateContentConfig(

                response_mime_type="application/json",

                response_schema=SkillsOutput
            )
        )

        resume_skills_data = (
            SkillsOutput.model_validate_json(
                response.text
            )
        )

        resume_skills = (
            resume_skills_data.skills
        )

        print(
            "Resume Skills:",
            resume_skills
        )

    except Exception as error:

        print(
            "Resume skill extraction error:",
            error
        )

        return render_template(
            "index.html",
            error=(
                "AI could not analyze the resume. "
                "Please try again."
            )
        )


    # ========================================================
    # STEP 2
    # DYNAMIC JD SKILL EXTRACTION
    # ========================================================

    jd_skills_prompt = f"""
You are an AI Job Description Skill Extractor.

Read the COMPLETE job description carefully.

Dynamically identify the important technical skills
and technologies requested by the employer.

IMPORTANT RULES:

1. Do NOT use a predefined or hardcoded skill list.

2. Discover skills dynamically from the supplied JD.

3. Extract skills that are required, preferred,
   good-to-have, or clearly requested.

4. Do NOT invent skills.

5. Include relevant:
   - Programming languages
   - Frameworks
   - Libraries
   - Databases
   - Operating systems
   - Developer tools
   - Cloud technologies
   - AI/ML technologies
   - Data Engineering technologies
   - APIs
   - Technical concepts

6. If the JD gives alternatives such as:

   Flask or FastAPI

   represent them meaningfully as:

   Flask/FastAPI

7. Remove unnecessary duplicates.

8. Return structured output only.


JOB DESCRIPTION:

{job_description}
"""


    try:

        response = client.models.generate_content(

            model=LLM_MODEL,

            contents=jd_skills_prompt,

            config=types.GenerateContentConfig(

                response_mime_type="application/json",

                response_schema=SkillsOutput
            )
        )

        jd_skills_data = (
            SkillsOutput.model_validate_json(
                response.text
            )
        )

        jd_skills = (
            jd_skills_data.skills
        )

        print(
            "JD Skills:",
            jd_skills
        )

    except Exception as error:

        print(
            "JD skill extraction error:",
            error
        )

        return render_template(
            "index.html",
            error=(
                "AI could not analyze the "
                "job description. Please try again."
            )
        )


    # ========================================================
    # STEP 3
    # LLM-BASED INTELLIGENT SKILL MATCHING
    # ========================================================

    matching_prompt = f"""
You are an AI Resume-to-Job Skill Matching Engine.

Compare the candidate's resume skills with
the job description skills.

RESUME SKILLS:

{resume_skills}


JOB DESCRIPTION SKILLS:

{jd_skills}


IMPORTANT MATCHING RULES:

1. Evaluate EVERY job-description skill.

2. Decide whether the candidate demonstrates
   that skill or a genuinely equivalent skill.

3. Matching should understand meaningful
   equivalents and alternatives.

Example:

JD:
Flask/FastAPI

Resume:
Flask

Result:
Matched


4. Minor naming differences may match when they
   clearly refer to the same technology.

Example:

JD:
Postgres

Resume:
PostgreSQL

Result:
Matched


5. Abbreviations may match their full forms when
   they clearly represent the same skill.

6. Related does NOT automatically mean equivalent.

VERY IMPORTANT:

Resume:
GitHub

JD:
Linux

Result:
Missing


Resume:
Data Structures & Algorithms

JD:
Data Engineering Fundamentals

Result:
Missing


Resume:
Python

JD:
Java

Result:
Missing


7. Do NOT invent candidate skills.

8. Every JD skill must belong to exactly one group:

   matched_skills

   OR

   missing_skills

9. Do not silently remove a JD skill.

10. Preserve JD skill names in the output.

11. Return structured output only.
"""


    try:

        response = client.models.generate_content(

            model=LLM_MODEL,

            contents=matching_prompt,

            config=types.GenerateContentConfig(

                response_mime_type="application/json",

                response_schema=SkillMatchOutput
            )
        )

        matching_data = (
            SkillMatchOutput.model_validate_json(
                response.text
            )
        )

        matched_skills = (
            matching_data.matched_skills
        )

        missing_skills = (
            matching_data.missing_skills
        )

        print(
            "Matched Skills:",
            matched_skills
        )

        print(
            "Missing Skills:",
            missing_skills
        )

    except Exception as error:

        print(
            "LLM skill matching error:",
            error
        )

        return render_template(
            "index.html",
            error=(
                "AI could not perform "
                "skill matching. Please try again."
            )
        )


    # ========================================================
    # STEP 4
    # MATCH SCORE
    # ========================================================

    total_jd_skills = len(
        jd_skills
    )

    total_matched_skills = len(
        matched_skills
    )

    if total_jd_skills > 0:

        match_score = (
            total_matched_skills
            / total_jd_skills
        ) * 100

    else:

        match_score = 0

    match_score = round(
        match_score,
        2
    )

    print(
        "Match Score:",
        match_score
    )


    # ========================================================
    # STEP 5
    # DYNAMIC STRENGTHS + SKILL GAPS + IMPROVEMENTS
    # ========================================================

    suggestion_prompt = f"""
You are an AI Resume Career Advisor.

Analyze the candidate's COMPLETE resume against
the COMPLETE job description.

Use the supplied skill-matching results.

RESUME:

{resume_text}


JOB DESCRIPTION:

{job_description}


RESUME SKILLS:

{resume_skills}


JOB DESCRIPTION SKILLS:

{jd_skills}


MATCHED SKILLS:

{matched_skills}


MISSING SKILLS:

{missing_skills}


MATCH SCORE:

{match_score}%


NUMBER OF MISSING SKILLS:

{len(missing_skills)}


Generate THREE separate output sections:

1. strengths

Generate evidence-based candidate strengths using
the actual resume and matched skills.


2. skill_gap_recommendations

The missing skills are:

{missing_skills}

IMPORTANT:

There are {len(missing_skills)} missing skills.

Generate ONE specific and practical recommendation
for EACH missing skill.

Every recommendation should explain what the
candidate should learn, practice, or build to
address that particular missing skill.

If missing_skills contains one or more skills,
skill_gap_recommendations MUST NOT be empty.

Do NOT move skill-gap recommendations into
improvement_suggestions.

Do NOT claim that the candidate already knows
a missing skill.


Example:

Missing Skill:
Linux

Recommendation:
Practice Linux command-line fundamentals including
file management, permissions, processes, and basic
shell commands.


3. improvement_suggestions

Generate practical suggestions for improving
the candidate's resume for this specific job
description.


GENERAL RULES:

1. Use ONLY information supported by the resume,
   job description, and matching results.

2. Do NOT invent:
   - work experience
   - projects
   - skills
   - certifications
   - percentages
   - achievements
   - business impact
   - metrics

3. Strengths must be supported by the resume.

4. Skill-gap recommendations must address the
   supplied missing skills.

5. If there are 3 missing skills, generate
   3 skill-gap recommendations.

6. If there are 4 missing skills, generate
   4 skill-gap recommendations.

7. Each missing skill should have its own
   recommendation.

8. Resume improvement suggestions should focus
   on better alignment with this specific JD.

9. Never tell the candidate to falsely add a skill
   they do not possess.

10. If recommending a missing skill, tell the
    candidate to learn, practice, or build
    experience with it first.

11. Keep all recommendations concise and practical.

12. Return structured output only.
"""


    try:

        response = client.models.generate_content(

            model=LLM_MODEL,

            contents=suggestion_prompt,

            config=types.GenerateContentConfig(

                response_mime_type="application/json",

                response_schema=SuggestionsOutput
            )
        )

        suggestion_data = (
            SuggestionsOutput.model_validate_json(
                response.text
            )
        )

        strengths = (
            suggestion_data.strengths
        )

        skill_gap_recommendations = (
            suggestion_data.skill_gap_recommendations
        )

        improvement_suggestions = (
            suggestion_data.improvement_suggestions
        )


        print(
            "Strengths:",
            strengths
        )

        print(
            "Skill Gap Recommendations:",
            skill_gap_recommendations
        )

        print(
            "Improvement Suggestions:",
            improvement_suggestions
        )


    except Exception as error:

        print(
            "Suggestion generation error:",
            error
        )

        strengths = []

        skill_gap_recommendations = []

        improvement_suggestions = [
            (
                "AI suggestions are temporarily "
                "unavailable. Please try again."
            )
        ]


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    return render_template(

        "results.html",

        match_score=match_score,

        resume_skills=resume_skills,

        jd_skills=jd_skills,

        matched_skills=matched_skills,

        missing_skills=missing_skills,

        strengths=strengths,

        skill_gap_recommendations=(
            skill_gap_recommendations
        ),

        improvement_suggestions=(
            improvement_suggestions
        )
    )


# ============================================================
# CAREER ADVISOR ROUTE
# ============================================================

@app.route(
    "/career-advisor",
    methods=["POST"]
)
def career_advisor():

    # ========================================================
    # GET QUESTION
    # ========================================================

    question = request.form.get(
        "question"
    )

    if (
        not question
        or not question.strip()
    ):

        return render_template(
            "index.html",
            error="Please enter a career question."
        )


    # ========================================================
    # QUESTION EMBEDDING
    # ========================================================

    try:

        question_embedding = get_embedding(

            question,

            task_type="RETRIEVAL_QUERY"
        )

        question_embedding = np.array(

            [question_embedding],

            dtype="float32"
        )

    except Exception as error:

        print(
            "Question embedding error:",
            error
        )

        return render_template(
            "index.html",
            error=(
                "Unable to process your career question. "
                "Please try again."
            )
        )


    # ========================================================
    # FAISS VECTOR SEARCH
    # ========================================================

    distances, indices = index.search(

        question_embedding,

        k=5
    )


    # ========================================================
    # RETRIEVE RELEVANT CHUNKS
    # ========================================================

    relevant_chunks = []

    for i in indices[0]:

        if 0 <= i < len(chunks):

            relevant_chunks.append(
                chunks[i]
            )


    # ========================================================
    # CREATE RAG CONTEXT
    # ========================================================

    context = "\n\n".join(
        relevant_chunks
    )


    # ========================================================
    # RAG PROMPT
    # ========================================================

    rag_template = """
    You are an AI Career Advisor.

    Answer the user's career question using ONLY
    the provided career knowledge context.

    RULES:

    1. Use only information available in the supplied context.

    2. Do not invent information that is not present
   in the context.

    3. If the answer is not available in the context,
   clearly say:

   "This information is not available in the
   provided career knowledge base."

    4. Give a clear, beginner-friendly and practical answer.

    5. Organize the answer using meaningful Markdown headings.

    6. When the context contains structured information such as:
   - roadmaps
   - skills
   - technologies
   - learning stages
   - career roles
   - comparisons
   - interview preparation
   - salary information
   - learning plans

   present that information in a Markdown table whenever
   a table makes the information easier to understand.

    7. For career roadmaps, prefer a table with columns such as:

   | Stage | What to Learn | Topics / Technologies |

    8. For skill information, prefer a table such as:

   | Category | Skills |

    9. After the table, provide short practical next steps
   only when those steps are supported by the context.

    10. Preserve important details retrieved from the context.
    Do not unnecessarily reduce a detailed context into
    only a few bullet points.

    11. Do not add technologies, skills, salaries, timelines,
    certifications, or career claims that are not supported
    by the retrieved context.

    12. Use valid Markdown table syntax.

    13. Do not put the Markdown table inside a code block.


    CONTEXT:

    {context}


    USER QUESTION:

    {question}


    ANSWER:
    """


    # ========================================================
    # LANGCHAIN PROMPT TEMPLATE
    # ========================================================

    prompt_template = PromptTemplate(

        input_variables=[
            "context",
            "question"
        ],

        template=rag_template
    )


    # ========================================================
    # INSERT CONTEXT + QUESTION
    # ========================================================

    final_prompt = prompt_template.format(

        context=context,

        question=question
    )


    # ========================================================
    # GEMINI RAG ANSWER
    # ========================================================

    try:

        response = client.models.generate_content(

            model=LLM_MODEL,

            contents=final_prompt,

            config=types.GenerateContentConfig(
                max_output_tokens=2500
            )
        )

        rag_answer = (
            response.text.strip()
        )

    except Exception as error:

        print(
            "Career Advisor LLM error:",
            error
        )

        return render_template(
            "index.html",
            error=(
                "Career Advisor is temporarily "
                "unavailable. Please try again."
            )
        )


    # ========================================================
    # MARKDOWN → HTML
    # ========================================================

    rag_answer_html = markdown.markdown(

        rag_answer,

        extensions=[
            "tables"
        ]
    )


    # ========================================================
    # DISPLAY CAREER ADVISOR RESULT
    # ========================================================

    return render_template(

        "career_results.html",

        question=question,

        rag_answer=rag_answer_html
    )


# ============================================================
# RUN FLASK APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(

        debug=(
            os.getenv(
                "FLASK_DEBUG",
                "false"
            ).lower()
            == "true"
        )
    )