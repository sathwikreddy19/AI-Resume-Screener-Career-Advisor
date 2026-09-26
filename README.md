# AI Resume Screener and Career Advisor

AI Resume Screener and Career Advisor is a Flask-based web application that helps job seekers check how well their resume matches a job description and get guidance for their career-related questions.

The project has two main modules:

1. **AI Resume Screener**
2. **RAG-Based Career Advisor**

---

## Project Overview

Finding the right job requires more than just having a resume. Candidates also need to understand whether their skills match the job requirements and what they should improve.

This project was developed to make that process easier.

The **Resume Screener** accepts a resume in PDF format along with a job description. It extracts skills from both, compares them, calculates a match score, and provides useful suggestions.

The **Career Advisor** allows users to ask career-related questions. It uses a RAG-based approach to retrieve relevant information from the career knowledge base before generating the response.

---

## Key Features

### AI Resume Screener

- Upload resume in PDF format
- Enter a job description
- Extract text from the resume
- Extract skills from the resume and job description
- Identify matched skills
- Identify missing skills
- Calculate resume-job match score
- Display candidate strengths
- Provide skill-gap recommendations
- Suggest resume improvements

### RAG-Based Career Advisor

- Accept career-related questions from the user
- Convert the question into an embedding
- Retrieve relevant information from the career knowledge base
- Use FAISS for similarity search
- Retrieve the top relevant career information
- Generate career guidance using the retrieved context
- Provide learning suggestions and career roadmaps

---

## Technologies Used

### Backend
- Python
- Flask

### Frontend
- HTML
- CSS
- Jinja2

### AI / LLM
- Gemini 3.5 Flash Lite
- Gemini Embedding Model (`gemini-embedding-001`)

### RAG and Retrieval
- LangChain
- FAISS
- Vector Embeddings
- Prompt Templates

### Resume Processing
- PyMuPDF

### Data Validation
- Pydantic

### Other Tools
- Git
- GitHub
- VS Code

---

## Project Architecture

The application contains two separate processing flows.

### Resume Screener Flow

```text
Resume PDF + Job Description
            ↓
        Flask App
            ↓
Resume Text Extraction using PyMuPDF
            ↓
Resume Skills + JD Skills Extraction
            ↓
      Gemini 3.5 Flash Lite
            ↓
       Pydantic Validation
            ↓
     Python Skill Matching
            ↓
Matched Skills + Missing Skills
            ↓
     Match Score Calculation
            ↓
      Gemini Suggestions
            ↓
Strengths + Skill Gap Recommendations
+ Resume Improvement Suggestions
            ↓
        Final Result
```

### RAG Career Advisor Flow

```text
Career Knowledge Base
        ↓
     Chunking
        ↓
gemini-embedding-001
        ↓
Knowledge Embeddings
        ↓
      FAISS
        ↑
        │
User Career Question
        ↓
gemini-embedding-001
        ↓
  Query Embedding
        ↓
   FAISS Search
        ↓
Top Relevant Career Chunks
        ↓
Retrieved Context + User Question
        ↓
    Prompt Template
        ↓
Gemini 3.5 Flash Lite
        ↓
Career Guidance / Roadmap
```

---

## How the Resume Screener Works

The user uploads a resume and enters a job description.

The application extracts text from the uploaded PDF using **PyMuPDF**.

The resume text and job description are processed to identify their respective skills.

The extracted skills are validated and then compared using Python matching logic.

The application identifies:

- Resume Skills
- Job Description Skills
- Matched Skills
- Missing Skills

The match score is calculated based on the number of job-description skills matched by the resume.

The results are then used to generate strengths, skill-gap recommendations, and resume improvement suggestions.

---

## How the Career Advisor Works

The Career Advisor follows a **Retrieval-Augmented Generation (RAG)** approach.

Career information is stored in a knowledge base and divided into smaller chunks.

The chunks are converted into embeddings using `gemini-embedding-001`.

The embeddings are indexed using **FAISS**.

When a user asks a career-related question:

1. The question is converted into an embedding.
2. FAISS compares the query embedding with the stored career knowledge embeddings.
3. The most relevant chunks are retrieved.
4. The retrieved context is combined with the user's question.
5. Gemini generates the final career guidance using this context.

This allows the Career Advisor to provide answers based on the available career knowledge instead of relying only on the LLM.

---

## Match Score

The Resume Screener calculates the match percentage using:

```text
Match Score =
(Number of Matched JD Skills / Total JD Skills) × 100
```

For example:

```text
Matched Skills = 6
Total JD Skills = 10

Match Score = (6 / 10) × 100
            = 60%
```

---

## Project Structure

```text
AI-Resume-Screener-Career-Advisor/
│
├── app.py
├── requirements.txt
├── .env
├── career_knowledge.txt
├── career_embeddings_gemini.npy
│
├── templates/
│   ├── index.html
│   ├── result.html
│   └── career_results.html
│
└── static/
    └── style.css
```

> **Note:** The `.env` file should not be uploaded to GitHub because it contains API credentials.

---

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
```

Move into the project directory:

```bash
cd AI-Resume-Screener-Career-Advisor
```

Create a virtual environment:

```bash
python -m venv prj
```

Activate it on Windows:

```bash
prj\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Create a `.env` file and add the required API credentials.

Run the application:

```bash
python app.py
```

Open the local application in your browser:

```text
http://127.0.0.1:5000
```

---

## Application Output

The Resume Screener displays:

- Resume Skills
- Job Description Skills
- Matched Skills
- Missing Skills
- Match Score
- Strengths
- Skill-Gap Recommendations
- Resume Improvement Suggestions

The Career Advisor displays career guidance based on the user's question and retrieved career knowledge.

---

## Challenges Faced

During development, some of the main challenges included:

- Getting structured responses from the LLM
- Handling invalid or incomplete model responses
- Extracting skills correctly from different resumes and job descriptions
- Improving skill matching for alternative skill names
- Working with embeddings
- Implementing FAISS similarity search
- Avoiding repeated embedding generation
- Handling API-related errors during development and deployment

These issues were handled through validation, exception handling, prompt improvements, embedding caching, and changes to the processing logic.

---

## Future Improvements

- Support DOCX resumes
- Improve skill normalization and synonym matching
- Add more career roles to the knowledge base
- Expand the RAG knowledge source
- Add user accounts and analysis history
- Add downloadable resume analysis reports
- Improve resume scoring using additional factors such as experience and projects
- Add more detailed career learning paths

---

## Author

**Sathwik Reddy**

B.Tech — Computer Science and Engineering

GitHub: Add your GitHub profile link here  
LinkedIn: Add your LinkedIn profile link here

---

## Project Status

The core features of the **AI Resume Screener and RAG-Based Career Advisor** are implemented and the application has been deployed successfully.
