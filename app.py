from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from datetime import date



app = FastAPI(title="Hindav Profile Q&A API")



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")

PROFILE = {
    "identity": {
        "name": "Hindav Deshmukh",
        "date_of_birth": "2002-11-15",
        "pronouns": "He/Him",
        "location": "Pune, Maharashtra, India",
        "nationality": "Indian",
        "languages": ["English", "Hindi", "Marathi"]
    },
    "education": {
        "degree": "Bachelor of Engineering in Computer Science",
        "graduation_year": 2024,
        "college": "Jawaharlal Darda Institute of Engineering and Technology",
        "university": "Sant Gadge Baba Amravati University",
        "cgpa": 7.23,
        "highlights": [
            "Focused on Machine Learning and Deep Learning coursework",
            "Completed projects in Computer Vision and Time Series Forecasting",
            "Strong foundation in Data Structures, Algorithms, and System Design"
        ]
    },
    "career": {
        "status": "Fresh Graduate",
        "primary_focus": "Machine Learning and Python Backend Development",
        "secondary_focus": "Full-stack systems",
        "target_roles": [
            "Machine Learning Engineer",
            "Python Backend Developer",
            "AI/ML Engineer",
            "Data Engineer"
        ],
        "interests": [
            "Building scalable ML systems",
            "Backend API development",
            "Computer Vision applications",
            "Time Series Analysis"
        ]
    },
    "skills": {
        "languages": ["Python"],
        "ml": ["Machine Learning", "Deep Learning", "LSTM", "Time Series Forecasting", "Neural Networks"],
        "backend": ["FastAPI", "Flask", "REST APIs", "Microservices"],
        "databases": ["MongoDB", "SQL", "PostgreSQL"],
        "tools": ["Linux", "Git", "Docker", "OpenCV"],
        "other": ["Data Analysis", "Computer Vision", "Problem Solving", "System Design"]
    },
    "projects": [
        {
            "name": "Movie Character Recognition System",
            "description": "Built a computer vision system that identifies movie characters from images using deep learning",
            "technologies": ["Python", "OpenCV", "Deep Learning", "CNN"],
            "highlights": [
                "Implemented facial recognition with high accuracy",
                "Trained custom models on movie character datasets",
                "Built end-to-end pipeline from image processing to prediction"
            ]
        },
        {
            "name": "Stock Price Prediction using LSTM",
            "description": "Time series forecasting system for predicting stock market trends",
            "technologies": ["Python", "LSTM", "TensorFlow", "Data Analysis"],
            "highlights": [
                "Developed LSTM models for sequential data prediction",
                "Implemented feature engineering for financial data",
                "Achieved meaningful predictions on historical stock data"
            ]
        },
        {
            "name": "AI-Powered Q&A System (This Application)",
            "description": "Personal portfolio chatbot using FastAPI and semantic search",
            "technologies": ["FastAPI", "Sentence Transformers", "NLP", "REST APIs"],
            "highlights": [
                "Built semantic search using sentence embeddings",
                "Designed conversational AI with context understanding",
                "Deployed full-stack application with modern UI"
            ]
        }
    ],
    "certifications": [
        "Machine Learning Specialization",
        "Python for Data Science",
        "Deep Learning Fundamentals"
    ],
    "soft_skills": [
        "Quick Learner",
        "Problem Solver",
        "Team Collaboration",
        "Effective Communication",
        "Self-Motivated",
        "Adaptable"
    ]
}


def calculate_age(dob_str):
    dob = date.fromisoformat(dob_str)
    today = date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

AGE = calculate_age(PROFILE["identity"]["date_of_birth"])


PROFILE_TEXT = f"""
Hindav Deshmukh is a {AGE}-year-old Computer Science engineering graduate (Class of 2024) from 
Jawaharlal Darda Institute of Engineering and Technology, affiliated with Sant Gadge Baba Amravati University. 
He graduated with a CGPA of 7.23 and is currently based in Pune, Maharashtra, India.

Hindav is a fresh graduate specializing in Machine Learning and Python Backend Development. His technical 
expertise includes Python programming, Machine Learning, Deep Learning, LSTM networks, Time Series Forecasting, 
Neural Networks, FastAPI, Flask, REST APIs, MongoDB, SQL, PostgreSQL, Linux, Git, Docker, OpenCV, Data Analysis, 
Computer Vision, and System Design.

He has completed several notable projects including a Movie Character Recognition System using Computer Vision 
and Deep Learning, a Stock Price Prediction system using LSTM for time series forecasting, and this AI-powered 
Q&A chatbot using FastAPI and semantic search with Sentence Transformers.

Hindav is seeking entry-level positions as a Machine Learning Engineer, Python Backend Developer, AI/ML Engineer, 
or Data Engineer. He is passionate about building scalable ML systems, backend API development, computer vision 
applications, and time series analysis. His career goal is to become a confident ML/Backend engineer working on 
production-grade systems within 1-2 years.

He holds certifications in Machine Learning Specialization, Python for Data Science, and Deep Learning Fundamentals. 
His soft skills include being a quick learner, problem solver, team player, effective communicator, self-motivated, 
and highly adaptable. He speaks English, Hindi, and Marathi fluently.
"""

GREETINGS = {
    r'\bhi\b': "Hi 👋 I'm Hindav's AI profile assistant! I can tell you about his skills, education, projects, and career goals. What would you like to know?",
    r'\bhello\b': "Hello! 😊 I'm here to answer any questions about Hindav Deshmukh. Feel free to ask about his technical skills, projects, education, or career aspirations!",
    r'\bhey\b': "Hey there! 👋 Ask me anything about Hindav - his ML projects, backend development experience, education, or what he's looking for in his next role!",
    r'\bwho are you\b': "I'm an AI assistant trained on Hindav Deshmukh's professional profile. I can answer questions about his education, skills, projects, and career objectives!"
}

def check_greeting(q):
    q_lower = q.lower().strip()
    if q_lower in ['hi', 'hello', 'hey', 'who are you', 'hi there', 'hello there']:
        for pattern, response in GREETINGS.items():
            if re.search(pattern, q_lower):
                return response
    for pattern, response in GREETINGS.items():
        if re.fullmatch(pattern, q_lower):
            return response
    return None

HR_QA = {
    r'why should (we )?hire (you|hindav)': 
        "Hindav brings a unique combination of strong technical fundamentals, hands-on ML/backend experience, and a proven ability to build production-ready systems. "
        "He has delivered real projects in Computer Vision, Time Series Forecasting, and Backend APIs. As a quick learner with strong problem-solving skills, "
        "he can adapt to new technologies and contribute meaningfully from day one. His focus on continuous improvement and building scalable solutions makes him a valuable addition to any team.",

    r'what are (your|his) (main )?strengths': 
        "Hindav's key strengths include: (1) Strong Python programming and ML fundamentals, (2) Hands-on experience with deep learning frameworks and time series models, "
        "(3) Backend API development using FastAPI and Flask, (4) End-to-end system design and implementation, (5) Quick learning and problem-solving abilities, "
        "(6) Self-motivated with a track record of completing complex projects independently.",

    r'what are (your|his) weaknesses':
        "As a fresh graduate, Hindav is still building industry-level experience at scale. However, he actively addresses this through continuous learning, "
        "personal projects, and staying updated with industry best practices. He's eager to learn from experienced professionals and grow through real-world challenges.",

    r'where do you see (yourself|hindav)':
        "In 1-2 years, Hindav aims to be a confident Machine Learning or Backend Engineer contributing to production systems at scale. "
        "He wants to deepen his expertise in ML engineering, system design, and scalable backend architectures while working on impactful projects. "
        "Long-term, he aspires to lead ML initiatives and architect robust, intelligent systems.",

    r'what (roles|positions|jobs) (are you|is he) looking for':
        "Hindav is actively seeking entry-level positions in: Machine Learning Engineer, Python Backend Developer, AI/ML Engineer, or Data Engineer roles. "
        "He's particularly interested in opportunities where he can work on production ML systems, backend APIs, computer vision applications, or data-intensive projects.",

    r'(are you|is he) open to learning':
        "Absolutely! Hindav is highly committed to continuous learning. He regularly takes online courses, builds personal projects to learn new technologies, "
        "and stays updated with the latest trends in ML and backend development. He's excited to learn from experienced team members and grow in a collaborative environment.",

    r'(tell me about|describe) (your|his) work ethic':
        "Hindav is self-motivated, disciplined, and takes ownership of his projects. He believes in delivering quality work, meeting deadlines, and continuously improving his skills. "
        "His personal projects demonstrate his ability to work independently while his academic background shows he can collaborate effectively in teams.",

    r'what motivates (you|him)':
        "Hindav is motivated by building intelligent systems that solve real-world problems. He enjoys the challenge of working with complex data, "
        "designing efficient algorithms, and seeing his work impact users. The intersection of ML and backend engineering excites him because it combines analytical thinking with practical system building.",

    r'how do you handle (pressure|stress|deadlines)':
        "Hindav manages pressure by breaking down complex problems into manageable tasks, prioritizing effectively, and maintaining clear communication. "
        "His project experience has taught him to stay focused under deadlines while maintaining code quality. He believes in proactive problem-solving and asking for help when needed.",

    r'why (machine learning|ml|backend|python)':
        "Hindav chose Machine Learning and Backend Development because he's fascinated by how intelligent systems can learn from data and make predictions. "
        "He enjoys the combination of mathematics, programming, and practical problem-solving that ML offers. Backend development complements this by allowing him to build "
        "scalable systems that serve ML models in production. Python ties everything together as a versatile language for both ML and backend work."
}

def check_hr(q):
    q_lower = q.lower()
    for pattern, answer in HR_QA.items():
        if re.search(pattern, q_lower):
            return answer  
    return None
def split_chunks(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

model = SentenceTransformer("all-MiniLM-L6-v2")
chunks = split_chunks(PROFILE_TEXT)
embeddings = model.encode(chunks)

def semantic_answer(question):
    q_emb = model.encode([question])
    scores = cosine_similarity(q_emb, embeddings)[0]
    idx = int(np.argmax(scores))
    conf = float(scores[idx])

    if conf < 0.25:
        return "I don't have specific information on that. Try asking about Hindav's skills, projects, education, career goals, or soft skills!", conf
    return chunks[idx], conf
class QuestionRequest(BaseModel):
    question: str
async def handle_question(question: str):
    ql = question.lower().strip()
    
    if any(word in ql for word in ["skill", "technology", "technologies", "tech stack", "stack", "tools", "know", "use", "work with"]):
        all_skills = (
            PROFILE["skills"]["languages"]
            + PROFILE["skills"]["ml"]
            + PROFILE["skills"]["backend"]
            + PROFILE["skills"]["databases"]
            + PROFILE["skills"]["tools"]
            + PROFILE["skills"]["other"] )
        return (
            f"Hindav has expertise in the following technologies:\n\n"
            f"🐍 Programming: {', '.join(PROFILE['skills']['languages'])}\n"
            f"🤖 ML/AI: {', '.join(PROFILE['skills']['ml'])}\n"
            f"⚙️ Backend: {', '.join(PROFILE['skills']['backend'])}\n"
            f"💾 Databases: {', '.join(PROFILE['skills']['databases'])}\n"
            f"🛠️ Tools: {', '.join(PROFILE['skills']['tools'])}\n"
            f"📊 Other: {', '.join(PROFILE['skills']['other'])}",
            1.0
        )

    if any(word in ql for word in ["project", "built", "created", "developed", "portfolio", "work"]):
        projects_info = "Hindav has completed several notable projects:\n\n"
        for i, proj in enumerate(PROFILE["projects"], 1):
            projects_info += f"{i}. **{proj['name']}**\n   {proj['description']}\n   Technologies: {', '.join(proj['technologies'])}\n\n"
        return projects_info.strip(), 1.0

    if any(word in ql for word in ["education", "degree", "college", "university", "graduate", "study", "studied", "cgpa", "gpa"]):
        edu = PROFILE["education"]
        return (
            f"🎓 Hindav graduated in {edu['graduation_year']} with a {edu['degree']} from "
            f"{edu['college']}, affiliated with {edu['university']}. "
            f"His CGPA is {edu['cgpa']}. During his studies, he focused on Machine Learning, Deep Learning, "
            f"Computer Vision, Data Structures, Algorithms, and System Design.",
            1.0
        )
    if any(word in ql for word in ["certification", "certificate", "course", "certified"]):
        certs = PROFILE["certifications"]
        return (
            f"Hindav holds the following certifications:\n" + "\n".join([f"• {cert}" for cert in certs]),
            1.0
        )
    if any(word in ql for word in ["location", "live", "based", "from", "where"]):
        if not any(word in ql for word in ["see yourself", "future"]):
            return f"Hindav is currently based in {PROFILE['identity']['location']}.", 1.0
    if "age" in ql or "old" in ql:
        return f"Hindav is {AGE} years old (born on {PROFILE['identity']['date_of_birth']}).", 1.0
 
    if "name" in ql and len(ql.split()) <= 6:
        return f"His name is {PROFILE['identity']['name']}.", 1.0
   
    if any(word in ql for word in ["career", "goal", "looking for", "seeking", "target", "role", "job", "position", "opportunity"]):
        roles = PROFILE["career"]["target_roles"]
        interests = PROFILE["career"]["interests"]
        return (
            f"Hindav is seeking entry-level positions as: {', '.join(roles)}. "
            f"He is passionate about {', '.join(interests)}. His goal is to become a confident ML/Backend engineer "
            f"working on production-grade systems within 1-2 years.",
            1.0
        )
    if any(word in ql for word in ["language speak", "languages speak", "speak", "fluent"]):
        langs = PROFILE["identity"]["languages"]
        return f"Hindav speaks {', '.join(langs)} fluently.", 1.0

    if any(word in ql for word in ["soft skill", "personality", "qualities", "traits"]):
        soft = PROFILE["soft_skills"]
        return f"Hindav's soft skills include: {', '.join(soft)}.", 1.0

    if any(word in ql for word in ["experience", "fresher", "experienced", "years"]):
        return (
            f"Hindav is a fresh graduate (Class of 2024) with hands-on project experience in ML and backend development. "
            f"While he doesn't have formal industry experience, he has built production-ready systems including Computer Vision apps, "
            f"Time Series forecasting models, and REST APIs.",
            1.0
        )

    hr = check_hr(question)
    if hr:
        return hr, 0.95
    g = check_greeting(question)
    if g:
        return g, 1.0
    return semantic_answer(question)

@app.post("/ask")
async def ask(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")

    answer, confidence = await handle_question(request.question)
    return {"answer": answer, "confidence": round(confidence, 2)}

@app.post("/api/ask")
async def ask_api(request: QuestionRequest):
    answer, confidence = await handle_question(request.question)
    return {
        "answer": answer,
        "confidence": round(confidence, 2),
        "success": True
    }

@app.get("/health")
async def health():
    return {"status": "ok", "model": "all-MiniLM-L6-v2"}
