from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="SHL Assessment Recommendation System")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://shl-recommendation.vercel.app",
        "http://localhost:3000",
        "http://localhost:5000"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=86400  # 24 hours
)

# Initialize Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# Initialize sentence transformer
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Load and prepare assessment data
class Assessment(BaseModel):
    name: str
    url: str
    remote_testing: bool
    adaptive_irt: bool
    duration: str
    test_type: str
    description: str

# Sample assessment data (in production, this would come from a database)
ASSESSMENTS = [
    {
        "name": "SHL Verify G+ Cognitive Ability Test",
        "url": "https://www.shl.com/solutions/products/verify-g-plus/",
        "remote_testing": True,
        "adaptive_irt": True,
        "duration": "24 minutes",
        "test_type": "Cognitive",
        "description": "Measures critical reasoning through numerical, verbal, and abstract tests"
    },
    {
        "name": "SHL Coding Pro",
        "url": "https://www.shl.com/solutions/products/coding-pro/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "60 minutes",
        "test_type": "Technical",
        "description": "Evaluates programming skills in multiple languages including Java, Python, JavaScript"
    },
    {
        "name": "SHL Personality Assessment (OPQ)",
        "url": "https://www.shl.com/solutions/products/opq/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "25 minutes",
        "test_type": "Personality",
        "description": "Measures workplace behavioral styles and preferences"
    },
    {
        "name": "SHL SQL Pro",
        "url": "https://www.shl.com/solutions/products/sql-pro/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "45 minutes",
        "test_type": "Technical",
        "description": "Tests SQL querying and database manipulation skills"
    },
    {
        "name": "SHL Verify Interactive",
        "url": "https://www.shl.com/solutions/products/verify-interactive/",
        "remote_testing": True,
        "adaptive_irt": True,
        "duration": "30 minutes",
        "test_type": "Cognitive",
        "description": "Interactive cognitive ability test with dynamic problem-solving scenarios"
    },
    {
        "name": "SHL Mechanical Comprehension Test",
        "url": "https://www.shl.com/solutions/products/mechanical-comprehension/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "20 minutes",
        "test_type": "Technical",
        "description": "Evaluates understanding of mechanical principles and concepts"
    },
    {
        "name": "SHL Situational Judgment Test",
        "url": "https://www.shl.com/solutions/products/situational-judgment/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "25 minutes",
        "test_type": "Behavioral",
        "description": "Assesses decision-making and problem-solving in workplace scenarios"
    },
    {
        "name": "SHL Talent Assessment",
        "url": "https://www.shl.com/solutions/products/talent-assessment/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "35 minutes",
        "test_type": "Personality",
        "description": "Comprehensive personality and behavioral assessment for talent development"
    },
    {
        "name": "SHL Numerical Reasoning Test",
        "url": "https://www.shl.com/solutions/products/numerical-reasoning/",
        "remote_testing": True,
        "adaptive_irt": True,
        "duration": "18 minutes",
        "test_type": "Cognitive",
        "description": "Measures ability to interpret and analyze numerical data"
    },
    {
        "name": "SHL Verbal Reasoning Test",
        "url": "https://www.shl.com/solutions/products/verbal-reasoning/",
        "remote_testing": True,
        "adaptive_irt": True,
        "duration": "19 minutes",
        "test_type": "Cognitive",
        "description": "Evaluates ability to understand and analyze written information"
    },
    {
        "name": "SHL Abstract Reasoning Test",
        "url": "https://www.shl.com/solutions/products/abstract-reasoning/",
        "remote_testing": True,
        "adaptive_irt": True,
        "duration": "20 minutes",
        "test_type": "Cognitive",
        "description": "Measures ability to identify patterns and solve abstract problems"
    },
    {
        "name": "SHL Inductive Reasoning Test",
        "url": "https://www.shl.com/solutions/products/inductive-reasoning/",
        "remote_testing": True,
        "adaptive_irt": True,
        "duration": "25 minutes",
        "test_type": "Cognitive",
        "description": "Evaluates ability to identify patterns and draw logical conclusions"
    },
    {
        "name": "SHL Deductive Reasoning Test",
        "url": "https://www.shl.com/solutions/products/deductive-reasoning/",
        "remote_testing": True,
        "adaptive_irt": True,
        "duration": "20 minutes",
        "test_type": "Cognitive",
        "description": "Measures ability to apply logical rules to reach conclusions"
    },
    {
        "name": "SHL Spatial Reasoning Test",
        "url": "https://www.shl.com/solutions/products/spatial-reasoning/",
        "remote_testing": True,
        "adaptive_irt": True,
        "duration": "25 minutes",
        "test_type": "Cognitive",
        "description": "Evaluates ability to visualize and manipulate 2D and 3D objects"
    },
    {
        "name": "SHL Error Checking Test",
        "url": "https://www.shl.com/solutions/products/error-checking/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "15 minutes",
        "test_type": "Cognitive",
        "description": "Measures attention to detail and accuracy in identifying errors"
    },
    {
        "name": "Account Manager Solution",
        "url": "https://www.shl.com/solutions/products/account-manager/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "60 minutes",
        "test_type": "Cognitive, Personality, Ability, Behavioral",
        "description": "Comprehensive assessment for account management roles"
    },
    {
        "name": "Administrative Professional - Short Form",
        "url": "https://www.shl.com/solutions/products/administrative-professional/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "45 minutes",
        "test_type": "Ability, Knowledge, Personality",
        "description": "Assessment for administrative professional roles"
    },
    {
        "name": "Agency Manager Solution",
        "url": "https://www.shl.com/solutions/products/agency-manager/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "55 minutes",
        "test_type": "Ability, Behavioral, Personality, Situational Judgment",
        "description": "Comprehensive assessment for agency management positions"
    },
    {
        "name": "Apprentice + 8.0 Job Focused Assessment",
        "url": "https://www.shl.com/solutions/products/apprentice-plus/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "40 minutes",
        "test_type": "Behavioral, Personality",
        "description": "Assessment focused on apprentice roles"
    },
    {
        "name": "Bank Administrative Assistant - Short Form",
        "url": "https://www.shl.com/solutions/products/bank-admin-assistant/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "50 minutes",
        "test_type": "Ability, Behavioral, Knowledge, Personality",
        "description": "Specialized assessment for bank administrative roles"
    },
    {
        "name": "Bank Collections Agent - Short Form",
        "url": "https://www.shl.com/solutions/products/bank-collections/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "45 minutes",
        "test_type": "Ability, Behavioral, Personality",
        "description": "Assessment for bank collections positions"
    },
    {
        "name": "Bank Operations Supervisor - Short Form",
        "url": "https://www.shl.com/solutions/products/bank-operations/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "55 minutes",
        "test_type": "Ability, Behavioral, Personality, Situational Judgment",
        "description": "Assessment for bank operations supervision roles"
    },
    {
        "name": "Bilingual Spanish Reservation Agent Solution",
        "url": "https://www.shl.com/solutions/products/bilingual-reservation/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "50 minutes",
        "test_type": "Behavioral, Personality, Situational Judgment, Ability",
        "description": "Bilingual assessment for reservation agent positions"
    },
    {
        "name": "Bookkeeping, Accounting, Auditing Clerk Short Form",
        "url": "https://www.shl.com/solutions/products/bookkeeping-clerk/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "60 minutes",
        "test_type": "Personality, Situational Judgment, Knowledge, Behavioral, Ability",
        "description": "Comprehensive assessment for accounting roles"
    },
    {
        "name": "Branch Manager - Short Form",
        "url": "https://www.shl.com/solutions/products/branch-manager/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "55 minutes",
        "test_type": "Ability, Behavioral, Personality",
        "description": "Assessment for branch management positions"
    },
    {
        "name": "Cashier Solution",
        "url": "https://www.shl.com/solutions/products/cashier/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "40 minutes",
        "test_type": "Behavioral, Ability, Personality",
        "description": "Assessment for cashier positions"
    },
    {
        "name": "Global Skills Development Report",
        "url": "https://www.shl.com/solutions/products/global-skills/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "75 minutes",
        "test_type": "Ability, Emotional Intelligence, Behavioral, Cognitive, Developmental, Personality",
        "description": "Comprehensive development assessment for global skills"
    },
    {
        "name": ".NET Framework 4.5",
        "url": "https://www.shl.com/solutions/products/dotnet-framework/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "45 minutes",
        "test_type": "Knowledge",
        "description": "Technical assessment for .NET Framework 4.5"
    },
    {
        "name": ".NET MVC",
        "url": "https://www.shl.com/solutions/products/dotnet-mvc/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "45 minutes",
        "test_type": "Knowledge",
        "description": "Technical assessment for .NET MVC"
    },
    {
        "name": ".NET MVVM",
        "url": "https://www.shl.com/solutions/products/dotnet-mvvm/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "45 minutes",
        "test_type": "Knowledge",
        "description": "Technical assessment for .NET MVVM"
    },
    {
        "name": ".NET WCF",
        "url": "https://www.shl.com/solutions/products/dotnet-wcf/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "45 minutes",
        "test_type": "Knowledge",
        "description": "Technical assessment for .NET WCF"
    },
    {
        "name": ".NET WPF",
        "url": "https://www.shl.com/solutions/products/dotnet-wpf/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "45 minutes",
        "test_type": "Knowledge",
        "description": "Technical assessment for .NET WPF"
    },
    {
        "name": ".NET XAML",
        "url": "https://www.shl.com/solutions/products/dotnet-xaml/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "45 minutes",
        "test_type": "Knowledge",
        "description": "Technical assessment for .NET XAML"
    },
    {
        "name": "Accounts Payable",
        "url": "https://www.shl.com/solutions/products/accounts-payable/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "45 minutes",
        "test_type": "Knowledge",
        "description": "Technical assessment for accounts payable"
    },
    {
        "name": "Accounts Payable Simulation",
        "url": "https://www.shl.com/solutions/products/accounts-payable-simulation/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "45 minutes",
        "test_type": "Situational Judgment",
        "description": "Simulation assessment for accounts payable scenarios"
    },
    {
        "name": "Accounts Receivable",
        "url": "https://www.shl.com/solutions/products/accounts-receivable/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "45 minutes",
        "test_type": "Knowledge",
        "description": "Technical assessment for accounts receivable"
    },
    {
        "name": "Accounts Receivable Simulation",
        "url": "https://www.shl.com/solutions/products/accounts-receivable-simulation/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "45 minutes",
        "test_type": "Situational Judgment",
        "description": "Simulation assessment for accounts receivable scenarios"
    },
    {
        "name": "ADO.NET",
        "url": "https://www.shl.com/solutions/products/ado-net/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "45 minutes",
        "test_type": "Knowledge",
        "description": "Technical assessment for ADO.NET"
    },
    {
        "name": "Industrial - Semi-skilled 7.1 (Americas)",
        "url": "https://www.shl.com/solutions/products/industrial-semi-skilled-americas/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "45 minutes",
        "test_type": "Behavioral",
        "description": "Assessment for semi-skilled industrial roles in the Americas region"
    },
    {
        "name": "Industrial - Semi-skilled 7.1 (International)",
        "url": "https://www.shl.com/solutions/products/industrial-semi-skilled-international/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "45 minutes",
        "test_type": "Behavioral",
        "description": "Assessment for semi-skilled industrial roles in international markets"
    },
    {
        "name": "Industrial Professional and Skilled 7.1 (Americas)",
        "url": "https://www.shl.com/solutions/products/industrial-professional-americas/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "60 minutes",
        "test_type": "Ability, Behavioral",
        "description": "Comprehensive assessment for professional and skilled industrial roles in the Americas"
    },
    {
        "name": "Industrial Professional and Skilled 7.1 Solution",
        "url": "https://www.shl.com/solutions/products/industrial-professional-solution/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "60 minutes",
        "test_type": "Ability, Behavioral",
        "description": "Complete solution for assessing professional and skilled industrial roles"
    },
    {
        "name": "Installation and Repair Technician Solution",
        "url": "https://www.shl.com/solutions/products/installation-repair-technician/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "55 minutes",
        "test_type": "Ability, Behavioral, Personality, Situational Judgment",
        "description": "Comprehensive assessment for installation and repair technician roles"
    },
    {
        "name": "Insurance Account Manager Solution",
        "url": "https://www.shl.com/solutions/products/insurance-account-manager/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "60 minutes",
        "test_type": "Ability, Behavioral, Personality, Situational Judgment",
        "description": "Complete assessment solution for insurance account management roles"
    },
    {
        "name": "Insurance Administrative Assistant Solution",
        "url": "https://www.shl.com/solutions/products/insurance-admin-assistant/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "55 minutes",
        "test_type": "Ability, Behavioral, Personality, Situational Judgment",
        "description": "Comprehensive assessment for insurance administrative assistant roles"
    },
    {
        "name": "Insurance Agent Solution",
        "url": "https://www.shl.com/solutions/products/insurance-agent/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "50 minutes",
        "test_type": "Ability, Behavioral, Personality",
        "description": "Complete assessment solution for insurance agent roles"
    },
    {
        "name": "Insurance Director Solution",
        "url": "https://www.shl.com/solutions/products/insurance-director/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "60 minutes",
        "test_type": "Ability, Behavioral, Personality",
        "description": "Comprehensive assessment for insurance director roles"
    },
    {
        "name": "Insurance Sales Manager Solution",
        "url": "https://www.shl.com/solutions/products/insurance-sales-manager/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "60 minutes",
        "test_type": "Ability, Behavioral, Personality, Situational Judgment",
        "description": "Complete assessment solution for insurance sales management roles"
    },
    {
        "name": "Manager - Short Form",
        "url": "https://www.shl.com/solutions/products/manager-short-form/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "55 minutes",
        "test_type": "Ability, Behavioral, Knowledge, Personality, Situational Judgment",
        "description": "Comprehensive assessment for management roles"
    },
    {
        "name": "Manager + 7.0 Solution",
        "url": "https://www.shl.com/solutions/products/manager-plus/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "60 minutes",
        "test_type": "Ability, Behavioral, Cognitive",
        "description": "Advanced assessment solution for management roles"
    },
    {
        "name": "COBOL Programming",
        "url": "https://www.shl.com/solutions/products/cobol-programming/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "45 minutes",
        "test_type": "Knowledge",
        "description": "Technical assessment for COBOL programming skills"
    },
    {
        "name": "Computer Science",
        "url": "https://www.shl.com/solutions/products/computer-science/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "45 minutes",
        "test_type": "Knowledge",
        "description": "Technical assessment for computer science knowledge"
    },
    {
        "name": "Contact Center Call Simulation",
        "url": "https://www.shl.com/solutions/products/contact-center-simulation/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "45 minutes",
        "test_type": "Situational Judgment",
        "description": "Simulation-based assessment for contact center roles"
    },
    {
        "name": "Conversational Multichat Simulation",
        "url": "https://www.shl.com/solutions/products/conversational-multichat/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "45 minutes",
        "test_type": "Situational Judgment",
        "description": "Simulation assessment for handling multiple chat conversations"
    },
    {
        "name": "Core Java (Advanced Level)",
        "url": "https://www.shl.com/solutions/products/core-java-advanced/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "45 minutes",
        "test_type": "Knowledge",
        "description": "Advanced level technical assessment for Java programming"
    },
    {
        "name": "Core Java (Entry Level)",
        "url": "https://www.shl.com/solutions/products/core-java-entry/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "45 minutes",
        "test_type": "Knowledge",
        "description": "Entry level technical assessment for Java programming"
    },
    {
        "name": "Count Out The Money",
        "url": "https://www.shl.com/solutions/products/count-out-money/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "45 minutes",
        "test_type": "Knowledge, Situational Judgment",
        "description": "Assessment for cash handling and money counting skills"
    },
    {
        "name": "CSS3",
        "url": "https://www.shl.com/solutions/products/css3/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "45 minutes",
        "test_type": "Knowledge",
        "description": "Technical assessment for CSS3 skills"
    },
    {
        "name": "Culinary Skills",
        "url": "https://www.shl.com/solutions/products/culinary-skills/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "45 minutes",
        "test_type": "Knowledge",
        "description": "Assessment for culinary skills and knowledge"
    },
    {
        "name": "Customer Service Phone Simulation",
        "url": "https://www.shl.com/solutions/products/customer-service-phone-simulation/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "45 minutes",
        "test_type": "Behavioral, Situational Judgment",
        "description": "Simulation-based assessment for phone customer service roles"
    },
    {
        "name": "Customer Service Phone Solution",
        "url": "https://www.shl.com/solutions/products/customer-service-phone/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "50 minutes",
        "test_type": "Behavioral, Personality, Situational Judgment",
        "description": "Complete assessment solution for phone customer service roles"
    },
    {
        "name": "Cyber Risk",
        "url": "https://www.shl.com/solutions/products/cyber-risk/",
        "remote_testing": True,
        "adaptive_irt": False,
        "duration": "45 minutes",
        "test_type": "Knowledge",
        "description": "Technical assessment for cyber risk knowledge and skills"
    }
]

# Create embeddings for assessments
assessment_embeddings = encoder.encode([f"{a['name']} {a['description']} {a['test_type']}" for a in ASSESSMENTS])

class RecommendationResponse(BaseModel):
    recommendations: List[Assessment]
    explanation: str

def get_recommendations(query: str, max_results: int = 10) -> RecommendationResponse:
    # Generate query embedding
    query_embedding = encoder.encode([query])
    
    # Calculate similarities
    similarities = cosine_similarity(query_embedding, assessment_embeddings)[0]
    
    # Get top indices
    top_indices = np.argsort(similarities)[::-1][:max_results]
    
    # Get recommendations
    recommendations = [Assessment(**ASSESSMENTS[i]) for i in top_indices if similarities[i] > 0.3]
    
    # Generate explanation using Gemini
    prompt = f"""Given the query: "{query}"
    I recommended these assessments: {[rec.name for rec in recommendations]}
    Please provide a brief explanation (2-3 sentences) of why these assessments are relevant."""
    
    try:
        response = model.generate_content(prompt)
        explanation = response.text
    except Exception as e:
        explanation = f"Unable to generate explanation: {str(e)}"
    
    return RecommendationResponse(recommendations=recommendations, explanation=explanation)

@app.get("/api/recommend", response_model=RecommendationResponse)
async def recommend(
    query: str = Query(..., description="Natural language query or job description"),
    max_results: int = Query(10, ge=1, le=10)
):
    try:
        return get_recommendations(query, max_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class TextRequest(BaseModel):
    text: str
    max_results: Optional[int] = 10

@app.post("/api/recommend/text", response_model=RecommendationResponse)
async def recommend_from_text(request: TextRequest):
    try:
        return get_recommendations(request.text, request.max_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/recommend/url", response_model=RecommendationResponse)
async def recommend_from_url(
    url: str = Query(..., description="URL of the job description"),
    max_results: int = Query(10, ge=1, le=10)
):
    # In production, this would fetch and parse the URL content
    raise HTTPException(status_code=501, detail="URL processing not implemented yet") 