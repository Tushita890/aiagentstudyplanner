import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel

# Load environment variables
load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION", "us-central1")
MODEL_NAME = os.getenv("MODEL", "gemini-1.5-flash")

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel(MODEL_NAME)

# FastAPI app
app = FastAPI()

# Request model
class StudyRequest(BaseModel):
    subjects: list[str]
    hours_per_day: int
    days_left: int
    weak_subjects: list[str]

# ----------- AGENTS -----------

# Planner Agent
def planner_agent(subjects, hours_per_day, days_left):
    prompt = f"""
You are an AI Study Planner.

Create a structured daily study plan.

Subjects: {subjects}
Study hours per day: {hours_per_day}
Days left: {days_left}

Make a clear daily schedule with time distribution and revision.
"""
    response = model.generate_content(prompt)
    return response.text

# Optimizer Agent
def optimizer_agent(plan, weak_subjects):
    prompt = f"""
You are a Study Plan Optimizer.

Improve this plan by prioritizing weak subjects.

Weak subjects: {weak_subjects}

Study Plan:
{plan}
"""
    response = model.generate_content(prompt)
    return response.text

# Motivation Agent
def motivation_agent(plan):
    prompt = f"""
You are a Study Coach.

Give:
- Motivation tips
- Productivity tips
- Study techniques

Based on this plan:

{plan}
"""
    response = model.generate_content(prompt)
    return response.text

# Root Sequential Agent
def generate_study_plan(subjects, hours_per_day, days_left, weak_subjects):
    base_plan = planner_agent(subjects, hours_per_day, days_left)
    optimized_plan = optimizer_agent(base_plan, weak_subjects)
    tips = motivation_agent(optimized_plan)

    final_output = f"""
STUDY PLAN:
{optimized_plan}

TIPS:
{tips}
"""
    return final_output

# ----------- API ENDPOINTS -----------

@app.get("/")
def home():
    return {
        "message": "Hello! I'm your AI Study Planner. Send study details to generate a plan."
    }

@app.post("/study-plan")
def study_plan(request: StudyRequest):
    plan = generate_study_plan(
        request.subjects,
        request.hours_per_day,
        request.days_left,
        request.weak_subjects
    )
    return {"study_plan": plan}

# Run server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
from google.adk.agents import Agent

def study_planner_root_agent(input: dict):
    try:
        response = model.generate_content("Say hello")
        return {"output": response.text}
    except Exception as e:
        return {"output": f"Error: {str(e)}"}

root_agent = Agent(
    name="ai_study_planner",
    description="Study planner",
    entrypoint=study_planner_root_agent,
)
