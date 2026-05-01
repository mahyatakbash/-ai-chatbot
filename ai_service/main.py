from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI
import os
import logging
import time
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("app.log", encoding="utf-8"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_id: str = "default_user"
    message: str

conversations = {}
user_last_requests = {}
SYSTEM_PROMPT = "You are a professional AI Assistant. Answer in the user's language clearly and concisely."

def check_rate_limit(user_id: str):
    current_time = time.time()
    if user_id not in user_last_requests:
        user_last_requests[user_id] = []
    user_last_requests[user_id] = [t for t in user_last_requests[user_id] if current_time - t < 60]
    if len(user_last_requests[user_id]) >= 5:
        return False
    user_last_requests[user_id].append(current_time)
    return True

@app.get("/")
def home():
    return {"message": "AI Chatbot Service is running 🚀"}

@app.post("/chat")
async def chat(chat_data: ChatRequest):
    user_id = chat_data.user_id
    user_message = chat_data.message

    logger.info(f"REQUEST | user_id={user_id} | message={user_message}")

    if not check_rate_limit(user_id):
        logger.warning(f"RATE LIMIT HIT | user_id={user_id}")
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a minute.")

    if user_id not in conversations:
        conversations[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    conversations[user_id].append({"role": "user", "content": user_message})

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=conversations[user_id],
        )
        ai_response = completion.choices[0].message.content
        conversations[user_id].append({"role": "assistant", "content": ai_response})
        
        logger.info(f"RESPONSE | user_id={user_id}")
        return {"response": ai_response}

    except Exception as e:
        logger.error(f"ERROR | {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
