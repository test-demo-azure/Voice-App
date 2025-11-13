# main.py

import os
import tempfile
import time
import json
import shutil
import google.generativeai as genai
from google.api_core import exceptions
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


import dotenv
dotenv.load_dotenv()
from faster_whisper import WhisperModel


# --- Configuration ---
# Make sure to set your API key in your environment variables
# For example: export GOOGLE_API_KEY="your_api_key"
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    raise Exception("GOOGLE_API_KEY environment variable not set.")

# --- FastAPI App Initialization ---
app = FastAPI()

# Configure CORS to allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for simplicity. For production, restrict this.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Loading ---
# Initialize the Gemini model
llm_model = genai.GenerativeModel('gemini-2.5-flash-lite')

# Initialize the Whisper model for speech-to-text
# This runs on the server's CPU. "tiny.en" is a small, fast English model.
stt_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

# --- API Endpoint ---
@app.post("/process-audio/")
async def process_audio_file(audio_file: UploadFile = File(...), history: str = Form("[]")):
    """
    This endpoint receives an audio file and conversation history. It transcribes the audio,
    sends the new query along with the history to Gemini, and returns the LLM's response.
    """
    if not audio_file:
        raise HTTPException(status_code=400, detail="No audio file sent.")

    try:
        # Create a temporary file to save the upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=audio_file.filename) as tmp_file:
            shutil.copyfileobj(audio_file.file, tmp_file)
            tmp_file_path = tmp_file.name

        # 1. Transcribe Audio using the local Whisper model
        segments, info = stt_model.transcribe(tmp_file_path, beam_size=5)
        transcript_text = "".join(segment.text for segment in segments).strip()

        if not transcript_text:
            raise HTTPException(status_code=400, detail="No speech detected in the audio.")

        # 2. Start a chat session with Gemini using the provided history
        history_data = json.loads(history)
        chat = llm_model.start_chat(history=history_data)

        # 3. Get a response from Gemini using the new transcribed text
        prompt = f"You are a helpful assistant. Please provide a concise response to the following user query: '{transcript_text}'"
        
        llm_response_text = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = chat.send_message(prompt)
                llm_response_text = response.text
                break # If successful, exit the loop
            except exceptions.ResourceExhausted as e:
                if attempt < max_retries - 1:
                    print(f"Rate limit hit. Retrying in 2 seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(2) # Wait for 2 seconds before retrying
                else:
                    print("Max retries reached. Failing the request.")
                    raise e # Re-raise the exception if all retries fail

        if llm_response_text is None:
            raise HTTPException(status_code=503, detail="The service is currently unavailable after multiple retries.")

        # Clean the response to remove markdown characters like asterisks
        cleaned_response_text = llm_response_text.replace('*', '').strip()

        # Return Gemini's Response
        return JSONResponse(content={"transcript": transcript_text, "response": cleaned_response_text})

    except Exception as e:
        # Log the error for debugging
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(e)}")
    finally:
        # Clean up the temporary file
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


@app.get("/")
def read_root():
    return {"message": "Voice Assistant Backend is running."}