# openai_client.py
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load API key from file
load_dotenv("api_key.env")  # Make sure this file exists
api_key = os.getenv("OPENAI_API_KEY")

# Initialize and export client
client = OpenAI(api_key=api_key)