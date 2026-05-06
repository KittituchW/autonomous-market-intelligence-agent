from google import genai
import os


# Check that API key exists
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("Missing GEMINI_API_KEY. Please set it first.")

# Create Gemini client
client = genai.Client()

# Send a simple test prompt
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Explain what an API is in one simple sentence."
)

print(response.text)