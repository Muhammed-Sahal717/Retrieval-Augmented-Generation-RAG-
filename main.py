import os

from dotenv import load_dotenv
from google import genai
from google.genai import errors


def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Initialize the client (it will automatically pick up GEMINI_API_KEY from env if available)
    # But we can also pass it explicitly if we want to be sure.
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment variables or .env file.")
        return

    print("Initializing Gemini client...")
    client = genai.Client(api_key=api_key)
    
    try:
        print("Testing connectivity to Gemini API...")
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents='Say hello in a very short, cheerful way.'
        )
        print("\nSuccess! Gemini API is working.")
        print(f"Gemini response: {response.text}")
    except errors.APIError as e:
        print(f"\nFailed to connect to Gemini API: {e}")

if __name__ == "__main__":
    main()
