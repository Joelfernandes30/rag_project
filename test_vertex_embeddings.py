import os
import subprocess
from dotenv import load_dotenv
from google import genai
from google.oauth2.credentials import Credentials

# Load environment variables from .env
load_dotenv()


def test_google_vertex_embeddings():
    print("=" * 60)
    print("TESTING GOOGLE VERTEX AI EMBEDDING GENERATION")
    print("=" * 60)

    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    print(f"Project ID : {project}")
    print(f"Location   : {location}")
    print(f"API Key    : {'Configured' if api_key else 'Not set (using Vertex AI gcloud auth)'}")
    print("-" * 60)

    try:
        # Initialize Google GenAI Client
        if api_key:
            client = genai.Client(api_key=api_key)
        elif project:
            token = subprocess.check_output("gcloud auth print-access-token", shell=True, text=True).strip()
            creds = Credentials(token)
            client = genai.Client(vertexai=True, project=project, location=location, credentials=creds)
        else:
            raise ValueError("Neither GEMINI_API_KEY nor GOOGLE_CLOUD_PROJECT found in .env")

        test_text = "Vaidya Charaka classical Ayurvedic remedy for Vata dosha"
        print(f"Input Text: '{test_text}'")
        print("Generating 768-dimensional embedding via Google text-embedding-004...")

        res = client.models.embed_content(
            model="text-embedding-004",
            contents=test_text,
            config={"output_dimensionality": 768}
        )

        vector = res.embeddings[0].values
        print("[SUCCESS] Vertex AI Embedding Generated Successfully.")
        print(f"Vector Length (Dimensions): {len(vector)}")
        print(f"Sample Vector Values      : {vector[:5]}...")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"[ERROR] Testing Vertex AI Embeddings: {e}")
        print("=" * 60)
        return False


if __name__ == "__main__":
    test_google_vertex_embeddings()
