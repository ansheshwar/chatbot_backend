from flask import Flask, request, jsonify
from pyngrok import ngrok
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)

# Configure the AI model
try:
    google_llm = GoogleGenerativeAI(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-pro"  # Replace with "chat-bison-001" if needed
    )
except Exception as e:
    print(f"Error initializing GoogleGenerativeAI: {e}")
    google_llm = None

# Define a prompt template for meaningful responses
prompt_template = PromptTemplate(
    input_variables=["user_message"],
    template="You are AI Mitra, an intelligent assistant specializing in the Ganga river. Provide a helpful response to the query: {user_message}"
)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Parse user input
        data = request.json
        user_message = data.get("message")
        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        if not google_llm:
            return jsonify({"error": "AI model not initialized. Check server logs for details."}), 500

        # Format the prompt using LangChain
        formatted_prompt = prompt_template.format(user_message=user_message)

        # Generate response using the AI model
        response = google_llm.invoke(formatted_prompt)

        # Return the response
        return jsonify({"response": response.strip()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Start Ngrok tunnel for exposing Flask app
    try:
        public_url = ngrok.connect(5000)
        print(f"Public URL: {public_url}")
    except Exception as e:
        print(f"Error starting Ngrok: {e}")
        public_url = "http://localhost:5000"  # Fallback to local server

    # Start Flask app
    app.run(port=5000)
