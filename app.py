from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from googletrans import Translator

# Load trained chatbot model and vectorizer
with open("chatbot_model.pkl", "rb") as model_file:
    clf = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

with open("responses.pkl", "rb") as resp_file:
    responses = pickle.load(resp_file)

app = Flask(__name__)
translator = Translator()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    user_lang = data.get("language", "en")  # ✅ Get language from request

    if not user_message:
        return jsonify({"response": "Sorry, I didn't understand that."})

    print(f"User Message in {user_lang}: {user_message}")

    # Translate user message to English
    try:
        translated_text = translator.translate(user_message, src=user_lang, dest="en").text
        print(f"Translated to English: {translated_text}")
    except Exception as e:
        return jsonify({"response": f"Translation error (user input): {str(e)}"})

    # Convert to vector and predict intent
    user_vector = vectorizer.transform([translated_text])
    predicted_intent = clf.predict(user_vector)[0]

    print(f"Predicted intent: {predicted_intent}")

    # Get bot response
    bot_response = np.random.choice(responses.get(predicted_intent, ["Sorry, I don't understand."]))

    print(f"Bot response (English): {bot_response}")

    # Translate bot response back to the user's language
    try:
        translated_response = translator.translate(bot_response, src="en", dest=user_lang).text
        print(f"Translated bot response in {user_lang}: {translated_response}")
    except Exception as e:
        return jsonify({"response": f"Translation error (bot response): {str(e)}"})

    return jsonify({"response": translated_response})

if __name__ == "__main__":
    app.run(debug=True)
