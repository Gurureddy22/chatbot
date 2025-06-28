# Customer Support Chatbot

A **Customer Support Chatbot** is a virtual assistant that uses predefined rules or AI to handle customer inquiries, understanding questions and providing instant answers. This project implements such a chatbot with a Python (Flask) backend and a web-based frontend (HTML/CSS/JavaScript). The bot loads a pretrained intent-classification model and response dataset, allowing it to classify user queries and return a relevant answer. It also integrates Google Translate to support multiple input/output languages, enhancing accessibility across different locales.

---

## 🚀 Features

- **NLP-Powered Responses:** Uses a trained ML model (scikit-learn + NLTK) to understand user intent and provide responses.
- **Multilingual Support:** Translates user messages to English for classification and then back to the original language for response.
- **Web Chat Interface:** Clean and interactive web interface with a chatbox and language selector.
- **Voice Interaction:** Speech-to-text input and text-to-speech output (browser-based).
- **Quick Replies:** Allows easier and faster interaction through UI enhancements.

---

## 🛠 Tech Stack

- **Backend:** Python 3, Flask
- **NLP Libraries:** Scikit-learn, NLTK, Googletrans
- **Frontend:** HTML, CSS, JavaScript
- **Others:** Pickle (for saving/loading ML models), CountVectorizer, JSON

---

## 🔧 Installation Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Gurureddy22/chatbot.git
   cd chatbot
