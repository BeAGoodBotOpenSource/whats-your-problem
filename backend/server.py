import io
import logging

import openai
import PyPDF2
from flask import Flask, jsonify, request
from flask_cors import CORS
from PyPDF2 import PdfReader

from config import debug_status, whitelist_origins
from debug import debug_only
#from gpt_utils import test
from whats_your_problem.text import get_further_info, get_personas, get_questions

app = Flask(__name__)
CORS(app, origins=whitelist_origins)


# Just for testing connection with backend; debugging purpose only
@app.route("/test", methods=["GET"])
@debug_only
def hello():
    return "Connected!!"


@app.route("/ask-ai", methods=["POST"])
def ask_ai():
    print("asking AI")
    data = request.get_json()
    background = data.get("background")
    problem = data.get("problem")

    personas = get_personas(background=background, problem=problem)
    questions = get_questions(background=background, personas=personas, problem=problem)

    # For now, return an empty dictionary
    return jsonify(questions), 200


@app.route("/get-further-info", methods=["POST"])
def get_further_infoo():
    data = request.get_json()
    background = data.get("background")
    problem = data.get("problem")
    persona = data.get("persona")
    question = data.get("question")

    further_info = get_further_info(
        question, persona, background, problem 
    )

    return jsonify({"further_info": further_info}), 200


if __name__ == "__main__":
    app.run(debug=debug_status)  # toggled for prod
