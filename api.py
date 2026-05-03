from flask import Flask, request, jsonify
from llm_assistant import generate_security_advice

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    prediction = "Malicious"
    state = "Alert"

    advice = generate_security_advice(
        0.92,
        state
    )

    return jsonify({
        "prediction": prediction,
        "state": state,
        "advice": advice
    })

if __name__ == "__main__":
    app.run(debug=True)