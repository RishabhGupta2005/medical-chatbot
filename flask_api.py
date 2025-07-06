# flask_api.py

from flask import Flask, request, jsonify
from logic import predict_department

app = Flask(__name__)

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json
    name = data.get("name")
    age = data.get("age")
    symptom = data.get("symptom")

    # Your logic may use only 'symptom', but you can modify it to use name/age if needed
    result = predict_department(symptom)  # Or change your logic to accept all three
    return jsonify({
        "department": result["department"],
        "explanation": result["explanation"]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
