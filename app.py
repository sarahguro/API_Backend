import joblib
import os

from flask import Flask, request, jsonify
import pandas as pd

# Use absolute path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "trained_data", "model_cls.pkl")  # updated path

# Load model
model = joblib.load(model_path)

app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        result = "Pass" if prediction == 1 else "Fail"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
