from flask import Flask, request, render_template
import joblib
import numpy as np

# Load saved model and scaler
model = joblib.load("salary_model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        rating = float(request.form["rating"])
        python_skill = int(request.form.get("python", 0))
        excel_skill = int(request.form.get("excel", 0))
        sql_skill = int(request.form.get("sql", 0))
        founded = int(request.form["founded"])

        # Tech skills score
        tech_skills = python_skill + excel_skill + sql_skill

        # Prepare input
        X_input = np.array([[rating, tech_skills, founded]])
        X_scaled = scaler.transform(X_input)

        # Prediction
        prediction = model.predict(X_scaled)[0]

        return render_template("index.html", 
                               prediction_text=f"Predicted Salary: ${prediction*1000:,.2f}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
