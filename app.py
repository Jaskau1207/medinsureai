from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
model = joblib.load("medical_insurance.joblib")
model2 = joblib.load("health_score.joblib")
diabetes_model = joblib.load("diabetes.joblib")
hypertension_model = joblib.load("hypertension.joblib")
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/new')
def show_form():
    return render_template('new.html')

@app.route('/new', methods=['POST'])
def new_prediction():
    bmi = float(request.form['bmi'])
    smoker = request.form['smoker']
    diabetes = request.form['diabetes']
    hypertension = request.form['hypertension']
    cholesterol = request.form['cholesterol']
    allergies = request.form['allergies']
    chronic = request.form['chronic']
    sleep_quality = request.form['sleep']
    diet = request.form['diet']
    exercise = request.form['exercise']
    alcohol = request.form['alcohol']
    last_visit = request.form['visit']
    checkups = request.form['checkups']
    sleep_hours = float(request.form['sleep_hours'])
    stress = request.form['stress']
    doctor = request.form['doctor']
    insured = request.form['insured']
    budget = float(request.form['budget'])
    coverage = request.form['coverage']

    # Basic model input example — adjust based on your model’s feature order
    # (e.g., [age, bmi, smoker(0/1), diabetes(0/1), ...])
    model_input = np.array([
        bmi,
        1 if smoker == "Yes" else 0,
        1 if diabetes == "Yes" else 0,
        1 if hypertension == "Yes" else 0,
        1 if cholesterol == "High" else 0
    ]).reshape(1, -1)

    # Predict base insurance cost
    base_cost = model.predict(model_input)[0]

    # Add custom logic for extra costs
    extra_cost = 0

    # Penalize “Not Sure” health answers
    if diabetes == "Yes":
        extra_cost += 500
    if hypertension == "Yes":
        extra_cost += 500
    if cholesterol == "Yes":
        extra_cost += 300

    # Lifestyle factors
    if sleep_quality == "Poor":
        extra_cost += 400
    elif sleep_quality == "Average":
        extra_cost += 200

    if diet == "Junk-heavy":
        extra_cost += 600
    elif diet == "Non-Veg":
        extra_cost += 200

    if exercise == "Rarely":
        extra_cost += 300
    elif exercise == "Sometimes":
        extra_cost += 150

    if alcohol == "Frequently":
        extra_cost += 500
    elif alcohol == "Occasionally":
        extra_cost += 200

    if stress == "High":
        extra_cost += 400
    elif stress == "Moderate":
        extra_cost += 200
    if checkups == "No":
        extra_cost += 300
    if insured == "No":
        extra_cost += 500
    coverage_cost_map = {
        "Individual": 0,
        "Family": 1000,
        "Senior": 1500,
        "Critical Illness": 2000
    }
    extra_cost += coverage_cost_map.get(coverage, 0)
    total_cost = base_cost + extra_cost
    return {
    "total": round(total_cost, 2)
}


@app.route('/tests')
def tests():
    return render_template('tests.html')

@app.route('/diabetes')
def diabetes_page():
    return render_template('diabetes.html')

@app.route('/hypertension')
def hypertension_page():
    return render_template('hypertension.html')
@app.route('/diabetes', methods=['POST'])
def predict_diabetes():
    try:
        gender= float(request.form['gender'])
        smoking_history = float(request.form['smoking_history'])
        hypertension = float(request.form['hypertension'])
        heart_disease = float(request.form['heart_disease'])
        HbA1c_level = float(request.form['HbA1c_level'])
        bmi = float(request.form['bmi'])
        blood_glucose_level = float(request.form['blood_glucose_level'])
        age = float(request.form['age'])

        input_data = np.array([[gender , age, smoking_history,blood_glucose_level , bmi , heart_disease,hypertension, HbA1c_level]])
        prediction = diabetes_model.predict(input_data)[0]
        result = "Positive" if prediction == 1 else "Negative"

        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)})
@app.route('/hypertension', methods=['POST'])
def predict_hypertension():
    try:
        age = float(request.form['age'])
        bmi = float(request.form['bmi'])
        bp_history = float(request.form['bp_history'])
        salt_intake = float(request.form['salt_intake'])
        smoker = float(request.form['smoking_status'])
        medication = float(request.form['medication'])
        sleep_duration = float(request.form['sleep_duration'])
        family_history = float(request.form['family_history'])
        stress = float(request.form['stress_score'])
        exercise_level = float(request.form['exercise_level'])

        input_data = np.array([[age, bmi, bp_history , exercise_level , family_history , smoker, stress, salt_intake, sleep_duration]])
        prediction = hypertension_model.predict(input_data)[0]

        hypertension_result = "High Risk" if prediction == "Yes" else "Low Risk"
        return jsonify({"result": hypertension_result})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/score')
def score():
    return render_template('score.html')

@app.route('/score', methods=['POST'] )
def pred_score():
    age = float(request.form['age'])
    bmi = float(request.form['bmi'])
    smoker = float(request.form['smoking_status'])
    diet = float(request.form['diet_quality'])
    exercise = float(request.form['exercise_freq'])
    alcohol = float(request.form['alcohol_consumption'])
    sleep_hours = float(request.form['sleep_hours'])

    model_input2 = np.array([
        bmi, age, smoker, diet, exercise, alcohol, sleep_hours
    ]).reshape(1, -1)

    score = model2.predict(model_input2)[0]

    return jsonify({
        "health_score": round(float(score), 2)
    })

@app.route('/planner')
def planner():
    return render_template('planner.html')

@app.route('/plans')
def plans():
    return render_template('plans.html')

if __name__ == '__main__':
    app.run(debug=True)
