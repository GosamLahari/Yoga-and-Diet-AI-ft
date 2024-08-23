from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load models and preprocessor
with open('diet_model_morning.pkl', 'rb') as f:
    diet_model_morning = pickle.load(f)
with open('diet_model_afternoon.pkl', 'rb') as f:
    diet_model_afternoon = pickle.load(f)
with open('diet_model_night.pkl', 'rb') as f:
    diet_model_night = pickle.load(f)
with open('yoga_model.pkl', 'rb') as f:
    yoga_model = pickle.load(f)
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get form data
        name = request.form['name']
        age = int(request.form['age'])
        weight = float(request.form['weight'])
        bp = request.form['bp']
        gender = request.form['gender']
        disease = request.form['disease']
        pain_severity = request.form['pain_severity']

        # Prepare data for prediction
        input_data = pd.DataFrame([{
            'Age': age,
            'Weight': weight,
            'Blood Pressure': bp,
            'Gender': gender,
            'Chronic Disease': disease,
            'Pain Severity': pain_severity
        }])

        # Apply preprocessing
        input_data_transformed = preprocessor.transform(input_data)

        # Predict diet and yoga recommendations
        diet_morning = diet_model_morning.predict(input_data_transformed)[0]
        diet_afternoon = diet_model_afternoon.predict(input_data_transformed)[0]
        diet_night = diet_model_night.predict(input_data_transformed)[0]
        yoga_recommendation = yoga_model.predict(input_data_transformed)[0]

        return render_template('results.html', 
                               diet_morning=diet_morning, 
                               diet_afternoon=diet_afternoon, 
                               diet_night=diet_night, 
                               yoga_recommendation=yoga_recommendation)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
