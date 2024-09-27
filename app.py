from flask import Flask, request, render_template
from wtforms import Form, FloatField, SelectField
from wtforms.validators import InputRequired, NumberRange
import pandas as pd
import pickle

app = Flask(__name__)

# Load the model and scaler
with open('heart_disease_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# PredictionForm class 
class PredictionForm(Form):
    age = FloatField('Age', validators=[InputRequired(), NumberRange(min=0, max=120)])
    sex = SelectField('Sex', choices=[('0', 'Female'), ('1', 'Male')], validators=[InputRequired()])
    cp = SelectField('Chest Pain Type', choices=[('0', 'Typical Angina'), ('1', 'Atypical Angina'), ('2', 'Non-anginal Pain'), ('3', 'Asymptomatic')], validators=[InputRequired()])
    trestbps = FloatField('Resting Blood Pressure', validators=[InputRequired(), NumberRange(min=0, max=300)])
    chol = FloatField('Serum Cholesterol', validators=[InputRequired(), NumberRange(min=0, max=600)])
    fbs = SelectField('Fasting Blood Sugar > 120 mg/dl', choices=[('0', 'No'), ('1', 'Yes')], validators=[InputRequired()])
    restecg = SelectField('Resting ECG Results', choices=[('0', 'Normal'), ('1', 'ST-T Wave Abnormality'), ('2', 'Left Ventricular Hypertrophy')], validators=[InputRequired()])
    thalach = FloatField('Maximum Heart Rate Achieved', validators=[InputRequired(), NumberRange(min=0, max=300)])
    exang = SelectField('Exercise Induced Angina', choices=[('0', 'No'), ('1', 'Yes')], validators=[InputRequired()])
    oldpeak = FloatField('ST Depression', validators=[InputRequired(), NumberRange(min=0, max=10)])
    slope = SelectField('Slope of Peak Exercise ST Segment', choices=[('0', 'Upsloping'), ('1', 'Flat'), ('2', 'Downsloping')], validators=[InputRequired()])
    ca = SelectField('Number of Major Vessels Colored by Fluoroscopy', choices=[('0', '0'), ('1', '1'), ('2', '2'), ('3', '3')], validators=[InputRequired()])
    thal = SelectField('Thalassemia', choices=[('0', 'Normal'), ('1', 'Fixed Defect'), ('2', 'Reversible Defect')], validators=[InputRequired()])

@app.route('/')
def home():
    form = PredictionForm(request.form)
    return render_template('index.html', form=form)

@app.route('/predict', methods=['POST'])
def predict():
    form = PredictionForm(request.form)
    if form.validate():
        try:
            input_data = {field.name: float(field.data) for field in form}
            df = pd.DataFrame([input_data])
            scaled_features = scaler.transform(df)
            prediction = model.predict(scaled_features)
            probability = model.predict_proba(scaled_features)[0][1]
            
            result = "High risk of heart disease" if prediction[0] == 1 else "Low risk of heart disease"
            probability_percentage = probability * 100  

            print(f"Prediction: {result}, Probability: {probability_percentage:.2f}%")  

            return render_template('result.html', 
                                   prediction=result, 
                                   probability=f"{probability_percentage:.2f}%",
                                   form=form)  
        except Exception as e:
            print(f"Error during prediction: {str(e)}")  
            return render_template('error.html', error=str(e))
    else:
        print(f"Form validation failed: {form.errors}")  
        return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
