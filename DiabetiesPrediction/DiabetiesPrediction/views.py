
from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

def index(request):
    return render(request, 'index.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    diabetes_df = pd.read_csv(r"C:\Users\Hp\Desktop\diabetes\diabetes_prediction_dataset.csv") 
    print(diabetes_df.columns)
    
    # Convert categorical features to numerical
    categorical_features = ['gender', 'smoking_history']
    diabetes_df = pd.get_dummies(diabetes_df, columns=categorical_features)
    
    # Define the features (X) and target (y)
    X = diabetes_df.drop(columns='diabetes')  # Assuming 'diabetes' is the target column
    y = diabetes_df['diabetes']
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)

    # Train the SVC model
    svc_model = SVC()
    svc_model.fit(X_train, y_train)

    # Get values from the request
    val1 = request.GET.get('n1', '')  # gender (string)
    val2 = float(request.GET.get('n2', 0))  # age
    val3 = float(request.GET.get('n3', 0))  # hypertension
    val4 = float(request.GET.get('n4', 0))  # heart_disease
    val5 = request.GET.get('n5', '')  # smoking_history (string)
    val6 = float(request.GET.get('n6', 0))  # bmi
    val7 = float(request.GET.get('n7', 0))  # HbA1c_level
    val8 = float(request.GET.get('n8', 0))  # blood_glucose_level

    # Convert categorical values to numerical using one-hot encoding
    input_data = pd.DataFrame({
        'gender': [val1],
        'age': [val2],
        'hypertension': [val3],
        'heart_disease': [val4],
        'smoking_history': [val5],
        'bmi': [val6],
        'HbA1c_level': [val7],
        'blood_glucose_level': [val8]
    })

    # Convert to one-hot encoding
    input_data = pd.get_dummies(input_data, columns=categorical_features)
    
    # Ensure the input_data has the same columns as the training data
    input_data = input_data.reindex(columns=X_train.columns, fill_value=0)
    
    # Convert boolean values to numeric (True -> 1, False -> 0)
    input_data = input_data.astype(int)

    # Debugging: Print the aligned input data
    print("Aligned Input Data:")
    print(input_data)

    # Make prediction
    pred = svc_model.predict(input_data)

    # Interpret prediction result
    result1 = "Positive" if pred == [1] else "Negative"

    # Debugging: Print the result
    print("Prediction Result: ", result1)
    # print()

    # Render the result
    return render(request, 'predict.html', {"result2": result1})


