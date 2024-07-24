# from django.shortcuts import render

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set()

# from mlxtend.plotting import plot_decision_regions
# # import missingno as msno
# from pandas.plotting import scatter_matrix
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import confusion_matrix
# from sklearn import metrics
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report
# import warnings
# warnings.filterwarnings('ignore')
# # %matplotlib inline

# def index(request):
#     return render(request, 'index.html')

# def predict(request):
#     return render(request, 'predict.html')

# def result(request):
#     diabetes_df = pd.read_csv(r"C:\Users\Hp\Desktop\diabetes\diabetes_prediction_dataset.csv") 
#     # from sklearn.model_selection import train_test_split
#       # Define the features (X) and target (y)
#     # print(diabetes_df.columns)

#         # Encode categorical data
#     le = LabelEncoder()
#     diabetes_df['gender'] = le.fit_transform(diabetes_df['gender'])
#     diabetes_df['smoking_history'] = le.fit_transform(diabetes_df['smoking_history'])

#     X = diabetes_df.drop(columns='diabetes')  # Assuming 'Outcome' is the target column
#     y = diabetes_df['diabetes']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)

#     svc_model = SVC()
#     svc_model.fit(X_train, y_train)


#     gender = request.GET['n1']
#     if gender.lower() == 'male':
#         val1 = 1  # Example: Male -> 1
#     else:
#         val1 = 0  # Example: Female -> 0

#     val1 = float(request.GET['n1'])
#     val2 = float(request.GET['n2'])
#     val3 = float(request.GET['n3'])
#     val4 = float(request.GET['n4'])
#     val5 = float(request.GET['n5'])
#     val6 = float(request.GET['n6'])
#     val7 = float(request.GET['n7'])
#     val8 = float(request.GET['n8'])

#     pred = svc_model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]]) 

#     result1 = ""
#     if pred == [1]:
#         result1 = "Positive"
#     else:
#         result1 = "Negative"
#     # val1 = float(request.GET['n1'])
#     return render(request, 'predict.html', {"result12" : result1})
