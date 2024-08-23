import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Load your dataset
data = pd.read_excel('chronic_diseases_recommendations_expanded.xlsx')

# Define features and target
X = data[['Age', 'Weight', 'Blood Pressure', 'Gender', 'Chronic Disease', 'Pain Severity']]
y_diet = data[['Diet Morning', 'Diet Afternoon', 'Diet Night']]  # Replace with your actual target columns
y_yoga = data[['Yoga Recommendation']]  # Replace with your actual target column

# Define numerical and categorical features
numerical_features = ['Age', 'Weight', 'Pain Severity']
categorical_features = ['Blood Pressure', 'Gender', 'Chronic Disease']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Apply preprocessing
X_transformed = preprocessor.fit_transform(X)

# Save the preprocessor
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

# Proceed with training your models here
# Example:
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier()
# model.fit(X_transformed, y_diet)
# Save your models using pickle
# with open('diet_model_morning.pkl', 'wb') as f:
#     pickle.dump(model, f)
