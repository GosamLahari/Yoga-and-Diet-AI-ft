import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
data = pd.read_excel('chronic_diseases_recommendations_expanded.xlsx')

# Display the first few rows
print(data.head())

# Preprocess the data
X = data.drop(['Diet Recommendation (Morning)', 'Diet Recommendation (Afternoon)', 'Diet Recommendation (Night)', 
               'Yoga Recommendation (Pose 1)', 'Yoga Recommendation (Pose 2)', 
               'Yoga Recommendation (Pose 3)', 'Yoga Recommendation (Pose 4)'], axis=1)

y_diet_morning = data['Diet Recommendation (Morning)']
y_diet_afternoon = data['Diet Recommendation (Afternoon)']
y_diet_night = data['Diet Recommendation (Night)']
y_yoga_pose1 = data['Yoga Recommendation (Pose 1)']
y_yoga_pose2 = data['Yoga Recommendation (Pose 2)']
y_yoga_pose3 = data['Yoga Recommendation (Pose 3)']
y_yoga_pose4 = data['Yoga Recommendation (Pose 4)']

# Convert categorical variables to dummy variables
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_diet_morning_train, y_diet_morning_test = train_test_split(X, y_diet_morning, test_size=0.2, random_state=42)
_, _, y_diet_afternoon_train, y_diet_afternoon_test = train_test_split(X, y_diet_afternoon, test_size=0.2, random_state=42)
_, _, y_diet_night_train, y_diet_night_test = train_test_split(X, y_diet_night, test_size=0.2, random_state=42)
_, _, y_yoga_pose1_train, y_yoga_pose1_test = train_test_split(X, y_yoga_pose1, test_size=0.2, random_state=42)
_, _, y_yoga_pose2_train, y_yoga_pose2_test = train_test_split(X, y_yoga_pose2, test_size=0.2, random_state=42)
_, _, y_yoga_pose3_train, y_yoga_pose3_test = train_test_split(X, y_yoga_pose3, test_size=0.2, random_state=42)
_, _, y_yoga_pose4_train, y_yoga_pose4_test = train_test_split(X, y_yoga_pose4, test_size=0.2, random_state=42)

# Train the model for diet recommendations
diet_model_morning = RandomForestClassifier(n_estimators=100, random_state=42)
diet_model_morning.fit(X_train, y_diet_morning_train)

diet_model_afternoon = RandomForestClassifier(n_estimators=100, random_state=42)
diet_model_afternoon.fit(X_train, y_diet_afternoon_train)

diet_model_night = RandomForestClassifier(n_estimators=100, random_state=42)
diet_model_night.fit(X_train, y_diet_night_train)

# Train the model for yoga recommendations
yoga_model_pose1 = RandomForestClassifier(n_estimators=100, random_state=42)
yoga_model_pose1.fit(X_train, y_yoga_pose1_train)

yoga_model_pose2 = RandomForestClassifier(n_estimators=100, random_state=42)
yoga_model_pose2.fit(X_train, y_yoga_pose2_train)

yoga_model_pose3 = RandomForestClassifier(n_estimators=100, random_state=42)
yoga_model_pose3.fit(X_train, y_yoga_pose3_train)

yoga_model_pose4 = RandomForestClassifier(n_estimators=100, random_state=42)
yoga_model_pose4.fit(X_train, y_yoga_pose4_train)

# Evaluate the models
diet_morning_accuracy = accuracy_score(y_diet_morning_test, diet_model_morning.predict(X_test))
diet_afternoon_accuracy = accuracy_score(y_diet_afternoon_test, diet_model_afternoon.predict(X_test))
diet_night_accuracy = accuracy_score(y_diet_night_test, diet_model_night.predict(X_test))

yoga_pose1_accuracy = accuracy_score(y_yoga_pose1_test, yoga_model_pose1.predict(X_test))
yoga_pose2_accuracy = accuracy_score(y_yoga_pose2_test, yoga_model_pose2.predict(X_test))
yoga_pose3_accuracy = accuracy_score(y_yoga_pose3_test, yoga_model_pose3.predict(X_test))
yoga_pose4_accuracy = accuracy_score(y_yoga_pose4_test, yoga_model_pose4.predict(X_test))

print(f"Diet Model Morning Accuracy: {diet_morning_accuracy}")
print(f"Diet Model Afternoon Accuracy: {diet_afternoon_accuracy}")
print(f"Diet Model Night Accuracy: {diet_night_accuracy}")
print(f"Yoga Model Pose 1 Accuracy: {yoga_pose1_accuracy}")
print(f"Yoga Model Pose 2 Accuracy: {yoga_pose2_accuracy}")
print(f"Yoga Model Pose 3 Accuracy: {yoga_pose3_accuracy}")
print(f"Yoga Model Pose 4 Accuracy: {yoga_pose4_accuracy}")

# Save the models
joblib.dump(diet_model_morning, 'diet_model_morning.pkl')
joblib.dump(diet_model_afternoon, 'diet_model_afternoon.pkl')
joblib.dump(diet_model_night, 'diet_model_night.pkl')
joblib.dump(yoga_model_pose1, 'yoga_model_pose1.pkl')
joblib.dump(yoga_model_pose2, 'yoga_model_pose2.pkl')
joblib.dump(yoga_model_pose3, 'yoga_model_pose3.pkl')
joblib.dump(yoga_model_pose4, 'yoga_model_pose4.pkl')
