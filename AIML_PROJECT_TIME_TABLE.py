import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import tkinter as tk
from tkinter import ttk, messagebox

# ---------------------------
# Step 1: Data Preparation
# ---------------------------

# Load data from CSV file
df = pd.read_csv('student_routine_data.csv')  

# Define features (X) and target variables (y)
X = df[['Age', 'Year_of_Study', 'Academic_Hours', 'Sleep_Hours',
        'Extracurricular_Hours', 'Productivity', 'Stress']]
y_study = df['Study_Prediction']
y_sleep = df['Sleep_Prediction']
y_extracurricular = df['Extracurricular_Prediction']

# Train-test split
X_train, X_test, y_train_study, y_test_study = train_test_split(X, y_study, test_size=0.2, random_state=42)
y_train_sleep, y_test_sleep = train_test_split(y_sleep, test_size=0.2, random_state=42)
y_train_extracurricular, y_test_extracurricular = train_test_split(y_extracurricular, test_size=0.2, random_state=42)

# Train models for study, sleep, and extracurricular predictions
model_study = DecisionTreeRegressor()
model_sleep = DecisionTreeRegressor()
model_extracurricular = DecisionTreeRegressor()
model_study.fit(X_train, y_train_study)
model_sleep.fit(X_train, y_train_sleep)
model_extracurricular.fit(X_train, y_train_extracurricular)

# ---------------------------
# Step 2: GUI Application
# ---------------------------

# Initialize main window
root = tk.Tk()
root.title("Student Routine Predictor")
root.geometry("400x600")

# Labels and entry fields for input features
labels = {
    'Age': 'Age (years):',
    'Year_of_Study': 'Year of Study (1-4):',
    'Academic_Hours': 'Academic Hours:',
    'Sleep_Hours': 'Sleep Hours:',
    'Extracurricular_Hours': 'Extracurricular Hours:',
    'Productivity': 'Productivity (1-10):',
    'Stress': 'Stress Level (1-10):'
}

# Dictionary to hold entry widgets
entries = {}
for key, label_text in labels.items():
    label = ttk.Label(root, text=label_text)
    label.pack(pady=5)
    entry = ttk.Entry(root)
    entry.pack(pady=5)
    entries[key] = entry

# Function to make predictions based on user input
def predict_routine():
    try:
        # Retrieve and format input data from user
        input_data = [
            int(entries['Age'].get()),
            int(entries['Year_of_Study'].get()),
            float(entries['Academic_Hours'].get()),
            float(entries['Sleep_Hours'].get()),
            float(entries['Extracurricular_Hours'].get()),
            int(entries['Productivity'].get()),
            int(entries['Stress'].get())
        ]

        # Convert input data into a DataFrame for prediction
        input_df = pd.DataFrame([input_data], columns=X.columns)

        # Predict study, sleep, and extracurricular hours
        study_hours = model_study.predict(input_df)[0]
        sleep_hours = model_sleep.predict(input_df)[0]
        extracurricular_hours = model_extracurricular.predict(input_df)[0]
        free_hours = 24 - (study_hours + sleep_hours + extracurricular_hours)

        # Display prediction results
        routine_message = (f"Suggested Routine for Tomorrow:\n"
                           f"Study Hours: {study_hours:.2f}\n"
                           f"Sleep Hours: {sleep_hours:.2f}\n"
                           f"Extracurricular Hours: {extracurricular_hours:.2f}\n"
                           f"Free/Relax Time: {free_hours:.2f} hours")

        messagebox.showinfo("Routine Prediction", routine_message)

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid values for all fields.")

# Button to trigger the prediction function
predict_button = ttk.Button(root, text="Predict Next Day Routine", command=predict_routine)
predict_button.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()