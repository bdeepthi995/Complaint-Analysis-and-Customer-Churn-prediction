import pandas as pd
import os
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, send_from_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Ensure 'static' directory exists
if not os.path.exists('static'):
    os.makedirs('static')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    if file and file.filename.endswith('.csv'):
        filepath = os.path.join('static', file.filename)
        file.save(filepath)

        # Read CSV file
        data = pd.read_csv(filepath)

        # --- Step 1: Classify complaints ---
        department_keywords = {
            'Customer Support': ['help', 'support', 'assist', 'service', 'issue', 'call', 'query'],
            'Technical Issues': ['error', 'bug', 'problem', 'crash', 'malfunction', 'fail', 'glitch'],
            'Billing & Payment': ['charge', 'billing', 'payment', 'invoice', 'transaction', 'amount', 'pay'],
            'General Feedback': ['feedback', 'suggestion', 'idea', 'general', 'recommendation', 'comment'],
            'Others': ['miscellaneous', 'unknown', 'query', 'general', 'other']
        }

        data['Department'] = data['complaint'].apply(
            lambda x: classify_department(x, department_keywords)
        )

        # --- Step 2: Save department-wise CSVs ---
        save_department_data(data)

        # --- Step 3: Generate Visualizations ---
        generate_visualizations(data)

        # --- Step 4: Churn prediction reasons ---
        churn_reasons = churn_prediction_reasons(data)

        # --- Counts for frontend ---
        dept_counts = data['Department'].value_counts().to_dict()

        return render_template(
            'result.html',
            dept_counts=dept_counts,
            churn_reasons=churn_reasons,
            message="File uploaded and processed successfully"
        )

    return "Invalid file type. Only CSV files are allowed."


# ================================
# Helper Functions
# ================================

def classify_department(complaint, department_keywords):
    complaint = str(complaint).lower()
    for dept, keywords in department_keywords.items():
        for keyword in keywords:
            if keyword.lower() in complaint:
                return dept
    return 'Others'


def save_department_data(data):
    for department in data['Department'].unique():
        dept_data = data[data['Department'] == department]
        dept_data.to_csv(f'static/{department}_complaints.csv', index=False)


def generate_visualizations(data):
    # Complaints per department
    dept_counts = data['Department'].value_counts()

    plt.figure(figsize=(10, 6))
    dept_counts.plot(kind='bar')
    plt.title('Complaints per Department')
    plt.xlabel('Department')
    plt.ylabel('Complaint Count')
    plt.tight_layout()
    plt.savefig('static/complaints_per_department.png')
    plt.close()

    # Complaints over time
    data['date_of_complaint'] = pd.to_datetime(data['date_of_complaint'])
    complaints_over_time = data.groupby(data['date_of_complaint'].dt.to_period('M')).size()

    plt.figure(figsize=(10, 6))
    complaints_over_time.plot(kind='line')
    plt.title('Complaints Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Complaints')
    plt.tight_layout()
    plt.savefig('static/complaints_over_time.png')
    plt.close()


def churn_prediction_reasons(data):
    churn_reasons = []
    dept_counts = data['Department'].value_counts()

    for dept, count in dept_counts.items():
        if dept == 'Customer Support' and count > 5:
            churn_reasons.append(f"{dept}: High complaints indicate unresolved customer issues.")
        elif dept == 'Technical Issues' and count > 5:
            churn_reasons.append(f"{dept}: High technical complaints show product dissatisfaction.")
        elif dept == 'Billing & Payment' and count > 5:
            churn_reasons.append(f"{dept}: High billing issues affect customer trust.")
        elif dept == 'General Feedback' and count > 5:
            churn_reasons.append(f"{dept}: Excess feedback indicates general dissatisfaction.")
        else:
            churn_reasons.append(f"{dept}: No major churn risk.")

    return churn_reasons


@app.route('/static/<filename>')
def static_file(filename):
    return send_from_directory('static', filename)


if __name__ == "__main__":
    app.run(debug=True)
