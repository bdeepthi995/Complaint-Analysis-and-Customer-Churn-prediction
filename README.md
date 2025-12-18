# Complaint Classification and Customer Churn Prediction

This project is a web-based application that analyzes customer complaints from CSV files, classifies them into departments using NLP techniques, and identifies potential customer churn risks based on complaint patterns.

## Features
- Upload customer complaint data in CSV format
- Automatically classify complaints into departments
- Generate department-wise complaint files
- Visualize complaint trends using charts
- Identify churn risk reasons based on complaint volume
- Web interface built using Flask

## Technologies Used
- Python
- Flask
- Pandas
- Scikit-learn
- Matplotlib
- Natural Language Processing (TF-IDF)

## Project Workflow
1. User uploads a CSV file containing customer complaints
2. Complaints are classified into departments using keyword-based NLP logic
3. Department-wise complaint data is saved automatically
4. Visualizations are generated for:
   - Complaints per department
   - Complaints over time
5. Churn risk reasons are predicted based on complaint trends

## Expected CSV Columns
The uploaded CSV file should contain at least the following columns:
- `complaint`
- `date_of_complaint`

## How to Run the Project
1. Clone or download the repository
2. Install required dependencies:
