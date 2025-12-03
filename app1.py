import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, url_for, send_from_directory
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import os

app = Flask(__name__)

# Ensure directories exist
os.makedirs('static/images', exist_ok=True)

# Load and prepare data
breast_cancer_dataset = load_breast_cancer()
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
data_frame['label'] = breast_cancer_dataset.target
selected_features = ['mean radius', 'mean texture', 'mean smoothness']
X = data_frame[selected_features]
Y = data_frame['label']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000).fit(X_train, Y_train),
    'SVM': SVC(probability=True).fit(X_train, Y_train),
    'Decision Tree': DecisionTreeClassifier().fit(X_train, Y_train),
    'Random Forest': RandomForestClassifier().fit(X_train, Y_train)
}

# Generate plots
def generate_plots():
    plt.figure(figsize=(12, 10))
    sns.heatmap(data_frame.corr(), cmap='coolwarm', annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.savefig('static/images/correlation_heatmap.png')
    plt.close()

    plt.figure()
    sns.countplot(x=data_frame['label'])
    plt.title("Distribution of Breast Cancer Labels")
    plt.xlabel("Label (0: Malignant, 1: Benign)")
    plt.ylabel("Count")
    plt.savefig('static/images/distribution.png')
    plt.close()

    feature_importances = models['Random Forest'].feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': feature_importances})
    plt.figure(figsize=(8, 5))
    sns.barplot(x=feature_importance_df['Importance'], y=feature_importance_df['Feature'])
    plt.title("Feature Importance (Random Forest)")
    plt.savefig('static/images/feature_importance.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data_frame[selected_features])
    plt.xticks(rotation=45)
    plt.title("Boxplot of Selected Features")
    plt.savefig('static/images/boxplot.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.violinplot(data=data_frame[selected_features])
    plt.xticks(rotation=45)
    plt.title("Violin Plot of Features")
    plt.savefig('static/images/violinplot.png')
    plt.close()

    sns.pairplot(data_frame, vars=selected_features, hue='label')
    plt.savefig('static/images/pairplot.png')
    plt.close()

    data_frame[selected_features].hist(figsize=(15, 10), bins=20)
    plt.suptitle("Feature Distributions")
    plt.savefig('static/images/histogram.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.kdeplot(data_frame['mean radius'], label='Mean Radius', fill=True)
    sns.kdeplot(data_frame['mean texture'], label='Mean Texture', fill=True)
    sns.kdeplot(data_frame['mean smoothness'], label='Mean Smoothness', fill=True)
    plt.legend()
    plt.title("Kernel Density Estimate of Selected Features")
    plt.savefig('static/images/kdeplot.png')
    plt.close()

# Prediction function
def make_prediction(input_data):
    input_data = input_data.reshape(-1, 3)  # Handle multiple rows
    predictions = {}
    probabilities = {}
    for name, model in models.items():
        class_predictions = model.predict(input_data)
        class_probs = model.predict_proba(input_data)[:, 1]  # Probability of Benign
        predictions[name] = ['Benign' if p == 1 else 'Malignant' for p in class_predictions]
        probabilities[name] = class_probs
    final_predictions = ['Benign' if round(np.mean([1 if p == 'Benign' else 0 for p in preds])) == 1 else 'Malignant'
                         for preds in zip(*predictions.values())]
    final_confidences = [np.mean([probs[i] for probs in probabilities.values()]) 
                         for i in range(len(final_predictions))]
    return predictions, probabilities, final_predictions, final_confidences

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/insights')
def insights():
    stats = data_frame.describe().to_html(classes=['table', 'table-striped'])
    label_dist = data_frame['label'].value_counts().to_dict()
    return render_template('insights.html', stats=stats, label_dist=label_dist)

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    form_data = {'mean_radius': '', 'mean_texture': '', 'mean_smoothness': ''}
    predictions = probabilities = final_pred = confidence = error = None
    
    if request.method == 'POST':
        try:
            form_data['mean_radius'] = request.form['mean_radius']
            form_data['mean_texture'] = request.form['mean_texture']
            form_data['mean_smoothness'] = request.form['mean_smoothness']
            
            input_data = np.array([[float(form_data['mean_radius']),
                                    float(form_data['mean_texture']),
                                    float(form_data['mean_smoothness'])]])
            predictions, probabilities, final_pred, confidence = make_prediction(input_data)
            final_pred = final_pred[0]  # Single prediction
            confidence = confidence[0]  # Single confidence
        except ValueError:
            error = "Please enter valid numerical values"
    
    return render_template('prediction.html', form_data=form_data, predictions=predictions,
                         probabilities=probabilities, final_pred=final_pred, confidence=confidence,
                         error=error)

if __name__ == '__main__':
    generate_plots()
    app.run(debug=True)
