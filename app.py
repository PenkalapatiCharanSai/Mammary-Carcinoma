# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from flask import Flask, render_template, request
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import os

# app = Flask(__name__)

# # Ensure directories exist
# os.makedirs('static/images', exist_ok=True)

# # Load and preprocess data
# file_path = r'C:\Users\Chara\OneDrive\Desktop\AD\dataset\breast-cancer.csv'  # Your file path
# data_frame = pd.read_csv(file_path)

# # Preprocessing function
# def preprocess_data(df):
#     # Step 1: Remove unnecessary columns
#     if 'id' in df.columns:
#         df = df.drop(columns=['id'])
    
#     # Step 2: Handle missing values (impute with mean for numerical columns)
#     numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
#     df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
#     # Step 3: Encode target variable (diagnosis: M=0, B=1)
#     df['diagnosis'] = df['diagnosis'].map({'M': 0, 'B': 1})
    
#     # Step 4: Remove duplicates (if any)
#     df = df.drop_duplicates()
    
#     return df

# # Apply preprocessing
# data_frame = preprocess_data(data_frame)

# # Selected features for prediction
# selected_features = ['radius_mean', 'texture_mean', 'smoothness_mean']
# X = data_frame[selected_features]
# y = data_frame['diagnosis']

# # Step 5: Scale the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Step 6: Split the dataset
# X_train, X_test, y_train, y_test = train_test_split(
#     X_scaled, y, test_size=0.2, random_state=42, stratify=y
# )

# # Train models
# models = {
#     'Logistic Regression': LogisticRegression(max_iter=1000).fit(X_train, y_train),
#     'SVM': SVC(probability=True).fit(X_train, y_train),
#     'Decision Tree': DecisionTreeClassifier().fit(X_train, y_train),
#     'Random Forest': RandomForestClassifier().fit(X_train, y_train)
# }

# # Generate plots
# def generate_plots():
#     # Correlation Heatmap (numeric columns only)
#     numeric_df = data_frame.select_dtypes(include=['float64', 'int64'])
#     plt.figure(figsize=(12, 10))
#     sns.heatmap(numeric_df.corr(), cmap='coolwarm', annot=False)
#     plt.title("Feature Correlation Heatmap")
#     plt.savefig('static/images/correlation_heatmap.png')
#     plt.close()

#     # Diagnosis Distribution
#     plt.figure()
#     sns.countplot(x=data_frame['diagnosis'])
#     plt.title("Distribution of Diagnosis")
#     plt.xlabel("Diagnosis (0: Malignant, 1: Benign)")
#     plt.ylabel("Count")
#     plt.savefig('static/images/distribution.png')
#     plt.close()

#     # Feature Importance (Random Forest, only selected features)
#     feature_importances = models['Random Forest'].feature_importances_
#     feature_importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': feature_importances})
#     plt.figure(figsize=(8, 5))
#     sns.barplot(x=feature_importance_df['Importance'], y=feature_importance_df['Feature'])
#     plt.title("Feature Importance (Random Forest)")
#     plt.savefig('static/images/feature_importance.png')
#     plt.close()

#     # Boxplot of Selected Features (using original data for interpretability)
#     plt.figure(figsize=(12, 6))
#     sns.boxplot(data=data_frame[selected_features])
#     plt.xticks(rotation=45)
#     plt.title("Boxplot of Selected Features")
#     plt.savefig('static/images/boxplot.png')
#     plt.close()

#     # Violin Plot of Selected Features
#     plt.figure(figsize=(12, 6))
#     sns.violinplot(data=data_frame[selected_features])
#     plt.xticks(rotation=45)
#     plt.title("Violin Plot of Features")
#     plt.savefig('static/images/violinplot.png')
#     plt.close()

#     # Pairplot of Selected Features
#     sns.pairplot(data_frame, vars=selected_features, hue='diagnosis')
#     plt.savefig('static/images/pairplot.png')
#     plt.close()

#     # Histograms for ALL Numeric Columns
#     numeric_columns = data_frame.select_dtypes(include=['float64', 'int64']).columns.drop('diagnosis', errors='ignore')
#     num_plots = len(numeric_columns)
#     cols = 3
#     rows = (num_plots + cols - 1) // cols
#     plt.figure(figsize=(15, rows * 3))
#     for i, column in enumerate(numeric_columns, 1):
#         plt.subplot(rows, cols, i)
#         sns.histplot(data_frame[column], bins=20, kde=True)
#         plt.title(f'Histogram of {column}', fontsize=10)
#         plt.xlabel(column, fontsize=8)
#         plt.ylabel('Frequency', fontsize=8)
#         plt.xticks(fontsize=6)
#         plt.yticks(fontsize=6)
#     plt.tight_layout()
#     plt.savefig('static/images/histogram_all.png')
#     plt.close()

#     # KDE Plot of Selected Features
#     plt.figure(figsize=(12, 6))
#     sns.kdeplot(data_frame['radius_mean'], label='Radius Mean', fill=True)
#     sns.kdeplot(data_frame['texture_mean'], label='Texture Mean', fill=True)
#     sns.kdeplot(data_frame['smoothness_mean'], label='Smoothness Mean', fill=True)
#     plt.legend()
#     plt.title("Kernel Density Estimate of Selected Features")
#     plt.savefig('static/images/kdeplot.png')
#     plt.close()

# # Prediction function (updated to scale input data)
# def make_prediction(input_data):
#     # Scale the input data using the same scaler used for training
#     input_data = scaler.transform(input_data.reshape(-1, 3))  # Reshape and scale
#     predictions = {}
#     probabilities = {}
#     for name, model in models.items():
#         class_predictions = model.predict(input_data)
#         class_probs = model.predict_proba(input_data)[:, 1]  # Probability of Benign
#         predictions[name] = ['Benign' if p == 1 else 'Malignant' for p in class_predictions]
#         probabilities[name] = class_probs
#     final_predictions = ['Benign' if round(np.mean([1 if p == 'Benign' else 0 for p in preds])) == 1 else 'Malignant'
#                          for preds in zip(*predictions.values())]
#     final_confidences = [np.mean([probs[i] for probs in probabilities.values()]) 
#                          for i in range(len(final_predictions))]
#     return predictions, probabilities, final_predictions, final_confidences

# # Routes
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/analysis')
# def analysis():
#     return render_template('analysis.html')

# @app.route('/insights')
# def insights():
#     stats = data_frame.describe().to_html(classes=['table', 'table-striped'])
#     diagnosis_dist = data_frame['diagnosis'].value_counts().to_dict()
#     return render_template('insights.html', stats=stats, label_dist=diagnosis_dist)

# @app.route('/prediction', methods=['GET', 'POST'])
# def prediction():
#     form_data = {'radius_mean': '', 'texture_mean': '', 'smoothness_mean': ''}
#     predictions = probabilities = final_pred = confidence = error = None
    
#     if request.method == 'POST':
#         # Debug: Print the incoming form data
#         print("Form data received:", request.form)
        
#         # Check if all required keys are present
#         required_fields = ['radius_mean', 'texture_mean', 'smoothness_mean']
#         if not all(field in request.form for field in required_fields):
#             error = "Missing one or more required fields: radius_mean, texture_mean, smoothness_mean"
#         else:
#             try:
#                 form_data['radius_mean'] = request.form['radius_mean']
#                 form_data['texture_mean'] = request.form['texture_mean']
#                 form_data['smoothness_mean'] = request.form['smoothness_mean']
                
#                 input_data = np.array([[float(form_data['radius_mean']),
#                                         float(form_data['texture_mean']),
#                                         float(form_data['smoothness_mean'])]])
#                 predictions, probabilities, final_pred, confidence = make_prediction(input_data)
#                 final_pred = final_pred[0]  # Single prediction
#                 confidence = confidence[0]  # Single confidence
#             except ValueError:
#                 error = "Please enter valid numerical values"
    
#     return render_template('prediction.html', form_data=form_data, predictions=predictions,
#                          probabilities=probabilities, final_pred=final_pred, confidence=confidence,
#                          error=error)

# if __name__ == '__main__':
#     generate_plots()
#     app.run(debug=True)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)

# Ensure directories exist
os.makedirs('static/images', exist_ok=True)

# Load and preprocess data
file_path = r'C:\Users\Chara\OneDrive\Desktop\AD\dataset\breast-cancer.csv'
data_frame = pd.read_csv(file_path)

# Preprocessing function
def preprocess_data(df):
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    df['diagnosis'] = df['diagnosis'].map({'M': 0, 'B': 1})
    df = df.drop_duplicates()
    return df

# Apply preprocessing
data_frame = preprocess_data(data_frame)

# Use all features except the target for training
all_features = data_frame.drop(columns=['diagnosis']).columns.tolist()  # 30 features
X = data_frame[all_features]
y = data_frame['diagnosis']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train models on all features
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000).fit(X_train, y_train),
    'SVM': SVC(probability=True).fit(X_train, y_train),
    'Decision Tree': DecisionTreeClassifier().fit(X_train, y_train),
    'Random Forest': RandomForestClassifier().fit(X_train, y_train)
}

# Selected features for manual input
selected_features = ['radius_mean', 'texture_mean', 'smoothness_mean']

# Calculate mean values for all features (to fill in missing ones during prediction)
feature_means = data_frame[all_features].mean().to_dict()

# Generate plots
def generate_plots():
    numeric_df = data_frame.select_dtypes(include=['float64', 'int64'])
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), cmap='coolwarm', annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.savefig('static/images/correlation_heatmap.png')
    plt.close()

    plt.figure()
    sns.countplot(x=data_frame['diagnosis'])
    plt.title("Distribution of Diagnosis")
    plt.xlabel("Diagnosis (0: Malignant, 1: Benign)")
    plt.ylabel("Count")
    plt.savefig('static/images/distribution.png')
    plt.close()

    feature_importances = models['Random Forest'].feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': all_features, 'Importance': feature_importances})
    plt.figure(figsize=(10, 8))
    sns.barplot(x=feature_importance_df['Importance'], y=feature_importance_df['Feature'])
    plt.title("Feature Importance (Random Forest)")
    plt.savefig('static/images/feature_importance.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data_frame[selected_features])  # Still showing selected features for simplicity
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

    sns.pairplot(data_frame, vars=selected_features, hue='diagnosis')
    plt.savefig('static/images/pairplot.png')
    plt.close()

    numeric_columns = data_frame.select_dtypes(include=['float64', 'int64']).columns.drop('diagnosis', errors='ignore')
    num_plots = len(numeric_columns)
    cols = 3
    rows = (num_plots + cols - 1) // cols
    plt.figure(figsize=(15, rows * 3))
    for i, column in enumerate(numeric_columns, 1):
        plt.subplot(rows, cols, i)
        sns.histplot(data_frame[column], bins=20, kde=True)
        plt.title(f'Histogram of {column}', fontsize=10)
        plt.xlabel(column, fontsize=8)
        plt.ylabel('Frequency', fontsize=8)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.savefig('static/images/histogram_all.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.kdeplot(data_frame['radius_mean'], label='Radius Mean', fill=True)
    sns.kdeplot(data_frame['texture_mean'], label='Texture Mean', fill=True)
    sns.kdeplot(data_frame['smoothness_mean'], label='Smoothness Mean', fill=True)
    plt.legend()
    plt.title("Kernel Density Estimate of Selected Features")
    plt.savefig('static/images/kdeplot.png')
    plt.close()

# Prediction function
def make_prediction(input_data_partial):
    # Create a full input array with default means
    full_input = np.array([feature_means[feat] for feat in all_features]).reshape(1, -1)
    
    # Update with user-provided values
    for i, feat in enumerate(selected_features):
        full_input[0, all_features.index(feat)] = input_data_partial[i]
    
    # Scale the full input
    input_data_scaled = scaler.transform(full_input)
    
    predictions = {}
    probabilities = {}
    for name, model in models.items():
        class_predictions = model.predict(input_data_scaled)
        class_probs = model.predict_proba(input_data_scaled)[:, 1]
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
    diagnosis_dist = data_frame['diagnosis'].value_counts().to_dict()
    return render_template('insights.html', stats=stats, label_dist=diagnosis_dist)

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    form_data = {'radius_mean': '', 'texture_mean': '', 'smoothness_mean': ''}
    predictions = probabilities = final_pred = confidence = error = None
    
    if request.method == 'POST':
        print("Form data received:", request.form)
        required_fields = ['radius_mean', 'texture_mean', 'smoothness_mean']
        if not all(field in request.form for field in required_fields):
            error = "Missing one or more required fields: radius_mean, texture_mean, smoothness_mean"
        else:
            try:
                form_data['radius_mean'] = request.form['radius_mean']
                form_data['texture_mean'] = request.form['texture_mean']
                form_data['smoothness_mean'] = request.form['smoothness_mean']
                
                input_data_partial = np.array([float(form_data['radius_mean']),
                                               float(form_data['texture_mean']),
                                               float(form_data['smoothness_mean'])])
                
                predictions, probabilities, final_pred, confidence = make_prediction(input_data_partial)
                final_pred = final_pred[0]
                confidence = confidence[0]
            except ValueError:
                error = "Please enter valid numerical values"
    
    return render_template('prediction.html', form_data=form_data, predictions=predictions,
                         probabilities=probabilities, final_pred=final_pred, confidence=confidence,
                         error=error)

if __name__ == '__main__':
    generate_plots()
    app.run(debug=True)