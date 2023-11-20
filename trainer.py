
import logging
from data_loader import prepare_data
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, log_loss
import pandas as pd
import time
import sys
import psutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(filename='model_evaluation.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


(X_train, X_test, y_train, y_test), encoder = prepare_data("data/twitter_news.csv", 'engagement_rate', 'text', 'engagement_level', 5, 0.25, 42)

models = [
    {"name": "Logistic Regression", "model": LogisticRegression(multi_class='multinomial', solver='lbfgs')},
    {"name": "Decision Tree", "model": DecisionTreeClassifier()},
    {"name": "Random Forest", "model": RandomForestClassifier()},
    {"name": "Support Vector Machine", "model": OneVsRestClassifier(SVC(decision_function_shape='ovo', probability=True))},# Wrap SVC in OneVsRestClassifier for multi-class classification
    {"name": "K-Nearest Neighbors", "model": KNeighborsClassifier()},
    {"name": "Naive Bayes", "model": GaussianNB()},
    {"name": "Multi-Layer Perceptron", "model": MLPClassifier()},
    {"name": "XGBoost", "model": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', objective='multi:softmax')},
    {"name": "LightGBM", "model": LGBMClassifier()},
    {"name": "CatBoost", "model": CatBoostClassifier(verbose=0, loss_function='MultiClass')}
]



metrics = []

for model in models:
    try:
        # Training
        start_time = time.time()
        model['model'].fit(X_train, y_train)
        train_time = time.time() - start_time

        # Prediction
        start_time = time.time()
        y_pred = model['model'].predict(X_test)
        predict_time = time.time() - start_time

        # Calculate probabilities for log loss
        y_prob = model['model'].predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=1)  # Set zero_division to 1 or 'warn'
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        logloss = log_loss(y_test, y_prob)

        # Resource usage
        model_size = sys.getsizeof(model['model'])
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent

        metrics.append({
            'model': model['name'],
            'train_time': train_time,
            'predict_time': predict_time,
            'model_size': model_size,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'log_loss': logloss
        })

        # Log success message
        logging.info(f"Model '{model['name']}' evaluation successful.")

    except Exception as e:
        # Log error message
        logging.error(f"Error evaluating model '{model['name']}': {str(e)}")


# Convert list of dicts to DataFrame
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv('output/metrics.csv', index=False)


# Create subplots: 2 rows, 5 cols
fig = make_subplots(rows=2, cols=5, subplot_titles=('train_time', 'predict_time', 'model_size', 'cpu_usage', 'memory_usage', 'accuracy', 'precision', 'recall', 'f1', 'log_loss'))

metrics_to_plot = ['train_time', 'predict_time', 'model_size', 'cpu_usage', 'memory_usage', 'accuracy', 'precision', 'recall', 'f1', 'log_loss']

for i, metric in enumerate(metrics_to_plot):
    row = 1 if i < 5 else 2
    col = i % 5 + 1
    fig.add_trace(go.Bar(x=metrics_df['model'], y=metrics_df[metric], name=metric), row=row, col=col)

fig.update_layout(height=900, width=1600, title_text="Model Comparison", showlegend=False)
fig.show()
