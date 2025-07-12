import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='Sandeshhegde-371', repo_name='MLOPS-Tutorial-MLFlow', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/Sandeshhegde-371/MLOPS-Tutorial-MLFlow.mlflow")  # Set your MLflow tracking URI

wine= load_wine()
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

max_depth = 10
n_estimators = 50

#Mention your experiment name
mlflow.autolog()
mlflow.set_experiment("MLOps-Exp3")

with mlflow.start_run():
    rf=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    accuracy= accuracy_score(y_test, y_pred)
    
    
    #Creating a confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig("Confusion_matrix.png")
    
    #Log artifactsusing mlflow
    mlflow.log_artifact(__file__)
    
    #tags
    mlflow.set_tags({"Author":"Sandesh", "Project":"Wine Classification"})
    
    print(accuracy)