from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import mlflow
import pandas as pd

data= load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Create a RandomForestClassifier model
rf = RandomForestClassifier(random_state=42)

#Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30]
}

#Apply GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1,verbose=2)

# #Run without MLflow autologging
# grid_search.fit(X_train, y_train)

# #Display the best parameters and score
# print("Best Parameters:", grid_search.best_params_)
# print("Best Score:", grid_search.best_score_)


#Run with MLflow autologging

mlflow.set_experiment("Breast-cancer-Hyperparameter-Tuning")

with mlflow.start_run() as parent:
    grid_search.fit(X_train, y_train)
    
    #log all the child runs
    for i in range(len(grid_search.cv_results_['params'])):
        
        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(grid_search.cv_results_['params'][i])
            mlflow.log_metric("accuracy", grid_search.cv_results_['mean_test_score'][i])
    
    #Display the best parameters and score
    best_params= grid_search.best_params_
    best_score = grid_search.best_score_
    
    #Log the best parameters and score
    mlflow.log_params(best_params)
    mlflow.log_metric("best_accuracy", best_score)
    
    #Log training data
    train_df=X_train.copy()
    train_df['target'] = y_train
    
    train_df=mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df, "training_data")
    
    #Log testing data
    test_df=X_test.copy()
    test_df['target'] = y_test
    
    test_df=mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df, "testing_data")
    
    #Log source code
    mlflow.log_artifact(__file__)
    
    #Log the best model
    mlflow.sklearn.log_model(grid_search.best_estimator_, "best_model")
    
    #Set tags
    mlflow.set_tags({"Author": "Sandesh", "Project": "Breast Cancer Classification"})
    
    print("Best Parameters:", best_params)
    print("Best Score:", best_score)    