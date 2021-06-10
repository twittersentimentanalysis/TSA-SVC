import json
import pickle
import pandas   as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics            import plot_confusion_matrix, classification_report
from sklearn.svm                import SVC
from sklearn.model_selection    import GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold


# Funtion to fine-tune hyperparameters to find the optimal ones
def find_best_params_svc(X_train_transformed, y_train_lables_trf):
    params = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 
        'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 'scale'],
        'degree': [0, 1, 2, 3, 4, 5, 6],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'class_weight': ['balanced', None],
        'decision_function_shape': ['ovo', 'ovr']
    }

    # define evaluation
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)

    # define the search
    search = RandomizedSearchCV(estimator=SVC(break_ties=True), param_distributions=params, n_jobs=-1, cv=cv, scoring='accuracy', verbose=3, refit=True)

    # fitting the model for search 
    search.fit(X_train_transformed, y_train_lables_trf) 

    # print results
    print(search.best_score_)
    print(search.best_estimator_) 

    return search.best_estimator_

# Load trained model saved
def load_model():
    # load configuration file
    js = open('config.json').read()
    config = json.loads(js)

    # load the model saved
    path = config['pre-trained-model']
    loaded_model = pickle.load(open(path, 'rb'))

    return loaded_model


# Evaluate the performance of the model 
# (performance metrics and confusion matrix)
def evaluate_model(model, X_test, y_test, label_dict):  
    # get predicitions 
    y_pred = model.predict(X_test)

    print(label_dict)

    # plot confusion matrix
    plot_confusion_matrix(  model, 
                            X_test, 
                            y_test,
                            labels = list(label_dict.values()),
                            display_labels = list(label_dict.keys()),
                            normalize = 'true')
    plt.show()
    
    # classification report (precision, recall, f1-score, accuracy)
    cr = classification_report(y_test, y_pred, digits=5, target_names=label_dict.keys())
    print(cr)