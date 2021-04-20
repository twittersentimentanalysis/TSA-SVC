import json
import pickle
import pandas   as pd
import numpy    as np
import matplotlib.pyplot as plt

from sklearn.model_selection    import cross_val_score
from sklearn.metrics            import confusion_matrix, plot_confusion_matrix, classification_report
from sklearn.svm                import LinearSVC, SVC, OneClassSVM
from sklearn.model_selection    import GridSearchCV, RandomizedSearchCV, StratifiedKFold, RepeatedStratifiedKFold
from skopt                      import BayesSearchCV


def find_best_params_svc(X_train_transformed, y_train_lables_trf, label_dict):
    # define search space
    params = dict()
    params['C'] = (1e-6, 10000.0, 'log-uniform')
    params['gamma'] = (1e-6, 100.0, 'log-uniform')
    params['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']

    # define evaluation
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
    # define the search
    search = BayesSearchCV(estimator=SVC(), search_spaces=params, n_jobs=-1, cv=cv, scoring='accuracy')

    # fitting the model for search 
    search.fit(X_train_transformed, y_train_lables_trf) 

    results = pd.DataFrame(search.cv_results_)[['mean_test_score', 'std_test_score', 'params', 'rank_test_score']]
    print(results.sort_values(by='mean_test_score', inplace=True))

    print(search.best_params_) 
    print(search.best_score_)
    print(search.best_estimator_) 

    return search.best_estimator_


def load_model():
    # load configuration file
    js = open('config.json').read()
    config = json.loads(js)

    # load the model from disk
    path = config['pre-trained-model']
    loaded_model = pickle.load(open(path, 'rb'))

    return loaded_model

def evaluate_model(model, X_test, y_test, labels, label_dict):   
    y_pred = model.predict(X_test)

    # plot confusion matrix
    plot_confusion_matrix(  model, 
                            X_test, 
                            y_test,
                            labels = list(label_dict.values()),
                            display_labels = list(label_dict.keys()),
                            normalize = 'all')
    plt.show()
    
    # classification report (precision, recall, f1-score, accuracy)
    cr = classification_report(y_test, y_pred, digits=5, target_names=label_dict.keys())
    print(cr)