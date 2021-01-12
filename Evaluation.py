import json
import pickle
import pandas   as pd
import numpy    as np
import matplotlib.pyplot as plt

from sklearn.model_selection    import cross_val_score
from sklearn.metrics            import confusion_matrix, plot_confusion_matrix, classification_report
from sklearn.svm                import LinearSVC, SVC, OneClassSVM
from sklearn.model_selection    import GridSearchCV
from skopt                      import BayesSearchCV


def find_best_params_linear_svc(X_train_transformed, y_train_lables_trf):
    # defining parameter range 
    param_grid={"C": [0.001, 0.01, 0.1, 1.0, 10.0],
                # "class_weight": [  'balanced',
                #                     {0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
                #                     {0: 2, 1: 3, 2: 4, 3: 5, 4: 1},
                #                     {0: 3, 1: 4, 2: 5, 3: 1, 4: 2},
                #                     {0: 4, 1: 5, 2: 1, 3: 2, 4: 3},
                #                     {0: 5, 1: 1, 2: 2, 3: 3, 4: 4},
                #                     {0: 50, 1: 1, 2: 1, 3: 1, 4: 1},
                #                     {0: 1, 1: 50, 2: 1, 3: 1, 4: 1},
                #                     {0: 1, 1: 1, 2: 50, 3: 1, 4: 1},
                #                     {0: 1, 1: 1, 2: 1, 3: 50, 4: 1},
                #                     {0: 1, 1: 1, 2: 1, 3: 1, 4: 50} ], 
                "max_iter": [1000, 1500, 2000, 2500],
                "multi_class": ['ovr', 'crammer_singer']
                }

    grid =  BayesSearchCV(LinearSVC(class_weight='balanced'),
                        param_grid, 
                        refit = True, 
                        scoring='f1_micro',
                        verbose = True,
                        n_jobs = -1) 

    # fitting the model for grid search 
    grid.fit(X_train_transformed, y_train_lables_trf) 

    print(grid.best_params_) 
    print(grid.best_estimator_) 

    return grid.best_estimator_


def find_best_params_svc(X_train_transformed, y_train_lables_trf):
    # defining parameter range 
    param_grid = {#'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 
                # 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid']} 

    grid = GridSearchCV(SVC(C= 1000, gamma = 0.1, class_weight={0: 5, 1: 7, 2: 16}), 
                        param_grid, 
                        cv = 4,
                        refit = True, 
                        scoring='f1_micro',
                        verbose = True,
                        n_jobs = -1) 

    # fitting the model for grid search 
    grid.fit(X_train_transformed, y_train_lables_trf) 

    results = pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params', 'rank_test_score']]
    results.sort_values(by='rank_test_score', inplace=True)
    print(results)

    # params_2nd_best = results.loc[0, 'params']
    # clf_2nd_best = grid.best_estimator_.set_params(**params_2nd_best)
    # print(clf_2nd_best)

    print(grid.best_params_) 
    print(grid.best_score_)
    print(grid.best_estimator_) 

    return grid.best_estimator_

def find_best_params_one_class_svc(X_train_transformed, y_train_lables_trf):
    # defining parameter range 
    param_grid = {  'gamma': ['auto', 'scale', 1, 0.1, 0.01, 0.001, 0.0001], 
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']} 

    grid = GridSearchCV(OneClassSVM(), 
                        param_grid, 
                        cv = 4,
                        refit = True, 
                        scoring='accuracy',
                        verbose = True,
                        n_jobs = -1) 

    # fitting the model for grid search 
    grid.fit(X_train_transformed, y_train_lables_trf) 

    results = pd.DataFrame(grid.cv_results_)
    results.sort_values(by='rank_test_score', inplace=True)
    print(results)

    # params_2nd_best = results.loc[0, 'params']
    # clf_2nd_best = grid.best_estimator_.set_params(**params_2nd_best)
    # print(clf_2nd_best)

    print(grid.best_params_) 
    print(grid.best_score_)
    print(grid.best_estimator_) 

    print(pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']])

    return grid.best_estimator_

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
    # pred = model.predict_proba(X_test)
    # print(pred)
    # cm = confusion_matrix(y_test, y_pred,  np.array(list((label_dict.values()))))
    # print(cm)

    from sklearn.metrics import accuracy_score
    print("ACCURACY: ", accuracy_score(y_test, y_pred))

    plot_confusion_matrix(  model, 
                            X_test, 
                            y_test,
                            labels = list(label_dict.values()),
                            display_labels = list(label_dict.keys()),
                            normalize = 'all')
    plt.show()
    
    cr = classification_report(y_test, y_pred, digits=5, target_names=label_dict.keys())
    print(cr)

    predicted = model.predict(X_test)
    print('Average accuracy on test set={}'.format(np.mean(predicted == labels.transform(y_test))))