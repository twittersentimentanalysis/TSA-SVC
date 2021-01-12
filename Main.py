import json
import numpy                as np
import pandas               as pd

import Training
import Evaluation
import Initialization


def main():
    # Initialize data
    df, label_dict = Initialization.initialize()

    # Preprocessing data and training 
    X_train, X_test, y_train, y_test, count_vect, labels = Training.initialize(df)
    X_train_transformed, y_train_lables_trf, X_test_transformed, y_test_labels_trf, _  = Training.encode_data(X_train, X_test, y_train, y_test, count_vect, labels)

    # model = Evaluation.find_best_params_linear_svc(X_train_transformed, y_train_lables_trf)
    # model = Evaluation.find_best_params_svc(X_train_transformed, y_train_lables_trf)
    # model = Evaluation.find_best_params_one_class_svc(X_train_transformed, y_train_lables_trf)
    from sklearn.svm import LinearSVC, SVC, OneClassSVM
    model = SVC(C=1000, class_weight='balanced', gamma=0.01, kernel='rbf')
    Training.train(model, X_train_transformed, y_train_lables_trf, X_test_transformed)

    # Load trained model
    calibrated_svc = Evaluation.load_model()
    print(calibrated_svc)
    Evaluation.evaluate_model(calibrated_svc, X_test_transformed, y_test_labels_trf, labels, label_dict)


def load():
    # Initialize data
    df, label_dict = Initialization.initialize()

    # Load model and necessary data
    X_train, X_test, y_train, y_test, count_vect, labels = Training.initialize(df)
    _, _, _, _, tf_transformer  = Training.encode_data(X_train, X_test, y_train, y_test, count_vect, labels)
    calibrated_svc = Evaluation.load_model()
    
    return count_vect, tf_transformer, calibrated_svc, label_dict

if __name__ == '__main__':
	main()