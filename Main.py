import Training
import Testing
import Evaluation
import Initialization


def main():
    train()
    # test()


# Function to train and test data with one dataset
def train():
    # Initialize data
    df, label_dict = Initialization.initialize()

    # Preprocessing data and training 
    X_train, X_test, y_train, y_test, count_vect, labels = Training.initialize(df)
    X_train_transformed, y_train_lables_trf, X_test_transformed, y_test_labels_trf  = Training.encode_data(X_train, X_test, y_train, y_test, count_vect, labels)

    # Obtain the model and train it
    model = Evaluation.find_best_params_svc(X_train_transformed, y_train_lables_trf)
    Training.train(model, X_train_transformed, y_train_lables_trf)

    # Load trained model and evaluate it 
    calibrated_svc = Evaluation.load_model()
    print(calibrated_svc)
    Evaluation.evaluate_model(calibrated_svc, X_test_transformed, y_test_labels_trf, label_dict)


# Function to test data with other dataset different 
# from the one used for training
def test():
    # Initialize data
    df, label_dict = Initialization.initialize()

    # Encodign testing data
    X_test, y_test, count_vect, transformer, labels = Testing.initialize(df)
    X_test_transformed, y_test_labels_trf  = Testing.encode_data(X_test, y_test, count_vect, transformer, labels)
    
    # Load trained model and evaluate 
    calibrated_svc = Evaluation.load_model()
    Evaluation.evaluate_model(calibrated_svc, X_test_transformed, y_test_labels_trf, label_dict)


if __name__ == '__main__':
	main()