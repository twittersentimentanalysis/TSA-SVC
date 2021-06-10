import json
import pickle

from sklearn.calibration                import CalibratedClassifierCV  
from sklearn.model_selection            import train_test_split
from sklearn.feature_extraction.text    import TfidfTransformer
from sklearn.feature_extraction.text    import CountVectorizer
from sklearn.preprocessing              import LabelEncoder

# Prepare data for training and testing
def initialize(df):
    # Split data into 80 % for training and 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(df.index.values, 
                                                    df.label.values, 
                                                    test_size = 0.2, 
                                                    random_state = 42)
    df['data_type'] = ['not_set']*df.shape[0]
    df.loc[X_train, 'data_type'] = 'train'
    df.loc[X_test, 'data_type'] = 'test'

    X_train = df[df.data_type == 'train'].processed_tweet.values
    X_test = df[df.data_type == 'test'].processed_tweet.values

    count_vect = CountVectorizer()
    labels = LabelEncoder()

    return X_train, X_test, y_train, y_test, count_vect, labels


# Encode training and test data
def encode_data(X_train, X_test, y_train, y_test, count_vect, labels):
    X_train_counts = count_vect.fit_transform(X_train)
    tf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_transformed = tf_transformer.transform(X_train_counts)
    
    X_test_counts = count_vect.transform(X_test)
    X_test_transformed = tf_transformer.transform(X_test_counts)

    labels.fit(y_train)
    y_train_lables_trf = labels.transform(y_train)
    y_test_labels_trf = labels.transform(y_test)

    # Get datasets length
    print("LENGTH TRAINING: " + str(len(y_train_lables_trf)))
    print("LENGTH VALIDATION: " + str(len(y_test_labels_trf)))

    # read configuration file
    js = open('config.json').read()
    config = json.loads(js)

    # save encoded data to use it for testing
    path = config['encoded-data']
    encoded_data = (count_vect, tf_transformer, labels)
    pickle.dump(encoded_data, open(path, 'wb'))    

    return X_train_transformed, y_train_lables_trf, X_test_transformed, y_test_labels_trf


# Training a SVC classifier and using CalibratedClassifierCV 
# to get probabilities for each predicted class
def train(model, X_train_transformed, y_train_lables_trf):
    # read configuration file
    js = open('config.json').read()
    config = json.loads(js)

    # train svc model
    model.fit(X_train_transformed, y_train_lables_trf)

    # train calibrated classifier
    calibrated_svc = CalibratedClassifierCV(base_estimator = model, cv = 'prefit', method='sigmoid')
    calibrated_svc.fit(X_train_transformed, y_train_lables_trf)

    # save the model to disk
    path = config['pre-trained-model']
    pickle.dump(calibrated_svc, open(path, 'wb'))