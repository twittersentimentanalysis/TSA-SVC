import json
import pickle

# Initialize testing data
def initialize(df):
    X_test = df.processed_tweet.values
    y_test = df.label.values

    # load configuration file
    js = open('config.json').read()
    config = json.loads(js)

    # load encoded_data tuple from disk
    path = config['encoded-data']
    count_vect, transformer, labels = pickle.load(open(path, 'rb'))

    return X_test, y_test, count_vect, transformer, labels

# Enocde data for testing
def encode_data(X_test, y_test, count_vect, transformer, labels):
    X_test_counts = count_vect.transform(X_test)
    X_test_transformed = transformer.transform(X_test_counts)
    y_test_labels_trf = labels.transform(y_test)

    return X_test_transformed, y_test_labels_trf