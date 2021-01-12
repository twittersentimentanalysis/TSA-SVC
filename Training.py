import json
import pickle

from sklearn.svm                        import LinearSVC, SVC
from sklearn.calibration                import CalibratedClassifierCV  
from sklearn.model_selection            import train_test_split
from sklearn.feature_extraction.text    import TfidfTransformer
from sklearn.feature_extraction.text    import CountVectorizer
from sklearn.preprocessing              import LabelEncoder

def initialize(df):
    X_train, X_test, y_train, y_test = train_test_split(df.index.values, 
                                                    df.label.values, 
                                                    test_size = 0.2, 
                                                    random_state = 17)

    df['data_type'] = ['not_set']*df.shape[0]
    df.loc[X_train, 'data_type'] = 'train'
    df.loc[X_test, 'data_type'] = 'test'

    count_vect = CountVectorizer()
    labels = LabelEncoder()

    X_train = df[df.data_type == 'train'].processed_tweet.values
    X_test = df[df.data_type == 'test'].processed_tweet.values

    return X_train, X_test, y_train, y_test, count_vect, labels


def encode_data(X_train, X_test, y_train, y_test, count_vect, labels):
    X_train_counts = count_vect.fit_transform(X_train)
    tf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_transformed = tf_transformer.transform(X_train_counts)

    X_test_counts = count_vect.transform(X_test)
    X_test_transformed = tf_transformer.transform(X_test_counts)

    labels.fit(y_train)
    y_train_lables_trf = labels.transform(y_train)
    y_test_labels_trf = labels.transform(y_test)
    # print(labels.classes_)

    print("LENGTH TRAINING: " + str(len(y_train_lables_trf)))
    print("LENGTH VALIDATION: " + str(len(y_test_labels_trf)))

    return X_train_transformed, y_train_lables_trf, X_test_transformed, y_test_labels_trf, tf_transformer

# Training a LinearSVC classifier and using CalibratedClassifierCV 
# to get probabilities for each predicted class
def train(model, X_train_transformed, y_train_lables_trf, X_test_transformed):
    # read configuration file
    js = open('config.json').read()
    config = json.loads(js)
    
    model.fit(X_train_transformed, y_train_lables_trf)

    calibrated_svc = CalibratedClassifierCV(base_estimator = model,
                                            cv = "prefit")

    calibrated_svc.fit(X_train_transformed, y_train_lables_trf)
    calibrated_svc.predict(X_test_transformed)

    # save the model to disk
    path = config['pre-trained-model']
    pickle.dump(calibrated_svc, open(path, 'wb'))
