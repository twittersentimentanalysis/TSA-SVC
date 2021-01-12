import json
import pandas as pd

def get_emotion(text, count_vect, tf_transformer, calibrated_svc, label_dict):
    # read configuration file
    js = open('config.json').read()
    config = json.loads(js)

    # predict emotions
    to_predict = [text]
    p_count = count_vect.transform(to_predict)
    p_tfidf = tf_transformer.transform(p_count)
    # print('Average accuracy on test set={}'.format(np.mean(predicted == labels.transform(y_test))))
    # print('Predicted probabilities of demo input string are')
    result = calibrated_svc.predict_proba(p_tfidf)
    print(result)

    # Printing predicted probability in json
    emotions_prob = {}
    possible_labels = config['possible-labels']

    for index, possible_label in enumerate(possible_labels):
        index = label_dict[possible_label]
        emotions_prob[possible_label] = round(result[0][index], 5)

    emotions = {"emotion": emotions_prob}
    print(emotions)

    return emotions
