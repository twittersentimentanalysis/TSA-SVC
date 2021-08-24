import json


# Get emotions probabilities for text recieved
def get_emotion(text, count_vect, tf_transformer, calibrated_svc, label_dict):
    # read configuration file
    js = open('config-api.json').read()
    config = json.loads(js)

    # predict emotions
    to_predict = [text]
    p_count = count_vect.transform(to_predict)
    p_tfidf = tf_transformer.transform(p_count)
    result = calibrated_svc.predict_proba(p_tfidf)
    print(result)

    # printing predicted probability in json
    emotions_prob = {}
    possible_labels = config['label-dict']

    for index, possible_label in enumerate(possible_labels):
        index = label_dict[possible_label]
        emotions_prob[possible_label] = round(result[0][index], 5)

    emotions = {"emotion": emotions_prob}
    print(emotions)

    return emotions
