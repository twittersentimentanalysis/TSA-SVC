import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
 

# Initialize data
def initialize():
    # load configuration file
    js = open('config.json').read()
    config = json.loads(js)

    # load the model from disk
    csv_file = config['csv-file']

    # read data from csv file
    df = pd.read_csv(   csv_file,
                        header = 0, 
                        index_col = 'id',
                        usecols = ['id', 'processed_tweet', 'emotion'])

    # count samples
    print("Original: ", len(df))

    # remove empty processed tweets
    df = df[~df.processed_tweet.isnull()]

    # count samples
    print("Relevant: ", len(df))

    # visualizing distribution of classes
    plt.figure(figsize=(8, 8))
    sns.countplot('emotion', data=df)
    plt.title('Balanced Classes')
    plt.show()

    # get number of samples of each class (emotion)
    df = df[['emotion','processed_tweet']]
    res = df.groupby('emotion').count()
    print(res)

    # enumerate categories
    possible_labels = sorted(df.emotion.unique())

    # get label dictionary
    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    df['label'] = df.emotion.replace(label_dict)
    label_dict = dict(sorted(label_dict.items()))   

    print(label_dict) 
    
    return df, label_dict