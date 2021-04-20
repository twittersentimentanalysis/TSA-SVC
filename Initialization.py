import json
import pandas as pd
import matplotlib.pyplot as plt

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

    # # BALANCE DATA
    # # Shuffle the Dataset.
    # shuffled_df = df.sample(frac=1,random_state=4)

    # # Randomly select 492 observations from the non-fraud (majority class)
    # happy_df = shuffled_df.loc[shuffled_df['emotion'] == 'happy'].sample(n=2700,random_state=63)
    # sad_df = shuffled_df.loc[shuffled_df['emotion'] == 'sad'].sample(n=2700,random_state=32)
    # # surprise_df = shuffled_df.loc[shuffled_df['emotion'] == 'surprise'].sample(n=315,random_state=42)
    # # nrelevant_df = shuffled_df.loc[shuffled_df['emotion'] == 'not-relevant'].sample(n=45,random_state=42)
    # angry_df = shuffled_df.loc[shuffled_df['emotion'] == 'angry'].sample(n=2700,random_state=75)

    # # Concatenate both dataframes again
    # df = pd.concat([happy_df, sad_df, angry_df])

    # import seaborn as sns
    # #plot the dataset after the undersampling
    # plt.figure(figsize=(8, 8))
    # sns.countplot('emotion', data=df)
    # plt.title('Balanced Classes')
    # plt.show()


    # visualizing distribution of classes
    df = df[['emotion','processed_tweet']]
    df.groupby('emotion').count().plot.bar(ylim=0)
    plt.show()

    df = df[['emotion','processed_tweet']]
    res = df.groupby('emotion').count()
    print(res)

    # enumerate categories
    possible_labels = sorted(df.emotion.unique())

    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index

    df['label'] = df.emotion.replace(label_dict)

    print(df.head())
    label_dict = dict(sorted(label_dict.items()))
    print(label_dict)
    
    
    return df, label_dict