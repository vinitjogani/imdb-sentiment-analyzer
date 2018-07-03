import os, pandas as pd

def read_dirs(directories, save_to):
    reviews = []
    ratings = []
    # Loop through directories
    for directory in directories:
        for filename in os.listdir(directory):
            # Get path
            filepath = os.path.join(directory, filename)
            # Extract features
            review = open(filepath, encoding="utf8").read().replace("<br />", " ")
            rating = int(filename.split("_")[1][:-4])
            # Accumulate in array
            reviews.append(review)
            ratings.append(rating)
    # Save as dataframe
    df = pd.DataFrame()
    df['review'] = reviews
    df['rating'] = ratings
    df.to_pickle(save_to)

read_dirs(['train/pos', 'train/neg'], 'train.pickle')
read_dirs(['test/pos', 'test/neg'], 'test.pickle')