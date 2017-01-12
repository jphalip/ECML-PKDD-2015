import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class Data:
    """
    Class for holding a number of datasets and metadata.
    """
    pass


def convert_coordinates(string):
    """
    Loads list of coordinates from given string and swap out longitudes & latitudes.
    We do the swapping because the standard is to have latitude values first, but
    the original datasets provided in the competition have it backwards.
    """
    return [(lat, long) for (long, lat) in json.loads(string)]


def random_truncate(coords):
    """
    Randomly truncate the end of the trip's polyline points to simulate partial trips.
    This is only intended to be used for our custom train/validation/test datasets
    and not for the final test dataset provided by the competition as that one is
    already partial.
    """
    
    # There's no need to truncate if there's not more than one item
    if len(coords) <= 1:
        return coords
    
    # Pick a random number of items to be removed from the list.
    # (We do "-1" to ensure we have at least one item left)
    n = np.random.randint(len(coords)-1)

    if n > 0:
        # Return the list without its last n items
        return coords[:-n]
    else:
        # No truncation needed in this case
        return coords

    
def encode_feature(feature, train, test):
    """
    Encode the labels for the given feature across both the train and test datasets.
    """
    encoder = LabelEncoder()
    train_values = train[feature].copy()
    test_values = test[feature].copy()
    # Replace missing values with 0's so we can later encode them
    train_values[np.isnan(train_values)] = 0
    test_values[np.isnan(test_values)] = 0
    # Fit the labels across all possible values in both datasets
    encoder.fit(pd.concat([train_values, test_values]))
    # Add new column to the datasets with encoded values
    train[feature + '_ENCODED'] = encoder.transform(train_values)
    test[feature + '_ENCODED'] = encoder.transform(test_values)
    return encoder


def extract_features(df):
    """
    Extract some features from the original columns in the given dataset.
    """
    # Convert polyline values from strings to list objects
    df['POLYLINE'] = df['POLYLINE'].apply(convert_coordinates)
    # Extract start latitudes and longitudes
    df['START_LAT'] = df['POLYLINE'].apply(lambda x: x[0][0])
    df['START_LONG'] = df['POLYLINE'].apply(lambda x: x[0][1])
    # Extract quarter hour of day
    datetime_index = pd.DatetimeIndex(df['TIMESTAMP'])
    df['QUARTER_HOUR'] = datetime_index.hour * 4 + datetime_index.minute / 15   
    # Extract day of week
    df['DAY_OF_WEEK'] = datetime_index.dayofweek
    # Extract week of year
    df['WEEK_OF_YEAR'] = datetime_index.weekofyear - 1
    # Extract trip duration (GPS coordinates are recorded every 15 seconds)
    df['DURATION'] = df['POLYLINE'].apply(lambda x: 15 * len(x))

    
def remove_outliers(df, labels):
    """
    Remove some outliers that could otherwise undermine the training's results.
    """
    # Remove trips that are either extremely long or short (potentially due to GPS recording issue)
    indices = np.where((df.DURATION > 60) & (df.DURATION <= 2 * 3600))
    df = df.iloc[indices]
    labels = labels[indices]
    
    # Remove trips that are too far away from Porto (also likely due to GPS issues)
    bounds = (  # Bounds retrieved using http://boundingbox.klokantech.com
        (41.052431, -8.727951),
        (41.257678, -8.456039)
    )
    indices = np.where(
        (labels[:,0]  >= bounds[0][0]) &
        (labels[:,1] >= bounds[0][1]) &
        (labels[:,0]  <= bounds[1][0]) &
        (labels[:,1] <= bounds[1][1])
    )
    df = df.iloc[indices]
    labels = labels[indices]
    
    return df, labels

    
def load_data():
    """
    Loads data from CSV files, processes and caches it in pickles for faster future loading.
    """
    
    train_cache = 'cache/train.pickle'
    train_labels_cache = 'cache/train-labels.npy'
    validation_cache = 'cache/validation.pickle'
    validation_labels_cache = 'cache/validation-labels.npy'
    test_cache = 'cache/test.pickle'
    test_labels_cache = 'cache/test-labels.npy'
    competition_test_cache = 'cache/competition-test.pickle'
    metadata_cache = 'cache/metadata.pickle'
    
    if os.path.isfile(train_cache):
        # Load from cached files if they already exist
        train = pd.read_pickle(train_cache)
        validation = pd.read_pickle(validation_cache)
        test = pd.read_pickle(test_cache)
        train_labels = np.load(train_labels_cache)
        validation_labels = np.load(validation_labels_cache)
        test_labels = np.load(test_labels_cache)
        competition_test = pd.read_pickle(competition_test_cache)
        with open(metadata_cache, 'rb') as handle:
            metadata = pickle.load(handle)
    else:
        datasets = []
        for kind in ['train', 'test']:
            # Load original CSV file
            csv_file = 'datasets/%s.csv' % kind
            df = pd.read_csv(csv_file)
            # Ignore items that are missing data
            df = df[df['MISSING_DATA'] == False]
            # Ignore items that don't have polylines
            df = df[df['POLYLINE'] != '[]']
            # Delete the now useless column to save a bit of memory
            df.drop('MISSING_DATA', axis=1, inplace=True)
            # Delete an apparently useless column (all values are 'A')
            df.drop('DAY_TYPE', axis=1, inplace=True)
            # Fix format of timestamps
            df['TIMESTAMP'] = df['TIMESTAMP'].astype('datetime64[s]')
            # Extra some new features
            extract_features(df)
            datasets.append(df)

        train, competition_test = datasets

        # Encode some features
        client_encoder = encode_feature('ORIGIN_CALL', train, competition_test)
        taxi_encoder = encode_feature('TAXI_ID', train, competition_test)
        stand_encoder = encode_feature('ORIGIN_STAND', train, competition_test)

        # Randomly truncate the trips to simulate partial trips like in the competition's test dataset.
        train['POLYLINE_FULL'] = train['POLYLINE'].copy()  # First, keep old version handy for future reference.
        train['POLYLINE'] = train['POLYLINE'].apply(random_truncate)  # Then truncate.

        # The labels are the last polyline coordinates, i.e. the trips' destinations.
        train_labels = np.column_stack([
            train['POLYLINE_FULL'].apply(lambda x: x[-1][0]),
            train['POLYLINE_FULL'].apply(lambda x: x[-1][1])
        ])
        
        # Remove some outliers
        train, train_labels = remove_outliers(train, train_labels)

        # Gather some metadata that will later be useful during training
        metadata = {
            'n_quarter_hours': 96,  # Number of quarter of hours in one day (i.e. 24 * 4).
            'n_days_per_week': 7,
            'n_weeks_per_year': 52,
            'n_client_ids': len(client_encoder.classes_),
            'n_taxi_ids': len(taxi_encoder.classes_),
            'n_stand_ids': len(stand_encoder.classes_),
        }
        
        # Split original train dataset into new train (98%), validation (1%) and test (1%) datasets.        
        train, validation, train_labels, validation_labels = train_test_split(train, train_labels, test_size=0.02)
        validation, test, validation_labels, test_labels = train_test_split(validation, validation_labels, test_size=0.5)
        
        # Cache results in files
        train.to_pickle(train_cache)
        validation.to_pickle(validation_cache)
        test.to_pickle(test_cache)
        np.save(train_labels_cache, train_labels)
        np.save(validation_labels_cache, validation_labels)
        np.save(test_labels_cache, test_labels)
        competition_test.to_pickle(competition_test_cache)
        with open(metadata_cache, 'wb') as handle:
            pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    data = Data()
    data.__dict__.update({
        'train': train,
        'train_labels': train_labels,
        'validation': validation,
        'validation_labels': validation_labels,
        'test': test,
        'test_labels': test_labels,
        'competition_test': competition_test,
        'metadata': metadata,
    })
    return data
