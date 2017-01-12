import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import scale
from keras.models import Sequential
from keras.optimizers import SGD, Adam, Adagrad
from keras import backend as K
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Reshape, Merge, Activation, Dropout
from keras.callbacks import ModelCheckpoint
from code.utils import tf_haversine
from code.data import load_data
from code.utils import get_clusters


def start_new_session():
    """
    Starts a new Tensorflow session.
    """
    
    # Make sure the session only uses the GPU memory that it actually needs
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    session = tf.Session(config=config, graph=tf.get_default_graph())
    K.tensorflow_backend.set_session(session)


def first_last_k(coords):
    """
    Returns a list with the first k and last k GPS coordinates from the given trip.
    The returned list contains 4k values (latitudes and longitudes for 2k points).
    """
    k = 5
    partial = [coords[0] for i in range(2*k)]
    num_coords = len(coords)
    if num_coords < 2*k:
        partial[-num_coords:] = coords
    else:
        partial[:k] = coords[:k]
        partial[-k:] = coords[-k:]
    partial = np.row_stack(partial)
    return np.array(partial).flatten()


def process_features(df):
    """
    Process the features required by our model from the given dataframe.
    Return the features in a list so that they can be merged in our model's input layer.
    """
    # Fetch the first and last GPS coordinates
    coords = np.row_stack(df['POLYLINE'].apply(first_last_k))
    # Standardize latitudes (odd columns) and longitudes (even columns)
    latitudes = coords[:,::2]
    coords[:,::2] = scale(latitudes)
    longitudes = coords[:,1::2]
    coords[:,1::2] = scale(longitudes)
    
    return [
        df['QUARTER_HOUR'].as_matrix(),
        df['DAY_OF_WEEK'].as_matrix(),
        df['WEEK_OF_YEAR'].as_matrix(),
        df['ORIGIN_CALL_ENCODED'].as_matrix(),
        df['TAXI_ID_ENCODED'].as_matrix(),
        df['ORIGIN_STAND_ENCODED'].as_matrix(),
        coords,
    ]


def create_model(metadata, clusters):
    """
    Creates all the layers for our neural network model.
    """
      
    # Arbitrary dimension for all embeddings
    embedding_dim = 10

    # Quarter hour of the day embedding
    embed_quarter_hour = Sequential()
    embed_quarter_hour.add(Embedding(metadata['n_quarter_hours'], embedding_dim, input_length=1))
    embed_quarter_hour.add(Reshape((embedding_dim,)))

    # Day of the week embedding
    embed_day_of_week = Sequential()
    embed_day_of_week.add(Embedding(metadata['n_days_per_week'], embedding_dim, input_length=1))
    embed_day_of_week.add(Reshape((embedding_dim,)))

    # Week of the year embedding
    embed_week_of_year = Sequential()
    embed_week_of_year.add(Embedding(metadata['n_weeks_per_year'], embedding_dim, input_length=1))
    embed_week_of_year.add(Reshape((embedding_dim,)))

    # Client ID embedding
    embed_client_ids = Sequential()
    embed_client_ids.add(Embedding(metadata['n_client_ids'], embedding_dim, input_length=1))
    embed_client_ids.add(Reshape((embedding_dim,)))

    # Taxi ID embedding
    embed_taxi_ids = Sequential()
    embed_taxi_ids.add(Embedding(metadata['n_taxi_ids'], embedding_dim, input_length=1))
    embed_taxi_ids.add(Reshape((embedding_dim,)))

    # Taxi stand ID embedding
    embed_stand_ids = Sequential()
    embed_stand_ids.add(Embedding(metadata['n_stand_ids'], embedding_dim, input_length=1))
    embed_stand_ids.add(Reshape((embedding_dim,)))
    
    # GPS coordinates (5 first lat/long and 5 latest lat/long, therefore 20 values)
    coords = Sequential()
    coords.add(Dense(1, input_dim=20, init='normal'))

    # Merge all the inputs into a single input layer
    model = Sequential()
    model.add(Merge([
                embed_quarter_hour,
                embed_day_of_week,
                embed_week_of_year,
                embed_client_ids,
                embed_taxi_ids,
                embed_stand_ids,
                coords
            ], mode='concat'))

    # Simple hidden layer
    model.add(Dense(500))
    model.add(Activation('relu'))

    # Determine cluster probabilities using softmax
    model.add(Dense(len(clusters)))
    model.add(Activation('softmax'))

    # Final activation layer: calculate the destination as the weighted mean of cluster coordinates
    cast_clusters = K.cast_to_floatx(clusters)
    def destination(probabilities):
        return tf.matmul(probabilities, cast_clusters)
    model.add(Activation(destination))

    # Compile the model
    optimizer = SGD(lr=0.01, momentum=0.9, clipvalue=1.)  # Use `clipvalue` to prevent exploding gradients
    model.compile(loss=tf_haversine, optimizer=optimizer)
    
    return model


def full_train(n_epochs=100, batch_size=200, save_prefix=None):
    """
    Runs the complete training process.
    """
    
    # Load initial data
    print "Loading data..."
    data = load_data()

    # Estimate the GPS clusters
    print "Estimating clusters..."
    clusters = get_clusters(data.train_labels)
    
    # Set up callbacks
    callbacks = []
    if save_prefix is not None:
        # Save the model's intermediary weights to disk after each epoch
        file_path="cache/%s-{epoch:03d}-{val_loss:.4f}.hdf5" % save_prefix
        callbacks.append(ModelCheckpoint(file_path, monitor='val_loss', mode='min', save_weights_only=True, verbose=1))

    # Create model
    print "Creating model..."
    start_new_session()
    model = create_model(data.metadata, clusters)
    
    # Run the training
    print "Start training..."
    history = model.fit(
        process_features(data.train), data.train_labels,
        nb_epoch=n_epochs, batch_size=batch_size,
        validation_data=(process_features(data.validation), data.validation_labels),
        callbacks=callbacks)

    if save_prefix is not None:
        # Save the training history to disk
        file_path = 'cache/%s-history.pickle' % save_prefix
        with open(file_path, 'wb') as handle:
            pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return history