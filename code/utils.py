import math
import csv
import copy
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.cluster import estimate_bandwidth, MeanShift
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec


# Coordinates of Porto's city centre
porto = [41.1579, -8.6291]


def np_haversine(latlon1, latlon2):
    """
    Numpy version of the Haversine function to calculate distances between two sets of points.
    Converted to Python from the R version provided in the competition's evaluation script.
    Returns the distance in km.
    """
    lat1 = latlon1[:, 0]
    lon1 = latlon1[:, 1]
    lat2 = latlon2[:, 0]
    lon2 = latlon2[:, 1]
    
    REarth = 6371
    lat = np.abs(lat1 - lat2) * np.pi / 180
    lon = np.abs(lon1 - lon2) * np.pi / 180
    lat1 = lat1 * np.pi / 180
    lat2 = lat2 * np.pi / 180
    a = np.sin(lat / 2) * np.sin(lat / 2) + np.cos(lat1) * np.cos(lat2) * np.sin(lon / 2) * np.sin(lon / 2)
    d = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return REarth * d


def tf_haversine(latlon1, latlon2):
    """
    Tensorflow version of the Haversine function to calculate distances between two sets of points.
    """
    lat1 = latlon1[:, 0]
    lon1 = latlon1[:, 1]
    lat2 = latlon2[:, 0]
    lon2 = latlon2[:, 1]

    REarth = 6371
    lat = tf.abs(lat1 - lat2) * np.pi / 180
    lon = tf.abs(lon1 - lon2) * np.pi / 180
    lat1 = lat1 * np.pi / 180
    lat2 = lat2 * np.pi / 180
    a = tf.sin(lat / 2) * tf.sin(lat / 2) + tf.cos(lat1) * tf.cos(lat2) * tf.sin(lon / 2) * tf.sin(lon / 2)
    d = 2 * tf_atan2(tf.sqrt(a), tf.sqrt(1 - a))
    return REarth * d


def tf_atan2(y, x):
    """
    Tensorflow doesn't have an Atan2 function (at least not yet, see: https://github.com/tensorflow/tensorflow/issues/6095).
    So we define it here ourselves.
    """
    angle = tf.select(tf.greater(x,0.0), tf.atan(y/x), tf.zeros_like(x))
    angle = tf.select(tf.logical_and(tf.less(x,0.0),  tf.greater_equal(y,0.0)), tf.atan(y/x) + np.pi, angle)
    angle = tf.select(tf.logical_and(tf.less(x,0.0),  tf.less(y,0.0)), tf.atan(y/x) - np.pi, angle)
    angle = tf.select(tf.logical_and(tf.equal(x,0.0), tf.greater(y,0.0)), 0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.select(tf.logical_and(tf.equal(x,0.0), tf.less(y,0.0)), -0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.select(tf.logical_and(tf.equal(x,0.0), tf.equal(y,0.0)), np.nan * tf.zeros_like(x), angle)
    return angle


def get_clusters(coords):
    """
    Estimate clusters for the given list of coordinates.
    """
    # First, grossly reduce the spatial dataset by rounding up the coordinates to the 4th decimal
    # (i.e. 11 meters. See: https://en.wikipedia.org/wiki/Decimal_degrees)
    clusters = pd.DataFrame({
        'approx_latitudes': coords[:,0].round(4),
        'approx_longitudes': coords[:,1].round(4)
    })
    clusters = clusters.drop_duplicates(['approx_latitudes', 'approx_longitudes'])
    clusters = clusters.as_matrix()
    
    # Further reduce the number of clusters
    # (Note: the quantile parameter was tuned to find a significant and reasonable number of clusters)
    bandwidth = estimate_bandwidth(clusters, quantile=0.0002)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(clusters)
    return ms.cluster_centers_


def density_map(latitudes, longitudes, center=porto, bins=1000, radius=0.1):
    """
    Displays a density map in a matplotlib histogram for all the points
    defined by the given latitudes and longitudes.
    """
    # LogNorm will flag 0-value pixels as "bad" and color them in white.
    # So we tweak the color map to color them in black.
    cmap = copy.copy(plt.cm.jet)
    cmap.set_bad((0,0,0))  # Fill background with black

    # Center the map around the provided center coordinates
    histogram_range = [
        [center[1] - radius, center[1] + radius],
        [center[0] - radius, center[0] + radius]
    ]
    
    plt.figure(figsize=(5,5))
    plt.hist2d(longitudes, latitudes, bins=bins, norm=LogNorm(),
               cmap=cmap, range=histogram_range)

    # Remove all axes and annotations to keep the map clean and simple
    plt.grid('off')
    plt.axis('off')
    plt.gca().xaxis.set_visible(False)
    plt.gca().yaxis.set_visible(False)
    plt.tight_layout()
    plt.show()
    

def plot_embeddings(embeddings):
    """
    Plot all the given embeddings on a 2D plane using t-SNE.
    """
    # Determine the dimension of the subplots' grid
    N = len(embeddings)
    cols = 2
    rows = int(math.ceil(N / float(cols)))
    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure(figsize=(8,7))

    for i, embedding in enumerate(embeddings):
        ax = fig.add_subplot(gs[i])

        weights = embedding[1].get_weights()[0]
        names = range(weights.shape[0])

        # Compute TSNE to convert the embedding's weights to a 2D plane
        tsne = TSNE(n_components=2, random_state=0)
        tsne = tsne.fit_transform(weights)
        x, y = tsne[:,0], tsne[:,1]

        # Plot the points
        scatter = ax.scatter(x, y, alpha=0.7, c=names, s=40, cmap="jet")
        fig.colorbar(scatter, ax=ax)

        # Display the scatter point annotations
        for i, name in enumerate(names):
            ax.annotate(name, (x[i], y[i]), size=6)

        # Make sure all points fit nicely within the space
        x_delta = x.max() - x.min()
        x_margin = x_delta / 10
        y_delta = y.max() - y.min()
        y_margin = y_delta / 10
        ax.set_xlim(x.min()-x_margin, x.max()+x_margin)
        ax.set_ylim(y.min()-y_margin, y.max()+y_margin)

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_title(embedding[0])
        fig.tight_layout()
    plt.show()


def export_answers(model, competition_test, filename='answers.csv'):
    """
    Export the predictions for the test dataset to be
    submitted as the answer for the competition.
    """
    from code.training import process_features
    predictions = model.predict(process_features(competition_test))
    answer_csv = open(filename, 'w')
    answer_csv = csv.writer(answer_csv, quoting=csv.QUOTE_NONNUMERIC)
    answer_csv.writerow(['TRIP_ID', 'LATITUDE', 'LONGITUDE'])
    for index, (latitude, longitude) in enumerate(predictions):
        answer_csv.writerow([competition_test['TRIP_ID'][index], latitude, longitude])
