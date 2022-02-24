import timeit

import numpy as np

# from matplotlib import pyplot as plt

from pandas import read_csv
from pandas import DataFrame

from sklearn.preprocessing import StandardScaler as scaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# suppress sklearn warning
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(
    "ignore", category=DataConversionWarning)

import argparse
import sys
from os.path import exists

#command-line arguments
arguments = []

for argument in sys.argv[1:]:
    arguments.append(argument.split("--")[1])
    if '=' in argument:
        filename = argument.split("--")[1].split("=")[1]

if 'dimred' not in arguments and 'classify' not in arguments:
    print("Please use either --dimred or --classify to run the program")
    sys.exit(1)

# constants
file_exists = exists(filename)
if file_exists:
    PATH = filename
else:
    PATH = ""
    print("Please enter a valid file path")
    sys.exit(1)

BATCH_SIZE = 5000

results = {}

df = read_csv(PATH)

# print(df.shape)
# print(df.describe)

X, y = df.iloc[:, 1:-1], df.iloc[:, [-1]]

# fig = plt.figure(figsize=(15, 20))
# ax = fig.gca()
# X.hist(ax=ax)

'''

Pre-processing

'''

print('Pre-processing...')

scaled = scaler().fit_transform(X)
scaled_X = DataFrame(scaled, columns=X.columns)


# print(scaled_X.shape)
# print(scaled_X.describe)

# fig = plt.figure(figsize=(15, 20))
# ax = fig.gca()
# scaled_X.hist(ax=ax)

'''

Built-in Methods

'''

# built-in PCA
print('Built-in PCA...')

start_time = timeit.default_timer()

pca = PCA(n_components=6)
pca_X = pca.fit_transform(scaled_X)
pca_df_X = DataFrame(pca_X)

results['builtin_pca_time'] = timeit.default_timer() - start_time

# print(pca_df_X.shape)
# print(pca_df_X.describe)


# built-in Logistic Regression
print('Built-in Logistic regression (training)...')

X_train, X_test, y_train, y_test = train_test_split(
    pca_df_X, y, test_size=0.20, random_state=0)

start_time = timeit.default_timer()

lr_model = LogisticRegression()
lr_model.fit(X_train.values, y_train.values)

results['builtin_lr_time'] = timeit.default_timer() - start_time

print('Built-in Logistic regression (evaluation)...')

predictions = lr_model.predict(X_test.values)

results['builtin_lr_accuracy'] = lr_model.score(X_test.values, y_test.values)
results['builtin_lr_predictions'] = predictions

# ConfusionMatrixDisplay.from_predictions(y_test.values, predictions)
# print(f"Accuracy = {lr_model.score(X_test.values, y_test.values)}")

'''

Manual Implementation

'''

# gamma matrix calculation


def Z(X, y=None):
    # assume that X is an n x d dataset (n rows of observations, d columns of features) and Y (output) is the last column in dataset
    Z = X.copy().T
    row_1 = np.ones(Z.shape[1])
    row_y = y.T[0] if y is not None else np.zeros(Z.shape[1])
    Z = np.insert(Z, 0, row_1, axis=0)
    Z = np.insert(Z, Z.shape[0], row_y, axis=0)
    return Z


def gamma(X, y=None):
    z = Z(X, y)
    gamma = np.dot(z, z.T)
    return gamma


def k_gamma(gamma):
    d = len(gamma) - 2
    l = L(gamma)
    q = Q(gamma)
    k_g = np.zeros((d+1, d+1))
    k_g[0, 0] = gamma[0, 0]
    k_g[1:, 0] = l
    k_g[0, 1:] = np.transpose(l)
    for i in range(d):
        k_g[i+1, i+1] = q[i, i]
    return k_g


def L(gamma):
    d = len(gamma) - 2
    return gamma[1:d+1, 0]


def Q(gamma):
    d = gamma.shape[1] - 2
    return gamma[1:d+1, 1:d+1]


# if working in chunks, intermediate gammas should be d x d in size according to paper.
def update_gamma(old_gamma, new_gamma):
    return np.add(old_gamma, new_gamma)


# PCA

def pca(gamma, ev_threshold=1.00):

    n = int(gamma[0][0])
    q = Q(gamma)
    l = L(gamma)

    corr_mat = np.zeros(q.shape)

    for a in range(q.shape[0]):
        for b in range(q.shape[1]):
            corr_mat[a][b] = ((n * q[a][b]) - (l[a] * l[b])) / (
                np.sqrt((n * q[a][a]) - (l[a] ** 2)) * np.sqrt((n * q[b][b]) - (l[b] ** 2)))

    # eigenvectors and eigenvalues
    U = np.zeros(n)
    S = np.zeros(n)

    U, S, V = np.linalg.svd(corr_mat)

    principal_comps = [dim for dim, ev in enumerate(S) if ev >= ev_threshold]

    return U[principal_comps]


def dim_reduction(X, U):
    return np.matmul(X, U.T)


# Class Decomposition using K-Means

def init_centroids(X, k):
    # randomly select k data points as initial centroids
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    return centroids


def find_closest_centroid(x, centroids):
    J = [np.dot((x - centroid).T, (x - centroid)) for centroid in centroids]

    return np.argmin(J)


def find_closest_centroids(X, centroids):
    # for each data point in X, find the index of the closest centroid
    idx = [find_closest_centroid(x, centroids) for x in X]

    return idx


def get_mean_from_gamma(cluster):
    gamma_matrix = gamma(cluster)
    k_gamma_matrix = k_gamma(gamma_matrix)

    n = k_gamma_matrix[0, 0]
    L = k_gamma_matrix[1:, 0]

    return (L / n)


def compute_means(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))

    for i in np.arange(K):
        cluster = X[idx == i]
        centroids[i] = get_mean_from_gamma(cluster)

    return centroids


def run_kmeans(X, K, max_iters=10):
    # initialize the centroids
    centroids = init_centroids(X, K)

    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)

        centroids = compute_means(X, idx, K)

        print(f"Iteration {i+1}")

    return idx, centroids


def fit(X, y, K, max_iters=10):
    X = np.array(X)

    # separate data into classes
    class_0_indices = np.where(y == 0)[0]
    class_1_indices = np.where(y == 1)[0]

    class_0 = np.take(X, class_0_indices, axis=0)
    class_1 = np.take(X, class_1_indices, axis=0)

    model = {
        'class_0': run_kmeans(class_0, K, max_iters),
        'class_1': run_kmeans(class_1, K, max_iters)
    }

    return model


def predict(X, model):
    predictions = []

    X = np.array(X)

    for x in X:
        # find closest centroids from each class
        idx_0 = find_closest_centroid(x, model['class_0'][1])
        idx_1 = find_closest_centroid(x, model['class_1'][1])

        # predict class
        closest_centroid = find_closest_centroid(
            x, np.array([model['class_0'][1][idx_0], model['class_1'][1][idx_1]]))

        predictions.append(closest_centroid)

    return predictions

if 'dimred' in arguments:
    # loading data and computing gamma
    print('Computing gamma for PCA...')

    chunks = read_csv(PATH, chunksize=BATCH_SIZE)

    gamma_final = np.array([])

    start_time = timeit.default_timer()

    for chunk in chunks:
        X, y_chunk = chunk.iloc[:, 1:-1].to_numpy(), chunk.iloc[:, [-1]].to_numpy()

        gamma_chunk = gamma(X, y_chunk)
        gamma_final = gamma_final if gamma_final.size else np.zeros(
            gamma_chunk.shape)
        gamma_final = update_gamma(gamma_final, gamma_chunk)

    # print(gamma_final.shape)
    # print(gamma_final)

    # computing PCA
    print('Performing PCA from gamma...')

    ev_treshold = 1.00

    pca_U = pca(gamma_final, ev_threshold=ev_treshold)

    # print(pca_U.shape)

    # dimensionality reduction
    pca_X = dim_reduction(scaled, pca_U)
    pca_df_X = DataFrame(pca_X)

    results['gamma_pca_time'] = timeit.default_timer() - start_time

    # print(pca_df_X.shape)
    # print(pca_df_X.describe)

if 'classify' in arguments:
    # training
    K = 5
    num_iters = 1

    print(f'Fitting K-Means (K={K} {num_iters} iterations)...')

    X_train, X_test, y_train, y_test = train_test_split(
        pca_df_X, y, test_size=0.20, random_state=0)

    start_time = timeit.default_timer()

    # @TODO: tune these parameters
    model = fit(X_train, y_train, K=5, max_iters=num_iters)

    results['gamma_kmeans_time'] = timeit.default_timer() - start_time

    # evaluation
    print('Evaluating K-Means...')

    predictions = predict(X_test, model)

    results['gamma_kmeans_accuracy'] = accuracy_score(y_test, predictions)
    results['gamma_kmeans_predictions'] = predictions

# ConfusionMatrixDisplay.from_predictions(y_test.values, predictions)
# print(f"Accuracy = {accuracy_score(y_test.values, predictions)}")


# printing results
print('\n\n## RESULTS ##\n\n')

# PCA
if 'dimred' in arguments and results['builtin_pca_time'] and results['gamma_pca_time']:
    print('PCA: \n')
    print('Time taken by built-in PCA: ', results['builtin_pca_time'])
    print('Time taken by gamma-based PCA: ', results['gamma_pca_time'])

    print('\n')

if 'classify' in arguments and results['builtin_lr_time'] and results['gamma_kmeans_time']:
    # Built-in LR
    print('Built-in Logistic Regression: \n')
    print('Time taken by built-in Logistic Regression: ',
          results['builtin_lr_time'])
    print('Accuracy of built-in Logistic Regression: ',
          results['builtin_lr_accuracy'])
    print('Confusion Matrix of built-in Logistic Regression: ')
    print(confusion_matrix(y_test.values, results['builtin_lr_predictions']))
    # ConfusionMatrixDisplay.from_predictions(
    #     y_test.values, results['builtin_lr_predictions'])
    # plt.show()

    print('\n')

    # Gamma-based K-Means
    print('Gamma-based K-Means: \n')
    print('Time taken by gamma-based K-Means: ', results['gamma_kmeans_time'])
    print('Accuracy of gamma-based K-Means: ',
          results['gamma_kmeans_accuracy'])
    print('Confusion Matrix of gamma-based K-Means: ')
    print(confusion_matrix(y_test.values, results['gamma_kmeans_predictions']))
    # ConfusionMatrixDisplay.from_predictions(
    #     y_test.values, results['gamma_kmeans_predictions'])
    # plt.show()


