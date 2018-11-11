import glob
import time

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from extractors import extract_features_for_images
import numpy as np


class Config:
    cspace = 'YCrCb'
    hog_channel = 'ALL'
    spatial_size = (16, 16)
    hist_bins = 16
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    spatial_feat = True
    hist_feat = True
    hog_feat = True

    @classmethod
    def to_kwargs(cls):
        return dict(
            [item for item in cls.__dict__.items() if not (item[0].startswith('__') or item[0] == 'to_kwargs')]
        )


def train(vehicles, non_vehicles, save_files=True):
    car_features = extract_features_for_images(
        vehicles, **Config.to_kwargs()
    )

    not_car_features = extract_features_for_images(
        non_vehicles, **Config.to_kwargs()
    )

    # Create an array stack of feature vectors
    X = np.vstack((car_features, not_car_features)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(not_car_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state
    )

    # Fit a per-column scaler only on the training data
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X_train and X_test
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    print('Examples lenght:' , len(X_train))
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    if save_files:
        print('Saving model ....')
        joblib.dump(svc, 'svc.joblib')
        print('Saving scaler .....')
        joblib.dump(X_scaler, 'scaler.joblib')

    return svc, X_scaler


def train_model():
    vehicles = glob.glob('./data/vehicles/**/*.png')
    non_vehicles = glob.glob('./data/non-vehicles/**/*.png')

    train(
        vehicles,
        non_vehicles,
    )
