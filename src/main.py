import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow import keras
from keras import models
from keras import layers
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder


if __name__ == '__main__':
    data = os.path.join('data/','spotify_songs.csv')
    df = pd.read_csv(data)
    isEDA = False
    isTrain = True
    split_seed = 20030407
    weight_seed = 20020503
    tensorflow.random.set_seed(weight_seed)

    # Data Prep
    df.dropna()
    df = df.drop('track_id', axis=1)
    df = df.drop('track_artist', axis=1)
    df = df.drop('track_name', axis=1)
    df = df.drop('track_album_id', axis=1)
    df = df.drop('track_album_name', axis=1)
    df = df.drop('playlist_name', axis=1)
    df = df.drop('playlist_id', axis=1)
    df = df.drop('track_album_release_date', axis=1)
    df = df.drop('playlist_subgenre', axis=1)
    label_encoder = LabelEncoder()
    df['genre_encode'] = label_encoder.fit_transform(df['playlist_genre'])
    inputs = ['track_popularity','danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms']
    output = ['genre_encode']
    len_inputs = len(inputs)
    len_outputs = len(output)
    n_neurons = 128
    n_epoch = 7

    # EDA
    if(isEDA):

        # Genre's Histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        hex_colors = ['#6C567B', '#C06C84','#F67280','#F8B195']
        plt.hist(df.loc[:,'playlist_genre'],  bins=15, edgecolor='black')
        plt.xlabel('Genres')
        plt.ylabel('Amount of Songs')
        plt.title('Histogram of Genres')
        histogram_path = os.path.join('output/', 'GenreHistogram.png')
        plt.savefig(histogram_path)
        plt.close()

        # Correlations
        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(16, 16))
        ax = sns.heatmap(corr, annot=True)
        correlation_path = os.path.join('output/', 'Correlation.png')
        plt.title('Correlations', fontsize=40)
        plt.savefig(correlation_path)
        plt.close()

        # Collinearity
        collin = corr.abs()
        fig, ax = plt.subplots(figsize=(16, 16))
        sns.heatmap(collin, vmin=0.9, vmax = 1.0, annot=True)
        collinearity_path = os.path.join('output/', 'Collinearity.png')
        plt.title('Collinearity', fontsize=40)
        plt.savefig(collinearity_path)
        plt.close()

    # Model
    if(isTrain):
        X = df.loc[:,inputs]
        y = df.loc[:,output]
        X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=split_seed)
        model = models.Sequential()
        model.add(layers.Dense(units=n_neurons,input_shape = (len_inputs,)))
        model.add(layers.Activation('relu'))
        model.add(layers.Dense(units=len_outputs))
        model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mse'])
        model.fit(X_train, y_train, epochs=n_epoch)

        # Validation
        test_mae = mean_absolute_error(y_test, model.predict(X_test))
        train_mae = mean_absolute_error(y_train, model.predict(X_train))
        print("Test MAE:", test_mae)
        print("Train MAE:", train_mae)

        prediction = model.predict(X_train)
        residuals = y_train - prediction
        plt.scatter(prediction, residuals)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Training Residual Plot')
        residual_plot_path = os.path.join('output/', 'TrainResidual.png')
        plt.savefig(residual_plot_path)
        plt.close()

        prediction = model.predict(X_test)
        residuals = y_test - prediction
        plt.scatter(prediction, residuals)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Testing Residual Plot')
        residual_plot_path = os.path.join('output/', 'TestResidual.png')
        plt.savefig(residual_plot_path)
        plt.close()

