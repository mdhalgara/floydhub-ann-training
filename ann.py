# Data Pre-Processing

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Fill in missing data
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# imputer = imputer.fit(x[:, 1:3])
# x[:, 1:3] = imputer.transform(x[:, 1:3])

# Encoding categorial data
labelencoder_x_1 = LabelEncoder()
labelencoder_x_2 = LabelEncoder()
x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])
x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1])
x = onehotencoder.fit_transform(x).toarray()
x = x[:, 1:]
# labelencoder_y = LabelEncoder()
# y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the training set and Test set
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

# Feature Scaling
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

classifier = Sequential()

# Adding the input layer and the first hidden layer with dropout
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu',
                     input_shape=(11,)))
classifier.add(Dropout(rate=0.1))

# Adding the second hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(rate=0.1))

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform',
                     activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy',
                   metrics=['accuracy'])

# Fitting the ANN to the Traininng Set
classifier.fit(x_train, y_train, batch_size=10, epochs=100, verbose=1)

# Part 3 - Making the predictions and evaluating the model

# Predicing the Test set results
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Part 4 - Evaluating, Improving,  and Tuning the ANN

# Evaluating the ANN

#
#def build_classifier():
#    classifier = Sequential()
#    # Adding the input layer and the first hidden layer
#    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu',
#                         input_shape=(11,)))
#    classifier.add(
#        Dense(units=6, kernel_initializer='uniform', activation='relu'))
#    # Adding the output layer
#    classifier.add(
#        Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
#    # Compiling the ANN
#    classifier.compile(optimizer='adam', loss='binary_crossentropy',
#                       metrics=['accuracy'])
#    return classifier
#
#
#classifier = KerasClassifier(
#    build_fn=build_classifier, batch_size=10, epochs=100,)
#accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10,
#                             n_jobs=1, error_score='raise')
#mean = accuracies.mean()
#variance = accuracies.std()

# Improving the ANN
# Dropout Regularization to reduce overfitting if needed


def build_classifier(optimizer):
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu',
                         input_shape=(11,)))
    classifier.add(
        Dense(units=6, kernel_initializer='uniform', activation='relu'))
    # Adding the output layer
    classifier.add(
        Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    # Compiling the ANN
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy',
                       metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, verbose=1)
parameters = {'batch_size': [16,32,64],
              'nb_epoch': [100, 500, 510],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10)
grid_search = grid_search.fit(x_train, y_train)
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_

print("Best acc Score: %f using %s" % (grid_search.best_score_
                                       , grid_search.best_params_))
# Tuning the ANN
