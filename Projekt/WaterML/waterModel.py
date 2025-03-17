import os
import numpy as np
import pickle

from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# prepare data
input_dir = r'C:\Users\rasmu\OneDrive\Skrivebord\clf-data' # path to data
categories = ['empty', 'not_empty'] # categories in data

data = []
labels = []

for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path) # reads picture
        img = resize(img, (15, 15)) # resize image to 15x15 pixels
        data.append(img.flatten()) # make image to array
        labels.append(category_idx) # add label to image


data = np.asarray(data)
labels = np.asarray(labels)

# train /test split

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels) # Make 80% train and 20% test data
 # Stratify = make sure labels matches in train and test data
classifier = SVC() # Support Vector Classifier / SVM
parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}] # Train mulitple image classificers for each gamma and C value = 12 image classificers
grid_search = GridSearchCV(classifier, parameters) 
grid_search.fit(x_train, y_train) # train classifier



# test performance
best_estimator = grid_search.best_estimator_ # chooese best classifier


y_prediction = best_estimator.predict(x_test) # predict test data

score = accuracy_score(y_test, y_prediction) # calculate accuracy score

print('{}% of samples were correctly classified'.format(score*100)) 

pickle.dump(best_estimator, open('./model.p', 'wb')) # save model to file



