import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

data = pd.read_csv('Hiring.csv')

data['experience'].fillna(0, inplace = True)

data['test_score'].fillna(data['test_score'].mean(), inplace = True)

X = data.iloc[:, :3]

def convert_to_int(word):
    word_dict = {
        'one' : 1,
        'two' : 2,
        'three' : 3,
        'four' : 4,
        'five' : 5,
        'six' : 6,
        'seven' : 7,
        'eight' : 8,
        'nine' : 9,
        'ten' : 10,
        'eleven' : 11,
        'twelve' : 12,
        'zero' : 0,
        0 : 0
        }
    return word_dict[word]

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

y = data.iloc[:, -1]

# We split the datast here, but as we have a  very small dataset, we are not splitting it

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


#Fitting the data into he model
regressor.fit(X,y)

# Saving the model in the pickle

pickle.dump(regressor, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
print (model.predict([[2,9,6]]))



