from keras.datasets import reuters
import tensorflow as tf

from keras import models
from keras import layers
import numpy as np

(train_data, train_labels) , (test_data, test_labels) = reuters.load_data(num_words = 10000)

def vectorize_data(my_data, dim = 10000):
  results = np.zeros((len(my_data), dim))
  for index, dt in enumerate(my_data):
    results[index][dt] = 1
  return results

def to_one_hot(ind , dim = 46):
  results = np.zeros((len(ind), dim))
  for i, j in enumerate(ind):
    results[i][j - 1] = 1
  return results


new_train_data = vectorize_data(train_data)
new_test_data = vectorize_data(test_data)

new_train_labels = np.array(train_labels)
new_test_labels = np.array(test_labels)
#new_train_labels = to_one_hot(train_labels)
#new_test_labels = to_one_hot(test_labels)



model = models.Sequential()
model.add(layers.Dense(4096, activation = 'relu', input_shape = (10000,)))
model.add(layers.Dense(46, activation = 'softmax'))

model.compile(optimizer = "rmsprop", loss = "sparse_categorical_crossentropy", metrics = ['accuracy'])

part_data_x = new_train_data[0:1000]
data_x = new_train_data[1000:]

part_labels = new_train_labels[0:1000]
data_labels = new_train_labels[1000:]


history = model.fit(part_data_x,
                    part_labels,
                    epochs = 10,
                    batch_size = 512,
                    validation_data = (data_x, data_labels))

print(model.evaluate(data_x, data_labels ))
