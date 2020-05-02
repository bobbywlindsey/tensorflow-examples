import tensorflow as tf
import numpy as np
from tensorflow import keras

# Data with linear relationship (y = 2x-1)
training_data = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
training_labels = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Construct linear regression model
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train it
model.fit(training_data, training_labels, epochs=500)

# Make some prediction
print(model.predict([10.0])) # Should be close to 19
