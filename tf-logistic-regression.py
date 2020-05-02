import tensorflow as tf

# Download fasion mnist data
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
# plt.imshow(training_images[0])
# plt.show()


# Normalize images
training_images  = training_images / 255.0
test_images = test_images / 255.0

# Construct logistic regression model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer = tf.optimizers.Adam(),
              loss      = 'sparse_categorical_crossentropy',
              metrics   = ['accuracy'])

model.fit(training_images, training_labels, epochs=5)
model.evaluate(test_images, test_labels)
