import tensorflow as tf

# Download fasion mnist data
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
# plt.imshow(training_images[0])
# plt.show()

# Stop training if accuracy reaches over 95%
class MyCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.95):
      print("\nReached 95% accuracy so cancelling training!")
      self.model.stop_training = True

callback = MyCallback()

# Construct deep neural network
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(100, activation='relu'),
                                    tf.keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer = tf.optimizers.Adam(),
              loss      = 'sparse_categorical_crossentropy',
              metrics   = ['accuracy'])

model.fit(training_images, training_labels, epochs=5, callbacks=[callback])
model.evaluate(test_images, test_labels)
