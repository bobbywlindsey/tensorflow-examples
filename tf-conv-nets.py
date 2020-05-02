import tensorflow as tf
import matplotlib.pyplot as plt


# Function to visualize convolutions
def conv_visual(model, FIRST_IMAGE, SECOND_IMAGE, THIRD_IMAGE, CONVOLUTION_NUMBER):
    f, axarr = plt.subplots(3,4)
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
    for x in range(0,4): # 4 activations because 2 conv2ds and 2 maxpoolings
        f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
        axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
        axarr[0,x].grid(False)
        f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
        axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
        axarr[1,x].grid(False)
        f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
        axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
        axarr[2,x].grid(False)
    plt.show()


# Download fasion mnist data
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
# Reshape the data
input_shape = (training_images[0].shape[0], training_images[0].shape[1], 1)
training_images = training_images.reshape(len(training_images), *input_shape)
test_images = test_images.reshape(len(test_images), *input_shape) # test images should be same shape as train images
# Normalize
training_images  = training_images / 255.0
test_images = test_images / 255.0


# Stop training if accuracy reaches over 95%
class MyCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.95):
      print("\nReached 95% accuracy so cancelling training!")
      self.model.stop_training = True

callback = MyCallback()

# Construct convolutional neural network
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(100, activation='relu'),
                                    tf.keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer = tf.optimizers.Adam(),
              loss      = 'sparse_categorical_crossentropy',
              metrics   = ['accuracy'])

model.fit(training_images, training_labels, epochs=5, callbacks=[callback])
model.evaluate(test_images, test_labels)

conv_visual(model, 0, 7, 26, 1)
