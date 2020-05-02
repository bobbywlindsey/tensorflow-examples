import tensorflow as tf
# from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# Download fasion mnist data
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
# Reshape the data
input_shape = (training_images[0].shape[0], training_images[0].shape[1], 1)
training_images = training_images.reshape(len(training_images), *input_shape)
test_images = test_images.reshape(len(test_images), *input_shape) # test images should be same shape as train images
# If using 'categorical_crossentropy' as the loss function,
# convert ordinal labels to one-hot
# num_classes = 10
# training_labels = to_categorical(training_labels, num_classes)
# test_labels = to_categorical(test_labels, num_classes)

# Normalize and augment
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
# Noramlize
validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator      = train_datagen.flow(training_images, training_labels, batch_size = 32)
validation_generator = validation_datagen.flow(test_images, test_labels, batch_size    = 16)

# Construct convolutional neural network
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(100, activation='relu'),
                                    tf.keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss      = 'sparse_categorical_crossentropy',
              metrics   = ['accuracy'])


# Train the model and store metrics
history = model.fit(
      train_generator,
      steps_per_epoch=len(training_images)/32,  
      epochs=20,
      verbose=1,
      validation_data=validation_generator,
      validation_steps=len(test_images)/32)

# Display metrics
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
