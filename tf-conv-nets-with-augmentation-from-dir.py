import tensorflow as tf
# from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

TRAINING_DIR = './train'
VALIDATION_DIR = './test'

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
# Normalize
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator      = train_datagen.flow_from_directory(TRAINING_DIR, target_size=(100, 100),
                                                         batch_size = 128, class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, target_size=(100, 100),
                                                              batch_size = 32, class_mode='binary')
input_shape = (100, 100, 3)

# Construct convolutional neural network
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(50, activation='relu'),
                                    tf.keras.layers.Dense(1, activation='sigmoid')])
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss      = 'binary_crossentropy',
              metrics   = ['accuracy'])


# Train the model and store metrics
history = model.fit(
      train_generator,
      steps_per_epoch=16,
      epochs=20,
      verbose=1,
      validation_data=validation_generator,
      validation_steps=16)

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
