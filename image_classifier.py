import os
from keras import layers, Model
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
# import matplotlib.pyplot as plt

train_dir = r"C:\Users\Vidhu\Documents\jupyter-notebooks\Cats_and_Dogs\train"
val_dir = r"C:\Users\Vidhu\Documents\jupyter-notebooks\Cats_and_Dogs\val"
test_dir = r"C:\Users\Vidhu\Documents\jupyter-notebooks\Cats_and_Dogs\test"

train_dog_dir = os.path.join(train_dir, "dog")
train_cat_dir = os.path.join(train_dir, "cat")

val_dog_dir = os.path.join(val_dir, "dog")
val_cat_dir = os.path.join(val_dir, "cat")

test_dog_dir = os.path.join(test_dir, "dog")
test_cat_dir = os.path.join(test_dir, "cat")

# print(train_dog_dir)
# print(train_cat_dir)
# print(val_dog_dir)
# print(val_cat_dir)
# print(test_dog_dir)
# print(test_cat_dir)


# Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
# the three color channels: R, G, and B
img_input = layers.Input(shape=(150, 150, 3))

# First convolution extracts 16 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(16, 3, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)

# Second convolution extracts 32 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# Third convolution extracts 64 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Convolution2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# Flatten feature map to a 1-dim tensor
x = layers.Flatten()(x)

# Create a fully connected layer with ReLU activation and 512 hidden units
x = layers.Dense(512, activation='relu')(x)

# Add a dropout rate of 0.5
# x = layers.Dropout(0.5)(x)

# Create output layer with a single node and sigmoid activation
output = layers.Dense(1, activation='sigmoid')(x)

# Configure and compile the model
model = Model(img_input, output)
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.0001),
              metrics=['acc'])

train_label_gen = ImageDataGenerator(rescale=1./255,)
                                    # rotation_range=40,
                                    # width_shift_range=0.2,
                                    # height_shift_range=0.2,
                                    # shear_range=0.2,
                                    # zoom_range=0.2,
                                    # horizontal_flip=True)

val_label_gen = ImageDataGenerator(rescale=1./255)

train_labels = train_label_gen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=10,
    class_mode='binary'
)

# print('Train labels created')

val_labels = val_label_gen.flow_from_directory(
    val_dir,
    target_size=(150,150),
    batch_size=10,
    class_mode='binary'
)

# print('Validation labels created')

history = model.fit(
    train_labels,
    batch_size=6,
    steps_per_epoch=256,
    epochs=10,
    validation_data=val_labels,
    validation_steps=64,
    verbose=1
)

model.save(r"model_weights_1.h5")

# Retrieve a list of accuracy results on training and validation data
# sets for each training epoch
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# # Retrieve a list of list results on training and validation data
# # sets for each training epoch
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# # Get number of epochs
# epochs = range(len(acc))

# # Plot training and validation accuracy per epoch
# plt.plot(epochs, acc)
# plt.plot(epochs, val_acc)
# plt.title('Training and validation accuracy')

# plt.figure()

# # Plot training and validation loss per epoch
# plt.plot(epochs, loss)
# plt.plot(epochs, val_loss)
# plt.title('Training and validation loss')