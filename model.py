from keras import layers, Model
from keras.optimizers import Adam

def conv_classifier():
    img_input = layers.Input(shape=(150,150,3))

    x1 = layers.Conv2D(64, 3, activation='relu')(img_input)
    x1 = layers.MaxPool2D(2)(x1)

    x2 = layers.Conv2D(32, 3, activation='relu')(x1)
    x2 = layers.MaxPool2D(2)(x2)

    x3 = layers.Conv2D(16, 3, activation='relu')(x2)
    x3 = layers.MaxPool2D(2)(x3)

    x4 = layers.Flatten()(x3)

    x5 = layers.Dense(512, activation='relu')(x4)

    output = layers.Dense(1, activation='sigmoid')(x5)

    model = Model(img_input, output)

    # model.summary()

    model.compile(loss='binary_crossentropy',
                optimizer=Adam(learning_rate=0.001),
                metrics=['accuracy'])

    return model