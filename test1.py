import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop

# Check if GPU is available and TensorFlow is using it
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

# Loading the CIFAR10 data
(X, y), (_, _) = tf.keras.datasets.cifar10.load_data()

# Selecting a single class of images
X = X[y.flatten() == 8]

# Defining the Input shape
image_shape = (32, 32, 3)
latent_dimensions = 100

# Build the generator
def build_generator():
    model = Sequential()

    model.add(Dense(256 * 8 * 8, activation="relu", input_dim=latent_dimensions))
    model.add(Reshape((8, 8, 256)))
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.78))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.78))
    model.add(Activation("relu"))
    model.add(Conv2D(3, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    noise = Input(shape=(latent_dimensions,))
    image = model(noise)

    return Model(noise, image)

# Build the discriminator
def build_discriminator():
    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=image_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.82))
    model.add(LeakyReLU(alpha=0.25))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.82))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.25))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    image = Input(shape=image_shape)
    validity = model(image)

    return Model(image, validity)

# Display images function remains the same

# Build and compile the discriminator
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0008, clipvalue=1.0), metrics=['accuracy'])

# Make the discriminator untrainable when we are training the generator
discriminator.trainable = False

# Build the generator
generator = build_generator()

z = Input(shape=(latent_dimensions,))
image = generator(z)

valid = discriminator(image)

combined_network = Model(z, valid)
combined_network.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0004, clipvalue=1.0))

# Training parameters
num_epochs = 15000
batch_size = 32
display_interval = 2500

# Rescale -1 to 1
X = (X / 127.5) - 1.

valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

# Lists to keep track of loss
discriminator_losses = []
generator_losses = []

# Begin training
with open('training_log.txt', 'w') as f:
    for epoch in range(num_epochs+1):
        # Training loop remains the same
        # The following code block should be indented to be inside the for loop

        # Select a random half batch of images
        index = np.random.randint(0, X.shape[0], batch_size)
        images = X[index]

        # Sample noise and generate a half batch of new images
        noise = np.random.normal(0, 1, (batch_size, latent_dimensions))
        generated_images = generator.predict(noise)

        # ... (rest of the training code)

        # Display progress for each specified interval
        if epoch % display_interval == 0 or epoch == num_epochs:
            print(f"Epoch: {epoch}, Discriminator Loss: {discm_loss[0]}, Discriminator Accuracy: {100*discm_loss[1]}, Generator Loss: {genr_loss}")
            f.write(f"Epoch: {epoch}, Discriminator Loss: {discm_loss[0]}, Discriminator Accuracy: {100*discm_loss[1]}, Generator Loss: {genr_loss}\n")
            display_images(epoch)
            
# After the training loop, plot the losses
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(generator_losses, label="G")
plt.plot(discriminator_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
