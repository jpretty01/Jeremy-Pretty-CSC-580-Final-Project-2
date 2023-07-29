# Jeremy Pretty
# CSC 580 Final Project Option 1
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import LeakyReLU
from keras.layers import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam,SGD
import keras

# Loading the CIFAR10 data
(X, y), (_, _) = keras.datasets.cifar10.load_data()

# Selecting a single class of images
X = X[y.flatten() == 8]

# Defining the Input shape
image_shape = (32, 32, 3)
latent_dimensions = 100

# Build the generator
def build_generator():
    model = Sequential()
    
    # Start with a dense layer that takes random noise as input
    model.add(Dense(128 * 8 * 8, activation="relu", input_dim=latent_dimensions))
    model.add(Reshape((8, 8, 128)))
    
    # Upsample to 16x16
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.78))
    model.add(Activation("relu"))
    
    # Upsample to 32x32
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.78))
    model.add(Activation("relu"))
    
    # Final output layer with tanh activation function
    model.add(Conv2D(3, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    # Input and output for the model
    noise = Input(shape=(latent_dimensions,))
    image = model(noise)

    return Model(noise, image)

# Build the discriminator
def build_discriminator():
    model = Sequential()

    # First Convolutional layer
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    # Second Convolutional layer
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.82))
    model.add(LeakyReLU(alpha=0.25))
    model.add(Dropout(0.25))

    # Third Convolutional layer
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.82))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    # Fourth Convolutional layer
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.25))
    model.add(Dropout(0.25))

    # Flatten layer
    model.add(Flatten())

    # Output layer with sigmoid activation function
    model.add(Dense(1, activation='sigmoid'))

    # Input and output for the model
    image = Input(shape=image_shape)
    validity = model(image)

    return Model(image, validity)

# Display images
def display_images(epoch):
    r, c = 4,4
    noise = np.random.normal(0, 1, (r * c,latent_dimensions))
    generated_images = generator.predict(noise)

    # Rescale images 0 - 1
    generated_images = 0.5 * generated_images + 0.5

    fig, axs = plt.subplots(r, c)
    count = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(generated_images[count, :,:,])
            axs[i,j].axis('off')
            count += 1
    plt.savefig('images_at_epoch_{:05d}.png'.format(epoch))
    plt.close()

# Build and compile the discriminator
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0001,0.5), metrics=['accuracy'])

# Make the discriminator untrainable when we are training the generator.  This doesn't effect the discriminator by itself
discriminator.trainable = False  

# Build and compile the combined network (generator and discriminator).  
# While training the generator, we'll keep the discriminator's weights fixed.
generator = build_generator()

z = Input(shape=(latent_dimensions,))
image = generator(z)

valid = discriminator(image)

combined_network = Model(z, valid)
combined_network.compile(loss='binary_crossentropy', optimizer=Adam(0.0002,0.5))

num_epochs=15001
batch_size=16
display_interval=2500
losses=[]

# Rescale -1 to 1
X = (X / 127.5) - 1.

valid = np.ones((batch_size, 1)) * 0.9
fake = np.zeros((batch_size, 1)) * 0.1

# Lists to keep track of loss
discriminator_losses = []
generator_losses = []

# Begin training
with open('training_log.txt', 'w') as f:
    for epoch in range(num_epochs):
        # Select a random half batch of images
        index = np.random.randint(0, X.shape[0], batch_size)
        images = X[index]

        # Sample noise and generate a half batch of new images
        noise = np.random.normal(0, 1, (batch_size, latent_dimensions))
        generated_images = generator.predict(noise)

        # Train the discriminator on real and fake images, separately
        discm_loss_real = discriminator.train_on_batch(images, valid)
        discm_loss_fake = discriminator.train_on_batch(generated_images, fake)
        discm_loss = 0.5 * np.add(discm_loss_real, discm_loss_fake)

        # Train the generator
        genr_loss = combined_network.train_on_batch(noise, valid)

        # Save losses
        discriminator_losses.append(discm_loss[0])
        generator_losses.append(genr_loss)

        # Display progress for each specified interval
        if epoch % display_interval == 0 or epoch == num_epochs - 1:
            print(f"Epoch: {epoch}, Discriminator Loss: {discm_loss[0]}, Discriminator Accuracy: {100*discm_loss[1]}, Generator Loss: {genr_loss}")
            f.write(f"Epoch: {epoch}, Discriminator Loss: {discm_loss[0]}, Discriminator Accuracy: {100*discm_loss[1]}, Generator Loss: {genr_loss}\n")
            display_images(epoch)

# After the training loop, plot the losses
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(generator_losses,label="G")
plt.plot(discriminator_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
