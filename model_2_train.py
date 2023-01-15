import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
from osgeo import gdal
class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer1 = keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')
        self.layer2 = keras.layers.Conv2D(128, (3,3), strides=(2,2), padding='same', activation='relu')
        self.layer3 = keras.layers.Conv2D(256, (3,3), strides=(2,2), padding='same', activation='relu')
        self.layer4 = keras.layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', activation='relu')
        self.layer5 = keras.layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', activation='relu')
        self.layer6 = keras.layers.Conv2D(3, (3,3), padding='same', activation='tanh')

    def call(self, inputs, training=None):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x

class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = keras.layers.Conv2D(64, (3,3), strides=(2,2), padding='same', activation='relu')
        self.layer2 = keras.layers.Conv2D(128, (3,3), strides=(2,2), padding='same', activation='relu')
        self.layer3 = keras.layers.Conv2D(256, (3,3), strides=(2,2), padding='same', activation='relu')
        self.layer4 = keras.layers.Flatten()
        self.layer5 = keras.layers.Dense(1)

    def call(self, inputs, training=None):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

# Define the generator and discriminator networks
generator = Generator()
discriminator = Discriminator()

# Define optimizers for the generator and discriminator
gen_optimizer = keras.optimizers.Adam(1e-4)
dis_optimizer = keras.optimizers.Adam(1e-4)


generator_loss = keras.losses.BinaryCrossentropy(from_logits=True)
discriminator_loss = keras.losses.BinaryCrossentropy(from_logits=True)
num_epochs = 2

def dataset():
    for SAR, RGB in zip(os.listdir('S1_data_256'), os.listdir('S2_data_256')):
        SAR = gdal.Open('S1_data_256/' + SAR)
        SAR = SAR.GetRasterBand(1)
        SAR = SAR.ReadAsArray()
        RGB = gdal.Open('S2_data_256/' + RGB)
        RGB_1 = RGB.GetRasterBand(1)
        RGB_2 = RGB.GetRasterBand(2)
        RGB_3 = RGB.GetRasterBand(3)
        RGB = np.dstack((RGB_1.ReadAsArray(), RGB_2.ReadAsArray(), RGB_3.ReadAsArray()))
        yield SAR, RGB

# Define the training loop
for epoch in range(num_epochs):
    print('Epoch: ', epoch)
    for inputs, targets in dataset():
        # Generate output image from the input image
        generated_images = generator(inputs)

        # Train the discriminator
        with tf.GradientTape() as dis_tape:
            dis_real_output = discriminator(targets)
            dis_fake_output = discriminator(generated_images)
            dis_loss = discriminator_loss(dis_real_output, dis_fake_output)
        dis_gradients = dis_tape.gradient(dis_loss, discriminator.trainable_variables)
        dis_optimizer.apply_gradients(zip(dis_gradients, discriminator.trainable_variables))

        # Train the generator
        with tf.GradientTape() as gen_tape:
            gen_output = generator(inputs)
            gen_loss = generator_loss(dis_fake_output, gen_output)
        gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    print('Generator loss: ', gen_loss.numpy())
    print('Discriminator loss: ', dis_loss.numpy())

# save the model
generator.save('generator.h5')
discriminator.save('discriminator.h5')

# test the model
import matplotlib.pyplot as plt
import cv2
import numpy as np

# load the model
generator = keras.models.load_model('generator.h5')
discriminator = keras.models.load_model('discriminator.h5')

# load the test image
for SAR,RGB in zip(os.listdir('S1_test_256'), os.listdir('S2_test_256')):
    SAR = gdal.Open('S1_test_256/' + SAR)
    SAR = SAR.ReadAsArray()
    SAR = SAR.reshape(1, 256, 256, 1)
    RGB = gdal.Open('S2_test_256/' + RGB)
    RGB = RGB.ReadAsArray()
    RGB = RGB.reshape(1, 256, 256, 3)
    break

# generate the output image
generated_images = generator(SAR)

# plot the output image
plt.imshow(generated_images[0])
plt.show()

# plot the input image
plt.imshow(RGB[0])
plt.show()

