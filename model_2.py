import tensorflow as tf
import numpy as np
import os
import sys
import psutil
from osgeo import gdal
import cv2
import matplotlib.pyplot as plt

memory_threshold = 96

# dimensions of images:
# SAR_img = 256x256
# opt_img = 256x256x3
#latent_dim = 10

# Within this architecture, the VAE is employed to generate a low level embedding z of the 
# color field of the Lab-based colorized SAR image. Relying on this low-dimensional 
# latent variable embedding, the MDN is used to generate a multi-modal conditional distribution that models
# the relationship between the gray-level SAR image and the color embedding z. By sampling from this distribution,
# the decoder network of the VAE is able to generate a diverse
# range of colorized representations of the original SAR image

latent_dim = (256, 256,2)
input_dim = (256, 256, 3)


# defining the encoder network architecture
encoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(256,256,3)),
    # tf.keras.layers.Reshape((1,256,256,3)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(4)
])

def reparameterize(z_mean, z_log_var):
    eps = tf.random.normal(shape=z_mean.shape)
    return eps * tf.exp(z_log_var * 0.5) + z_mean

# defining the decoder network architecture
decoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(latent_dim)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(3)
])

class MDN(tf.keras.layers.Layer):
    def __init__(self, output_dim, num_mixtures, **kwargs):
        super(MDN, self).__init__(**kwargs)
        self.num_mixtures = num_mixtures
        self.output_dim = output_dim
        self.mdn_layers = [tf.keras.layers.Dense(3, activation='relu'),
                           tf.keras.layers.Dense(2, activation='relu')]

    def call(self, inputs):
        x = inputs
        for layer in self.mdn_layers:
            x = layer(x)
        return x
    

def lab_image_fusion(SAR_img, opt_img):
    fused_img = (SAR_img + opt_img)/2
    fused_img = tf.cast(fused_img, tf.uint8)
    return fused_img

path_SAR = 'S1_data_256'
path_opt = 'S2_data_256'
def get_data():
  for file1,file2 in zip(os.listdir(path_SAR),os.listdir(path_opt)):
    SAR_img = gdal.Open(os.path.join(path_SAR,file1))
    band_SAR_img = SAR_img.GetRasterBand(1)
    band_SAR_img = band_SAR_img.ReadAsArray()
    band_SAR_img = band_SAR_img.astype(np.uint8)
    band_SAR_img = band_SAR_img.reshape(256,256,1)
    opt_img = gdal.Open(os.path.join(path_opt,file2))
    band_opt_img_R = opt_img.GetRasterBand(1)
    band_opt_img_G = opt_img.GetRasterBand(2)
    band_opt_img_B = opt_img.GetRasterBand(3)
    band_opt_img_R = band_opt_img_R.ReadAsArray()
    band_opt_img_G = band_opt_img_G.ReadAsArray()
    band_opt_img_B = band_opt_img_B.ReadAsArray()
    img = np.dstack((band_opt_img_R, band_opt_img_G, band_opt_img_B))
    img = (img - img.min())/(img.max() - img.min())*255
    #TODO : standardize the image
    # img = (img - np.mean(img))/np.std(img)
    img = img.astype(np.uint8)
    yield band_SAR_img, img

data = tf.data.Dataset.from_generator(get_data, (tf.uint8, tf.uint8))

num_epochs = 100
# define a loss function as ssim loss
# def ssim_loss(y_true, y_pred):
#     mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
#     return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=255))+mse_loss
loss_function = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# First, the VAE is trained on the Lab-based colorized SAR images to \
# learn a low-dimensional embedding (z) of the color field.

for epoch in range(num_epochs):
    for step, (SAR_img, opt_img) in enumerate(data):
        fused_img = lab_image_fusion(SAR_img, opt_img)
        fused_img = tf.reshape(fused_img, (1,256,256,3))
        with tf.GradientTape() as tape:
            encoder_output = encoder(fused_img)
            z_mean, z_log_var= tf.split(encoder_output, num_or_size_splits=2, axis=-1)
            z = reparameterize(z_mean, z_log_var)
            decoder_output = decoder(z)
            loss = loss_function(opt_img, decoder_output[0])
        gradients = tape.gradient(loss, encoder.trainable_variables + decoder.trainable_variables)
        optimizer.apply_gradients(zip(gradients, encoder.trainable_variables + decoder.trainable_variables))
        if step % 10 == 0:
            fig,axes = plt.subplots(1,3)
            axes[0].imshow(opt_img)
            axes[0].set_title('opt_img')
            axes[1].imshow(decoder_output[0])
            axes[1].set_title('decoder_output')
            axes[2].imshow(fused_img[0])
            axes[2].set_title('fused_img')
            plt.show()
        #     # break
        #     print('Epoch: {}, Step: {}, Loss: {}'.format(epoch, step, loss))
    print(f'{epoch=},loss={loss.numpy()}')
sys.exit()
# Second, the MDN is trained on the gray-level SAR image and the latent variable embedding (z) to model the
# relationship between the gray-level SAR image and the color embedding z. By sampling from this distribution,
# the decoder network of the VAE is able to generate a diverse range of colorized representations of the original
# SAR image.
epoch_mdn = 100
loss_function = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
mdn_model = MDN(output_dim=2, num_mixtures=6)
for epoch in range(epoch_mdn):
    for step, (SAR_img, opt_img) in enumerate(data):
        # print(f'Epoch {epoch+1} Step {step+1}')
        fused_img = lab_image_fusion(SAR_img, opt_img)
        fused_img = tf.reshape(fused_img, (1,256,256,3))
        with tf.GradientTape() as tape:
            encoder_output = encoder(fused_img)
            z_mean, z_log_var = tf.split(encoder_output, num_or_size_splits=2, axis=-1)
            z = reparameterize(z_mean, z_log_var)
            mdn_embedding = mdn_model(SAR_img)
            mdn_embedding = tf.reshape(mdn_embedding, (1,256,256,3))
            loss = loss_function(z, mdn_embedding)
        gradients = tape.gradient(loss, mdn_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, mdn_model.trainable_variables))
        if step % 10 == 0:
            print('Epoch: {}, Step: {}, Loss: {}'.format(epoch, step, loss))

# Finally, the decoder network of the VAE is used to generate a colorized representation of the original SAR image
# by sampling from the MDN distribution.
for step, (SAR_img, opt_img) in enumerate(data):
    embedding = mdn_model(SAR_img)
    embedding = tf.reshape(embedding, (1,256,256,3))
    decoder_output = decoder(embedding)
    loss_count = loss_function(opt_img, decoder_output)
    decoder_output = tf.cast(decoder_output, tf.uint8)
    print('Loss: {}'.format(loss_count))
    print(decoder_output[0])
    plt.imshow(decoder_output[0])
    plt.show()
    plt.imshow(opt_img)
    plt.show()
    break