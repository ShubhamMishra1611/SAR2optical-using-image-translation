import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np
from osgeo import gdal
import sys
from itertools import islice

class MDN(tf.keras.Model):
  def __init__(self):
    super(MDN, self).__init__()
    self.reshape = tf.keras.layers.Reshape((256,256,1))
    self.conv1 = tf.keras.layers.Conv2D(32, kernel_size= (3, 3), activation='relu')
    self.conv2 = tf.keras.layers.Conv2D(64, kernel_size= (3, 3), activation='relu')
    self.flatten = tf.keras.layers.Flatten()
    self.dense = tf.keras.layers.Dense(6)

  def call(self, SAR_img):
    SAR_img = tf.cast(SAR_img, tf.float32)
    x = tf.expand_dims(SAR_img, 0)
    x = self.reshape(x)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.flatten(x)
    x = self.dense(x)
    return x

class Encoder(tf.keras.Model):
  def __init__(self):
    super(Encoder,self).__init__()
    self.reshape = tf.keras.layers.Reshape((1,256,256,3))
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(256, activation='relu')
    self.dense2 = tf.keras.layers.Dense(64, activation='relu')
    self.dense3 = tf.keras.layers.Dense(6, activation='relu')

  def call(self, inputs):
    x,rgb_dist = inputs
    x = tf.cast(x, tf.float32)
    x = tf.expand_dims(x, 0)
    x = self.reshape(x)
    x = self.flatten(x)
    x = tf.concat([x,rgb_dist], axis=1)
    x = self.dense1(x)
    x = self.dense2(x)
    x = self.dense3(x)
    return x

class Decoder(tf.keras.Model):
  def __init__(self):
    super(Decoder,self).__init__()
    self.dense1 = tf.keras.layers.Dense(6, activation='relu')
    self.dense2 = tf.keras.layers.Dense(256, activation='relu')
    self.dense3 = tf.keras.layers.Dense(256*256*3, activation='sigmoid')
    self.reshape = tf.keras.layers.Reshape((256,256,1))

  def call(self, inputs):
    inputs = tf.cast(inputs, tf.float32)
    x = self.dense1(inputs)
    x = self.dense2(x)
    x = self.dense3(x)
    x = tf.reshape(x, (-1, 256, 256, 3))
    return x

class Model(tf.keras.Model):
  def __init__(self):
    super(Model,self).__init__()
    self.mdn = MDN()
    self.encoder = Encoder()
    self.decoder = Decoder()

  def call(self, SAR_img, opt_img):
    rgb_dist = self.mdn(SAR_img)
    latent = self.encoder([opt_img, rgb_dist])
    recon = self.decoder(latent)
    return recon



path_SAR = 'S1_data_256'
path_opt = 'S2_data_256'
def get_data():
  for file1,file2 in zip(os.listdir(path_SAR),os.listdir(path_opt)):
    SAR_img = gdal.Open(os.path.join(path_SAR,file1))
    band_SAR_img = SAR_img.GetRasterBand(1)
    band_SAR_img = band_SAR_img.ReadAsArray()
    band_SAR_img = band_SAR_img.astype(np.uint8)
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
    img = img.astype(np.uint8)
    yield band_SAR_img, img
  
  

# Create a dataset from your SAR and opt images
data = tf.data.Dataset.from_generator(get_data, (tf.uint8, tf.uint8))


# Define the number of steps per epoch and the number of total epochs
num_steps = 50
num_epochs = 10
model = Model()

# Create an optimizer and a loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.MeanSquaredError()
save_model_path = 'weight'
cck_pnt = tf.train.Checkpoint(model=model, optimizer=optimizer)
manager = tf.train.CheckpointManager(cck_pnt, './tf_ckpts', max_to_keep=3)
# Iterate through the number of epochs
for epoch in range(num_epochs):
    print("Epoch: {}".format(epoch))
    for step, (SAR_img, opt_img) in enumerate(data.take(num_steps)):
        with tf.GradientTape() as tape:
            recon = model(SAR_img, opt_img)
            loss = loss_fn(opt_img, recon[0])
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            print("Step: {}, Loss: {}".format(step, loss.numpy()))
        model.mdn.save_weights(os.path.join(save_model_path, 'mdn_weights.h5'))
        model.decoder.save_weights(os.path.join(save_model_path, 'decoder.h5'))

# save the weights
model.save_weights('model_weights.h5')

############################################################################
#                              Testing                                    #
############################################################################

dummy_input_1 = np.zeros((256,256))
dummy_input_2 = np.zeros(shape=(1,6))
decoder = Decoder()
decoder(dummy_input_2)
decoder.load_weights(r'weight\decoder.h5')
MDN = MDN()
MDN(dummy_input_1)
MDN.load_weights(r'weight\mdn_weights.h5')

SAR_img = os.listdir(path_SAR)[0]
SAR_img = gdal.Open(os.path.join(path_SAR,SAR_img))
band_SAR_img = SAR_img.GetRasterBand(1)
band_SAR_img = band_SAR_img.ReadAsArray()
band_SAR_img = band_SAR_img.astype(np.uint8)
opt_img = os.listdir(path_opt)[0]
opt_img = gdal.Open(os.path.join(path_opt,opt_img))
band_opt_img_R = opt_img.GetRasterBand(1)
band_opt_img_G = opt_img.GetRasterBand(2)
band_opt_img_B = opt_img.GetRasterBand(3)
band_opt_img_R = band_opt_img_R.ReadAsArray()
band_opt_img_G = band_opt_img_G.ReadAsArray()
band_opt_img_B = band_opt_img_B.ReadAsArray()
img = np.dstack((band_opt_img_R, band_opt_img_G, band_opt_img_B))
img = (img - img.min())/(img.max() - img.min())*255
img = img.astype(np.uint8)


rgb_dist = MDN(band_SAR_img)
mu = tf.reduce_mean(rgb_dist, axis=0)
sigma = tf.math.reduce_std(rgb_dist, axis=0)

random_sample = mu+sigma* tf.random.normal(tf.shape(mu), mean=0., stddev=1., dtype=tf.float32)
random_sample = tf.cast(random_sample, tf.uint8)
random_sample = tf.reshape(random_sample, (1,6))
recon = decoder(random_sample)

mse_loss = tf.keras.losses.MeanSquaredError()
loss = mse_loss(img, recon[0])
print("loss of the model: {}".format(loss.numpy()))

plt.imshow(recon[0])



