import tensorflow as tf
# TODO: to functional API
class DenseAutoencoder(tf.keras.models.Model):

  def __init__(self,latent_dim,):
    super(DenseAutoencoder, self).__init__()
    self.latent_dim = latent_dim   
    
    # encoder
    self.encoder = tf.keras.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(latent_dim, activation='relu'),
    ])

    # decoder
    self.decoder = tf.keras.Sequential([
      tf.keras.layers.Dense(4096, activation='sigmoid'),
      tf.keras.layers.Reshape((64,64))
    ])

  def call(self, x):

    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded