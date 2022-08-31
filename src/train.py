from platform import architecture
import dvc.api
import tensorflow as tf
import numpy as np
import random
from pathlib import Path

from loaders.dataset import DataGenerator
from models import (
    DenseAutoencoder,
    CNNAutoencoder,
    CNNAutoencoderBN, 
    CNNAutoencoderCAE,
    CNNAutoencoderCAEBN,
    CNNAutoencoderCAEBNLeakyRelu,
    CNNAutoencoderCAEBNL2Emb,
    )
from callbacks import CSVTimeHistory

## Params
params=dvc.api.params_show()

# train
LATENT_DIM=params['train']['latent_dim']
EPOCHS=params['train']['epochs']
BATCH_SIZE=params['train']['batch_size']
ARCHITECTURE=params['train']['architecture']
PATIENTE_EARLY_STOPPING=params['train']['patiente_early_stopping']
PATIENTE_LEARNING_RATE=params['train']['patiente_learning_rate']
TRAIN_SIZE=params['train']['train_size']

# dataset
PATH_FCGR=params['fcgr_from_msa']['path_fcgr']

## Datasets
# paths to fcgr 
list_npy = list(Path(PATH_FCGR).rglob('*.npy'))
random.shuffle(list_npy)
n_train=round(len(list_npy)*TRAIN_SIZE) # number of fcgr in the train set
preprocessing = lambda x: x / x.max() 

# dataset train
ds_train = DataGenerator(
    list_paths=list_npy[:n_train],
    batch_size=BATCH_SIZE,
    shuffle=True,
    preprocessing=preprocessing
)

# dataset validation
ds_val = DataGenerator(
    list_paths=list_npy[n_train:],
    batch_size=BATCH_SIZE,
    shuffle=False,
    preprocessing=preprocessing
)

# - Callbacks
# checkpoint: save best weights
Path('data/train/checkpoints').mkdir(exist_ok=True, parents=True)
cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    # filepath='../data/train/checkpoints/weights-{epoch:02d}-{val_loss:.3f}.hdf5',
    filepath=f'data/train/checkpoints/weights-{ARCHITECTURE}.hdf5',
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)

# reduce learning rate
cb_reducelr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    factor=0.1,
    patience=PATIENTE_LEARNING_RATE,
    verbose=1,
    min_lr=0.00001
)

# stop training if
cb_earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    mode='min',
    min_delta=0.001,
    patience=PATIENTE_EARLY_STOPPING,
    verbose=1
)

# save history of training
Path('data/train').mkdir(exist_ok=True, parents=True)
cb_csvlogger = tf.keras.callbacks.CSVLogger(
    filename='data/train/training_log.csv',
    separator='\t',
    append=False
)

# save time by epoch
cb_csvtime = CSVTimeHistory(
    filename='data/train/time_log.csv',
    separator='\t',
    append=False
)

autoencoder=eval(f"{ARCHITECTURE}(LATENT_DIM)")
autoencoder.compile(optimizer='adam', loss="binary_crossentropy")
autoencoder.fit(
    ds_train, 
    validation_data=ds_val, 
    epochs=EPOCHS,
    callbacks=[
        cb_checkpoint,
        cb_reducelr,
        cb_earlystop,
        cb_csvlogger,
        cb_csvtime
        ]
)