#!/bin/bash

dvc exp run --queue -S train.architecture=CNNAutoencoderBN
dvc exp run --queue -S train.architecture=CNNAutoencoderCAE
dvc exp run --queue -S train.architecture=CNNAutoencoderCAEBN
dvc exp run --queue -S train.architecture=CNNAutoencoderCAEBNLeakyRelu
dvc exp run --queue -S train.architecture=CNNAutoencoderCAEBNL2Emb

dvc exp run --run-all