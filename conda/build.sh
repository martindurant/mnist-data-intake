#!/usr/bin/env bash

python setup.py install --single-version-externally-managed --record=record.txt
mkdir -p $PREFIX/share/intake/mnist
cp $SRC_DIR/intake_mnist/cat.yaml $PREFIX/share/intake/mnist.yaml
