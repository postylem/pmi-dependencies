#!/bin/sh
mkdir XLNet-base
cd XLNet-base
wget -O pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-pytorch_model.bin
wget -O config.json https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-config.json
wget -O spiece.model https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-spiece.model
~
