#!/bin/bash

cd "$(dirname "$0")"
mkdir -p assets
curl -C - -L https://huggingface.co/gpt2/raw/main/vocab.json -o assets/vocab.json
curl -C - -L https://huggingface.co/gpt2/raw/main/tokenizer.json -o assets/tokenizer.json
curl -C - -L https://huggingface.co/gpt2/raw/main/merges.txt -o assets/merges.txt
curl -C - -L https://huggingface.co/gpt2/resolve/main/tf_model.h5 -o assets/tf_model.h5
