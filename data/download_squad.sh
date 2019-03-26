#!/bin/sh
if [ ! -f data/raw/train-v1.1.json ]; then
    wget 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json' -P data/raw
    jq '.' < data/raw/train-v1.1.json > data/raw/train-v1.1-processed.json
    rm data/raw/train-v1.1.json
    mv data/raw/train-v1.1-processed.json data/raw/train-v1.1.json
fi

if [ ! -f data/raw/dev-v1.1.json ]; then
    wget 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json' -P data/raw
    jq '.' < data/raw/dev-v1.1.json > data/raw/dev-v1.1-processed.json
    rm data/raw/dev-v1.1.json
    mv data/raw/dev-v1.1-processed.json data/raw/dev-v1.1.json
fi

if [ ! -f data/raw/train-v2.0.json ]; then
    wget 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json' -P data/raw
    jq '.' < data/raw/train-v2.0.json > data/raw/train-v2.0-processed.json
    rm data/raw/train-v2.0.json
    mv data/raw/train-v2.0-processed.json data/raw/train-v2.0.json
fi

if [ ! -f data/raw/dev-v2.0.json ]; then
    wget 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json' -P data/raw
    jq '.' < data/raw/dev-v2.0.json > data/raw/dev-v2.0-processed.json
    rm data/raw/dev-v2.0.json
    mv data/raw/dev-v2.0-processed.json data/raw/dev-v2.0.json
fi
