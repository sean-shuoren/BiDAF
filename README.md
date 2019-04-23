# A re-implementation of Bidirectional Attention Flow for Machine Comprehension (BiDAF).

### Performance
|                |  EM  |  F1  |
|:--------------:|:----:|:----:|
| Original BiDAF | 67.7 | 77.3 |
|   Replication  | 61.0 | 72.7 |

I had to decrease the batch size from 60 to 12 when training on my machine. This hurts perforance.

### Environment
```
Python 3.6.7
Pytorch 1.0.1
TorchText 0.4.0
TensorBoardX
jq  # formating json file
```

### Usage
```
$ ./data/download_squad.sh  # download datasets
$ python train.py

optional arguments:
  -h, --help            show this help message and exit
  --char-emb-dim CHAR_EMB_DIM (default: 8)
  --char-channel-width CHAR_CHANNEL_WIDTH (default: 5)
  --char-channel-num CHAR_CHANNEL_NUM (default: 100)
  --dev-batch-size DEV_BATCH_SIZE (default: 60)
  --disable-c2q DISABLE_C2Q (default: False)
  --disable-q2c DISABLE_Q2C (default: False)
  --dropout DROPOUT (default: 0.2)
  --epoch EPOCH (default: 12)
  --gpu GPU (default: 0)
  --hidden-size HIDDEN_SIZE (default: 100)
  --learning-rate LEARNING_RATE (default: 0.5)
  --moving-average-decay MOVING_AVERAGE_DECAY (default: 0.999)
  --squad-version SQUAD_VERSION (default: "1.1")
  --train-batch-size TRAIN_BATCH_SIZE (default: 60)
  --word-vec-dim WORD_VEC_DIM (default: 100)
  --validation-freq VALIDATION_FREQ (default: 500)
```
