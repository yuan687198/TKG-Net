## Getting Started
1. Move your data to __data/ dir__.

## Tuning for own Corpus
1. Assuming are done with Point 1 under __Getting Started__
```
2. Run python3 yuan_train_int.py --data_file <filename> --epoch <number_of_epochs> --warmup <warmup_steps> --model_name <model_name> --max_len <max_seq_length> --learning_rate <learning_rate> --batch <batch_size>
```
## Generating Text
```
1. python3 generate.py --model_name <model_name> --sentences <number_of_sentences> --label <class_of_training_data>
```

_* It is recommended that you tune the parameters for your task. Not doing so may result in choosing default parameters and eventually giving sub-optimal performace._

## Training Data
1. You can download the training data from [here](https://pan.baidu.com/s/18-FFI9nEQwwPOC-imjgY3Q)(Extraction code: dg9j)
2. You can download gpt model from [here](https://pan.baidu.com/s/14B_fjE7kMOXnaPN7cLnyCg )(Extraction code: 1sp2). We provide the model of gpt-medium. You can also try gpt-large, gpt-xl, etc.



