# Multi-Type Itemset Embedding for Learning Behavior Success

## Desciption
This is the showcase implementation of LearnSUC in python 3.6 using Tensorflow 1.3.


## Requirements
 - numpy==1.14.0
 - tensorflow==1.3.0

Please make sure packages in requirements.txt are properly installed before running.


## Usage
```
python model_mtie.py --itemlist data/itemlist.txt --behaviorlist data/behaviorlist.txt --output embeddings.mode1 --mode 1 --threads 8
```

 - --itemlist: the input file of items list
 - --behaviorlist: the input file of behaviors list
 - --output: the output file of context item embeddings
 - --size: the dimension of the embedding; default is 128
 - --mode: the negative sampling method used; 1 for size-constrained, 2 for type-constrained; default is 1
 - --negative: the number of negative samples used in negative sampling; default is 5
 - --samples: the total number of training samples (*Thousand); default is 1
 - --batch_size: the mini-batch size of the stochastic gradient descent; default is 1
 - --rho: the starting value of the learning rate; default is 0.025
 - --threads: the total number of threads used; default is 10

## Files

In ./data folder, itemlist and behaviorlist for the dataset used in the original paper are included.

## Examples
Two examples are included in train.sh

## Miscellaneous

Authors: Daheng Wang, Meng Jiang, Qingkai Zeng, Zachary Eberhart, Nitesh V. Chawla

For more information, please see the original paper.
