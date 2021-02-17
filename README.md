# Multi-Type Itemset Embedding for Learning Behavior Success

## About
This repository contains the C++ efficiency implementation of LearnSUC model proposed in the paper _Multi Type Itemset Embedding for Learning Behavior Success_ accepted by KDD18.

## Usage
### 1. Make
First, you need to make the executable file. Run command in the project folder:
```
make all
```
In case you want to remove the executable file, run command:
```
make clean
```
If you want more advanced control over options when compiling the program, please look into the `./Makefile` file.

### 2. Execute
Once you have the executable file `./learn_suc`, run command:

```
./learn_suc --itemlist data/itemlist.txt --behaviorlist data/behaviorlist.txt --output learn_suc-m1.txt --mode 1 --threads 8
```

+ --itemlist: The input file of item list. Each line follows format `<item>\t<item type>`
+ --behaviorlist: The input file of behavior list. Each line follows format `<behavior>\t<item 1>[,<item 2>,...]`
+ --output: The output file of item embeddings. First line is header: `<#item>\t<#dimension>`. Then, each line follows format `<item>\t<dimension 1>\t<dimension 2>\t...\t<last dimension>`
+ --dim: The dimension of the embedding; default is 128.
+ --mode: The negative behavior sampling strategy used; 1 for size-constrained, 2 for type-constrained; default is 1.
+ --samples: The total number of training samples in **millions**; default is 1.
+ --negative: The number of negative samples used in negative sampling; default is 10.
+ --rate: The starting value of the learning rate; default is 0.025.
+ --threads: The total number of threads used; default is 8.

Optional:
+ --typeweights: The input file of item type weights. Each line follows format `<item type>\t<item type weight>`. **If this file is provided, it would override the default uniform weights.**
+ --behaviorrates: The input file of behavior success rates. Each line follows format `<behavior>\t<behavior success rates>`. **If this file is provided, it assumes that both positive and negative behaviors can be observed. No further negative behavior samplings would be conducted.**

_Note:_ All `item`, `item type`, `behavior` in input files should be integers. `item type weight` and `behavior success rates` can be float numbers.

## Data
A pre-processed demonstration dataset is included.

+ `./data/behaviorlist.txt`: Academic papers and corresponding authors, conference, keywords, and references.
+ `./data/itemlist.txt`: All items of authors, conferences, keywords, references and their corresponding types.
+ `./data/typeweights.txt`: Arbitrary weight values of author, conference, keyword, reference type.

## Examples
Other examples are provided in the `./train.sh` file.

## Miscellaneous
**Authors**: Daheng Wang, Meng Jiang, Qingkai Zeng, Zachary Eberhart, Nitesh V. Chawla\
**Address**: University of Notre Dame, Notre Dame, Indiana, 46556, USA\
**Contact**: {dwang8,mjiang2,qzeng,zeberhar,nchawla}@nd.edu

If you find this code package to be useful, please consider cite the original paper: Multi-Type Itemset Embedding for Learning Behavior Success.
