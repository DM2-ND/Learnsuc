#!/bin/bash

make clean
make all

## Default options
# ./learn_suc --itemlist data/itemlist.txt --behaviorlist data/behaviorlist.txt --output learn_suc.txt

## Complete options
./learn_suc --itemlist data/itemlist.txt --behaviorlist data/behaviorlist.txt --output learn_suc.tw.txt --dim 128 --mode 1 --samples 1 --negative 10 --rate 0.025 --threads 8 --typeweights data/typeweights.txt
