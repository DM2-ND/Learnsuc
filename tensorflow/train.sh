#!/bin/bash

### Make sure packges in requirements.txt are installed
### If you use virtualenv:
#source venv/bin/activate

python model_mtie.py --itemlist data/itemlist.txt --behaviorlist data/behaviorlist.txt --output embeddings.mode1 --mode 1 --threads 8

# python model_mtie.py --itemlist data/itemlist.txt --behaviorlist data/behaviorlist.txt --output embeddings.mode2 --mode 2 --threads 8
