#!/bin/bash

python3 graphsaint_data_train_model.py 16 amazon > outputamazon16.txt

python3 graphsaint_data_train_model.py 256 amazon > outputamazon256.txt

# python3 yelp_train_model.py 512 > output512.txt

# python3 yelp_train_model.py 768 > output768.txt

# python3 yelp_train_model.py 1024 > output1024.txt

# python3 yelp_train_model.py 2048 > output2048.txt

# python3 yelp_train_model.py 4096 > output4096.txt