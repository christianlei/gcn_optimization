#!/bin/bash

python3 main.py 16 > output16.txt

python3 reddit_main.py 256 > output256.txt

python3 main.py 1024 > output1024.txt

python3 main.py 2048 > output2048.txt

python3 main.py 4096 > output4096.txt

python3 main.py 16 > output4096.txt
