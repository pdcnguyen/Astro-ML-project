#! /bin/bash

#if possible, download all the bands files from 80-130 (can be changed) and gal/star coord files beforehand, 
#save them into the 'source/data' folder then run this script. Save time downloading data.

docker build --network host -t sdss .

docker run --gpus all -it sdss