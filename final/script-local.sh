#! /bin/bash

cd source

python3 -m venv env

source env/bin/activate

pip install -r requirements.txt

python3 prepare.py

python3 main.py