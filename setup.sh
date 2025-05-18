#!/bin/bash
sudo apt-get update
sudo apt-get install -y $(cat packages.txt)
python -m pip install --upgrade pip
pip install -r requirements.txt
python download_model.py