#!/bin/bash


for dir in */; do
	cd $dir
	python3 script.py
	cd ..
done

