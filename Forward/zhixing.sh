#!/bin/bash

file='./input/marmousi_270_767.dat'
#file='./input/test1_1_1.dat'
#./Obser ${file} && ./CUT ${file} && ./RTM ${file}
./Obser ${file}  && ./RTM ${file}
