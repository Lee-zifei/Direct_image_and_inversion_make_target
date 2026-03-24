#!/bin/bash

file='./input/inversion_mar_234_663_noise.dat'
#file='./input/test1_1_1.dat'
#./Obser ${file} && ./CUT ${file} && ./RTM ${file}
./Obser ${file}  && ./RTM ${file}
