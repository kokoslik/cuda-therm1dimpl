#!/usr/bin/gnuplot -p
unset key
plot for [i=0:5000:500] 'datacpu.dat' index(i) w l,for [i=0:5000:500] 'datagpu.dat' index(i)
