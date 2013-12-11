#! /bin/bash

awk 'BEGIN{srand()} {if ($3 == 1) print $0}' $1 | head -n $2 > small_clients_set
awk 'BEGIN{srand()} {if ($3 == 0) print $0}' $1 | head -n $2 > small_impostors_set
cat small_clients_set small_impostors_set | shuf > small_dataset
rm small_*_set
