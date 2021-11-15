echo sanmiguel & tail -n $1 Experiments/nerfv3.2__donerf_sanmiguel__S16W181D88*202111[1-4]*/log/log.txt | grep "\[TEST\] Iter $2"
echo pavillon & tail -n $1 Experiments/nerfv3.2__donerf_pavillon__S16W181D88*202111[1-4]*/log/log.txt | grep "\[TEST\] Iter $2"
echo classroom & tail -n $1 Experiments/nerfv3.2__donerf_classroom__S16W181D88*202111[1-4]*/log/log.txt | grep "\[TEST\] Iter $2"
echo bulldozer & tail -n $1 Experiments/nerfv3.2__donerf_bulldozer__S16W181D88*202111[1-4]*/log/log.txt | grep "\[TEST\] Iter $2"
echo forest & tail -n $1 Experiments/nerfv3.2__donerf_forest__S16W181D88*202111[1-4]*/log/log.txt | grep "\[TEST\] Iter $2"
echo barbershop & tail -n $1 Experiments/nerfv3.2__donerf_barbershop__S16W181D88*202111[1-4]*/log/log.txt | grep "\[TEST\] Iter $2"

# usage: sh <this_file> 10000 400000