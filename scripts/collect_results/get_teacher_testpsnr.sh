# SERVER005
tte DONERF_nerf__sanmiguel_S
tte DONERF_nerf__pavillon_S
tte DONERF_nerf__classroom_S
tte DONERF_nerf__bulldozer_S
tte DONERF_nerf__forest_S
tte DONERF_nerf__barbershop_S

echo chair & tail -n $1 Experiments/nerf__chair_SERVER*2021*/log/log.txt | grep "\[TEST\] Iter $2"
echo drums & tail -n $1 Experiments/nerf__drums_SERVER*2021*/log/log.txt | grep "\[TEST\] Iter $2"
echo ficus & tail -n $1 Experiments/nerf__ficus_SERVER*2021*/log/log.txt | grep "\[TEST\] Iter $2"
echo hotdog & tail -n $1 Experiments/nerf__hotdog_SERVER*2021*/log/log.txt | grep "\[TEST\] Iter $2"
echo lego & tail -n $1 Experiments/nerf__lego_SERVER*2021*/log/log.txt | grep "\[TEST\] Iter $2"
echo materials & tail -n $1 Experiments/nerf__materials_SERVER*2021*/log/log.txt | grep "\[TEST\] Iter $2"
echo mic & tail -n $1 Experiments/nerf__mic_SERVER*2021*/log/log.txt | grep "\[TEST\] Iter $2"
echo ship & tail -n $1 Experiments/nerf__ship_SERVER*2021*/log/log.txt | grep "\[TEST\] Iter $2"

echo chair: & tail -n $1 Experiments/nerf__chair_SERVER-*202205*800x800*/log/log.txt | grep "\[TEST\] Iter $2"
echo drums: & tail -n $1 Experiments/nerf__drums_SERVER-*202205*800x800*/log/log.txt | grep "\[TEST\] Iter $2"
echo ficus: & tail -n $1 Experiments/nerf__ficus_SERVER-*202205*800x800*/log/log.txt | grep "\[TEST\] Iter $2"
echo hotdog: & tail -n $1 Experiments/nerf__hotdog_SERVER-*202205*800x800*/log/log.txt | grep "\[TEST\] Iter $2"
echo lego: & tail -n $1 Experiments/nerf__lego_SERVER-*202205*800x800*/log/log.txt | grep "\[TEST\] Iter $2"
echo materials: & tail -n $1 Experiments/nerf__materials_SERVER-*202205*800x800*/log/log.txt | grep "\[TEST\] Iter $2"
echo mic: & tail -n $1 Experiments/nerf__mic_SERVER-*202205*800x800*/log/log.txt | grep "\[TEST\] Iter $2"
echo ship: & tail -n $1 Experiments/nerf__ship_SERVER-*202205*800x800*/log/log.txt | grep "\[TEST\] Iter $2"