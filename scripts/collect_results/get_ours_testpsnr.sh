echo room & tail -n $1 Experiments/nerfv3.2__room__*W256D88*SERVER-202111*/log/log.txt | grep "\[TEST\] Iter $2"
echo fern & tail -n $1 Experiments/nerfv3.2__fern__*W256D88*SERVER-202111*/log/log.txt | grep "\[TEST\] Iter $2"
echo leaves & tail -n $1 Experiments/nerfv3.2__leaves__*W256D88*SERVER-202111*/log/log.txt | grep "\[TEST\] Iter $2"
echo fortress & tail -n $1 Experiments/nerfv3.2__fortress__*W256D88*SERVER-202111*/log/log.txt | grep "\[TEST\] Iter $2"
echo orchids & tail -n $1 Experiments/nerfv3.2__orchids__*W256D88*SERVER-202111*/log/log.txt | grep "\[TEST\] Iter $2"
echo flower & tail -n $1 Experiments/nerfv3.2__flower__*W256D88*SERVER-202111*/log/log.txt | grep "\[TEST\] Iter $2"
echo trex & tail -n $1 Experiments/nerfv3.2__trex__*W256D88*SERVER-202111*/log/log.txt | grep "\[TEST\] Iter $2"
echo horns & tail -n $1 Experiments/nerfv3.2__horns__*W256D88*SERVER-202111*/log/log.txt | grep "\[TEST\] Iter $2"

echo chair & tail -n $1 Experiments/nerfv3.2__chair__*W256D88*SERVER-202111*/log/log.txt | grep "\[TEST\] Iter $2"
echo drums & tail -n $1 Experiments/nerfv3.2__drums__*W256D88*SERVER-202111*/log/log.txt | grep "\[TEST\] Iter $2"
echo ficus & tail -n $1 Experiments/nerfv3.2__ficus__*W256D88*SERVER-202111*/log/log.txt | grep "\[TEST\] Iter $2"
echo hotdog & tail -n $1 Experiments/nerfv3.2__hotdog__*W256D88*SERVER-202111*/log/log.txt | grep "\[TEST\] Iter $2"
echo lego & tail -n $1 Experiments/nerfv3.2__lego__*W256D88*SERVER-202111*/log/log.txt | grep "\[TEST\] Iter $2"
echo materials & tail -n $1 Experiments/nerfv3.2__materials__*W256D88*SERVER-202111*/log/log.txt | grep "\[TEST\] Iter $2"
echo mic & tail -n $1 Experiments/nerfv3.2__mic__*W256D88*SERVER-202111*/log/log.txt | grep "\[TEST\] Iter $2"
echo ship & tail -n $1 Experiments/nerfv3.2__ship__*W256D88*SERVER-202111*/log/log.txt | grep "\[TEST\] Iter $2"

# usage: sh <this_file> 10000 800000