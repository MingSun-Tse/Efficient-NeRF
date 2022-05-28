echo '------------- After Finetune -------------'
echo chair: && tail -n 2 Experiments/Test__nerfv3.2_W256D88__chair__AfterFT*/log/log.txt | grep TEST
echo drums: && tail -n 2 Experiments/Test__nerfv3.2_W256D88__drums__AfterFT*/log/log.txt | grep TEST
echo ficus: && tail -n 2 Experiments/Test__nerfv3.2_W256D88__ficus__AfterFT*/log/log.txt | grep TEST
echo hotdog: && tail -n 2 Experiments/Test__nerfv3.2_W256D88__hotdog__AfterFT*/log/log.txt | grep TEST
echo lego: && tail -n 2 Experiments/Test__nerfv3.2_W256D88__lego__AfterFT*/log/log.txt | grep TEST
echo materials: && tail -n 2 Experiments/Test__nerfv3.2_W256D88__materials__AfterFT*/log/log.txt | grep TEST
echo mic: && tail -n 2 Experiments/Test__nerfv3.2_W256D88__mic__AfterFT*/log/log.txt | grep TEST
echo ship: && tail -n 2 Experiments/Test__nerfv3.2_W256D88__ship__AfterFT*/log/log.txt | grep TEST


echo '------------- No Finetune -------------'
echo chair: && tail -n 2 Experiments/Test__nerf__chair_S*/log/log.txt | grep TEST
echo drums: && tail -n 2 Experiments/Test__nerf__drums_S*/log/log.txt | grep TEST
echo ficus: && tail -n 2 Experiments/Test__nerf__ficus_S*/log/log.txt | grep TEST
echo hotdog: && tail -n 2 Experiments/Test__nerf__hotdog_S*/log/log.txt | grep TEST
echo lego: && tail -n 2 Experiments/Test__nerf__lego_S*/log/log.txt | grep TEST
echo materials: && tail -n 2 Experiments/Test__nerf__materials_S*/log/log.txt | grep TEST
echo mic: && tail -n 2 Experiments/Test__nerf__mic_S*/log/log.txt | grep TEST
echo ship: && tail -n 2 Experiments/Test__nerf__ship_S*/log/log.txt | grep TEST