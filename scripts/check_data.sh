cd data/nerf_synthetic
for d in *_v8*/; do echo $d && ls $d | wc -l; done

cd ../nerf_llff_data
for d in *_v8*/; do echo $d && ls $d | wc -l; done