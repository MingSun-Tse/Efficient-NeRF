cd data
unzip NeRF_Data*.zip
cp -r nerf_synthetic/*_v8_* NeRF_Data/nerf_synthetic
rm -rf nerf_synthetic
ln -s NeRF_Data/nerf_synthetic .
ln -s NeRF_Data/nerf_llff_data .
