cd data
DIR=NeRF_Data
if test -d "$DIR"; then
    echo "Dir $DIR exists."
    exit
else
    echo "Dir $DIR does not exists. Scp data from Huan's 005 SERVER..."
    sshpass -p "tmp" scp -r wanghuan@155.33.199.5:/home/wanghuan/Projects/Efficient-NeRF/data/NeRF_Data*.zip .
    echo "Scp done. Unzip..."
    for f in NeRF_Data*.zip; do unzip $f; done
    echo "Unzip done."

    mv nerf_synthetic/*_v8_* NeRF_Data/nerf_synthetic
    mv nerf_llff_data/*_v8_* NeRF_Data/nerf_llff_data
    rm -rf nerf_synthetic nerf_llff_data
    ln -s NeRF_Data/nerf_synthetic .
    ln -s NeRF_Data/nerf_llff_data .
    cd ..
fi
