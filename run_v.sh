SRC_DIR="./ckps/src"
TAR_DIR="./ckps/tar"

python ISKD.py --gpu_id 0 --seed 2021 --output_src $SRC_DIR --dset VISDA-C --s 0 --da uda --net_src resnet101 --max_epoch 10

python ISKD.py --gpu_id 0 --seed 2021 --output_src $SRC_DIR --dset VISDA-C --s 0 --da uda --net_src resnet101 --max_epoch 15 --net resnet101 --output $TAR_DIR --distill --topk 1
