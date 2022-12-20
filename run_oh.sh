SRC_DIR="./ckps/src"
TAR_DIR="./ckps/tar"
TAR_MY_DIR="./ckps/tar_my"
GPU_ID=0
DA="uda"
SEED=2021
SOURCE=1

while(($SOURCE<4))
do 
    python ISKD.py --gpu_id $GPU_ID --seed $SEED --output_src $SRC_DIR --dset office-home --s $SOURCE --da $DA --net_src resnet50 --max_epoch 50
    python ISKD.py --gpu_id $GPU_ID --seed $SEED --output_src $SRC_DIR --dset office-home --s $SOURCE --da $DA --net_src resnet50 --max_epoch 30  --net resnet50 --output $TAR_MY_DIR  --topk 1 --iter 10
    python ISKD_ft.py --da $DA --dset office-home --gpu_id $GPU_ID --net resnet50 --s $SOURCE  --output $TAR_MY_DIR --max_epoch 20 --interval 10 --alpha 0.2 --layer wn --smooth 0.1 --lr 1e-2 --batch_size 64 --lr_gamma 0.0 --seed $SEED --coeff JMDS --warm 1 --weight 0
    (( SOURCE = $SOURCE + 1 ))
done