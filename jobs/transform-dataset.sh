DATASET_ORIGINAL=/media/data/datasets/suncg
DATASET_NEW=/media/data/datasets/suncg_compiled
TOOLBOX_PATH=/media/data/repos/SUNCGtoolbox/gaps/bin/x86_64

HOUSES=""
_HOUSES="05fc50487eb1da4939e5a4905b1776ea
00065ecbdd7300d35ef4328ffe871505
cf57359cd8603c3d9149445fb4040d90
31966fdc9f9c87862989fae8ae906295
ff32675f2527275171555259b4a1b3c3
7995c2a93311717a3a9c48d789563590
8b8c1994f3286bfc444a7527ffacde86
775941abe94306edc1b5820e3a992d75
32e53679b33adfcc5a5660b8c758cc96
4383029c98c14177640267bd34ad2f3c
0884337c703e7c25949d3a237101f060
492c5839f8a534a673c92912aedc7b63
a7e248efcdb6040c92ac0cdc3b2351a6
2364b7dcc432c6d6dcc59dba617b5f4b
e3ae3f7b32cf99b29d3c8681ec3be321
f10ce4008da194626f38f937fb9c1a03
e6f24af5f87558d31db17b86fe269cf2
1dba3a1039c6ec1a3c141a1cb0ad0757
b814705bc93d428507a516b866efda28
26e33980e4b4345587d6278460746ec4
5f3f959c7b3e6f091898caa8e828f110
b5bd72478fce2a2dbd1beb1baca48abd
9be4c7bee6c0ba81936ab0e757ab3d61"

BASE_PATH="$(pwd)"
for fn in $HOUSES; do
    echo "Processing house $fn"

    cd "$DATASET_ORIGINAL/house/$fn"
    eval "$TOOLBOX_PATH/scn2scn" house.json house.obj
    eval "python $BASE_PATH/copy-textures.py" "$DATASET_ORIGINAL/house/$fn/house.json" "$DATASET_ORIGINAL/texture" "$DATASET_NEW/texture"
    mkdir -p "$DATASET_NEW/house/$fn"
    cp house.json "$DATASET_NEW/house/$fn/"
    cp house.mtl "$DATASET_NEW/house/$fn/"
    cp house.obj "$DATASET_NEW/house/$fn/"
    echo "Completed house $fn"

    
done

