#!/usr/bin/env bash
logdir=logs/my_TREC
mkdir -p $logdir # logs are dumped here

MODE="implication" # "learn2reweight" / "implication" / "pr_loss" / "label_snorkel" / "gcross" / "gcross_snorkel" / "f_d" / "test_f" / "test_w" / "test_all"
EPOCHS=4
LR=0.0003
CKPT_LOAD_MODE=mru
DROPOUT_KEEP_PROB=0.8
D_PICKLE_NAME="d_processed.p"
VALID_PICKLE_NAME=validation_processed.p
U_PICKLE_NAME="U_processed.p"
GAMMA=0.1
LAMDA=0.1
USE_JOINT_f_w=False #this flag should be True if you need to use output of rule network while doing inference (See eqn6 in the paper)
Q=1
OUTPUT_DIR="$MODE"_"$GAMMA"_"$LAMDA"_"$Q"

# DATA_DIR=../../data/TREC
DATA_DIR=/home/parth/Desktop/SEM6/RnD/Learning-From-Rules/data/TREC
# echo "Hello 2 1"


W_LAYERS="512,512"
F_LAYERS="512,512"

F_D_CLASS_SAMPLING="10,10,10,10,10,10" # while mixing d and U sets, oversample data from d 10 times
                                     # this is because size of d is just 68
                                     # while size of U is ~ 4.6k
                                     # this is done so that in any batch there are enough instances from "d"
                                     # along with instances from U

python3 my_main.py \
  --output_dir="$DATA_DIR"/outputs/"$OUTPUT_DIR" \
  --run_mode=$MODE \
  --checkpoint_load_mode=$CKPT_LOAD_MODE \
  --data_dir=$DATA_DIR \
  --f_d_primary_metric=accuracy \
  --f_d_epochs=$EPOCHS \
  --f_d_U_epochs=$EPOCHS \
  --f_d_batch_size=16 \
  --f_d_U_batch_size=32 \
  --f_d_adam_lr=$LR \
  --f_d_U_adam_lr=$LR \
  --validation_pickle_name=$VALID_PICKLE_NAME \
  --d_pickle_name=$D_PICKLE_NAME \
  --dropout_keep_prob=$DROPOUT_KEEP_PROB \
  --w_layers_str=$W_LAYERS \
  --f_layers_str=$F_LAYERS \
  --f_d_class_sampling_str=$F_D_CLASS_SAMPLING \
  --U_pickle_name=$U_PICKLE_NAME \
  --gamma=$GAMMA \
  --lamda=$LAMDA \
  --early_stopping_p=20 \
  --use_joint_f_w=$USE_JOINT_f_w
