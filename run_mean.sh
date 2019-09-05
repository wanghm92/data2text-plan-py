#!/usr/bin/env bash
MYFOLDER=${PWD##*/}
##################################################################################################
PY3='/mnt/cephfs2/nlp/hongmin.wang/anaconda3/envs/pt11py37/bin/python'
cd /mnt/cephfs2/nlp/hongmin.wang/table2text/$MYFOLDER
echo "current working directory:"
pwd

# ###################################################################################################
echo "Checking GPU availabilities"
tot_gpus=`nvidia-smi -q|grep "Attached GPUs"|awk '{print $NF-1}'`
gpu_ids=`nvidia-smi -q |grep -e "Minor" -e "Process ID" | grep -B 1 Process|grep Minor|awk '{print $NF}'`
use_gpu=-1
for i in `seq 0 ${tot_gpus}` ; do
    if [[ ! "${gpu_ids}" =~ "${i}" ]]; then
        use_gpu=$i
        break
    fi
done
if [ $use_gpu = -1 ]; then
    echo "Error! There is no GPU card available"
    exit 1
fi
echo use gpu card $use_gpu

export CUDA_VISIBLE_DEVICES=$use_gpu
##################################################################################################

DATA=aaai
VER=template
# VER=ncpcc
MID=.addsp.rule
# MID=
BASE=/mnt/cephfs2/nlp/hongmin.wang/table2text/boxscore-data/scripts_$DATA/new_dataset/new_$VER
ENCODER=mean

echo $DATA
echo $VER
echo $MID
echo $BASE

TRAIN_SRC1=$BASE/train/src_train.norm.trim$MID.ncp.full.txt
TRAIN_TGT1=$BASE/train/train_content_plan_ids$MID.ncp.full.txt
TRAIN_SRC2=$BASE/train/train_content_plan_tks$MID.txt
TRAIN_TGT2=$BASE/train/tgt_train.norm.mwe.trim.txt
TRAIN_PTR=$BASE/train/train_ptrs$MID.txt

wc $TRAIN_SRC1 $TRAIN_TGT1 $TRAIN_SRC2 $TRAIN_TGT2 $TRAIN_PTR

VALID_SRC1=$BASE/valid/src_valid.norm.trim$MID.ncp.full.txt
VALID_TGT1=$BASE/valid/valid_content_plan_ids$MID.ncp.full.txt
VALID_SRC2=$BASE/valid/valid_content_plan_tks$MID.txt
VALID_TGT2=$BASE/valid/tgt_valid.norm.mwe.trim.txt

wc $VALID_SRC1 $VALID_TGT1 $VALID_SRC2 $VALID_TGT2

TEST_SRC1=$BASE/test/src_test.norm.trim$MID.ncp.full.txt
TEST_TGT1=$BASE/test/test_content_plan_ids$MID.ncp.full.txt
TEST_SRC2=$BASE/test/test_content_plan_tks$MID.txt
TEST_TGT2=$BASE/test/tgt_test.norm.mwe.trim.txt

wc $TEST_SRC1 $TEST_TGT1 $TEST_SRC2 $TEST_TGT2

###################################################################################################
SUFFIX=new
PREPATH=$ENCODER\_$DATA\_$SUFFIX
echo $PREPATH
PREPRO=/mnt/cephfs2/nlp/hongmin.wang/table2text/boxscore-data/scripts_$DATA/new_dataset/new_$VER/pt_data/$PREPATH
mkdir -p $PREPRO
ls -l $PREPRO

DIM=256
BAT=16
DEC=2

IDENTIFIER=$1
printf "IDENTIFIER = $IDENTIFIER \n"
OUTPUT=/mnt/cephfs2/nlp/hongmin.wang/table2text/outputs/$MYFOLDER/$DATA\_models/$ENCODER/$IDENTIFIER
mkdir -p $OUTPUT
printf "Saving/Loading models to/from: $OUTPUT\n"

SUM_OUT=/mnt/cephfs2/nlp/hongmin.wang/table2text/outputs/$MYFOLDER/$DATA\_outputs/$ENCODER/$IDENTIFIER
mkdir -p $SUM_OUT
printf "Saving/Loading summary outputs to/from: $SUM_OUT\n"

#####################################################################################################
# echo "run preprocessing"
# python preprocess.py -train_src1 $TRAIN_SRC1 -train_tgt1 $TRAIN_TGT1 -train_src2 $TRAIN_SRC2 -train_tgt2 $TRAIN_TGT2 -valid_src1 $VALID_SRC1 -valid_tgt1 $VALID_TGT1 -valid_src2 $VALID_SRC2 -valid_tgt2 $VALID_TGT2 -save_data $PREPRO/roto-$PREPATH -src_seq_length 1000 -tgt_seq_length 1000 -dynamic_dict -train_ptr $TRAIN_PTR

####################################################################################################
echo "run training"
$PY3 train.py -data $PREPRO/roto-$PREPATH -save_model $OUTPUT/roto -encoder_type1 $ENCODER -decoder_type1 pointer -enc_layers1 1 -dec_layers1 1 -encoder_type2 brnn -decoder_type2 rnn -enc_layers2 $DEC -dec_layers2 $DEC -batch_size $BAT -feat_merge mlp -feat_vec_size $DIM -word_vec_size $DIM -rnn_size $DIM -seed 1234 -epochs 50 -optim adagrad -learning_rate 0.15 -adagrad_accumulator_init 0.1 -report_every 50 -copy_attn -truncated_decoder 100 -max_generator_batches 1000 -gpuid 0 -attn_hidden 64 -reuse_copy_attn -start_decay_at 4 -learning_rate_decay 0.97 -valid_batch_size $BAT -tensorboard -tensorboard_log_dir $OUTPUT/events # -stage2_input_type embedding

# echo "run training Adam Optimizer"
# $PY3 train.py -data $PREPRO/roto-$PREPATH -save_model $OUTPUT/roto -encoder_type1 $ENCODER -decoder_type1 pointer -enc_layers1 1 -dec_layers1 1 -encoder_type2 brnn -decoder_type2 rnn -enc_layers2 $DEC -dec_layers2 $DEC -batch_size $BAT -feat_merge mlp -feat_vec_size $DIM -word_vec_size $DIM -rnn_size $DIM -seed 1234 -epochs 50 -optim adam -learning_rate 0.001 -start_decay_at 4 -learning_rate_decay 0.95 -report_every 50 -copy_attn -truncated_decoder 100 -max_generator_batches 1000 -gpuid 0 -attn_hidden 64 -reuse_copy_attn -valid_batch_size $BAT -tensorboard -tensorboard_log_dir $OUTPUT/events


# MIN=56
# MAX=100
# #################################################################################################
# echo " ****** Evaluation ****** "
# for EPOCH in $(seq 10 50)
# do
#     for MODEL1 in $(ls $OUTPUT/roto_stage1*_e$EPOCH.pt)
#     do

#         for MODEL2 in $(ls $OUTPUT/roto_stage2*_e$EPOCH.pt)
#         do

#         echo "--"
#         echo $MODEL1
#         echo $MODEL2

#         printf "\n--"
#         echo " ****** STAGE 1 ****** \n"
#         echo "input src: $VALID_SRC1"
#         echo "saving to: $SUM_OUT/roto_stage1_$IDENTIFIER.e$EPOCH.valid.min$MIN.txt"
#         $PY3 translate.py -model $MODEL1 -src1 $VALID_SRC1 -edges $VALID_EDGE -output $SUM_OUT/roto_stage1_$IDENTIFIER.e$EPOCH.valid.min$MIN.txt -batch_size $BAT -max_length $MAX -gpu 0 -min_length $MIN -stage1

#         printf "\n ****** STAGE 1 create_content_plan_from_index ****** \n"
#         $PY3 create_content_plan_from_index.py $VALID_SRC1 $SUM_OUT/roto_stage1_$IDENTIFIER.e$EPOCH.valid.min$MIN.txt $SUM_OUT/roto_stage1_$IDENTIFIER.e$EPOCH.h5-tuples.valid.txt $SUM_OUT/roto_stage1_inter_$IDENTIFIER.e$EPOCH.valid.min$MIN.txt

#         cd /mnt/cephfs2/nlp/hongmin.wang/table2text/boxscore-data/scripts_aaai/evaluate
#         echo " ****** STAGE 1 RG CS CO ****** "
#         $PY3 eval_content_plan.py --path $BASE --dataset valid --plan $SUM_OUT/roto_stage1_inter_$IDENTIFIER.e$EPOCH.valid.min$MIN.txt
#         # cd /mnt/cephfs2/nlp/hongmin.wang/table2text/boxscore-data/scripts_aaai/process
#         # $PY3 checkdups.py --plan $SUM_OUT/roto_stage1_inter_$IDENTIFIER.e$EPOCH.valid.min$MIN.txt
#         cd /mnt/cephfs2/nlp/hongmin.wang/table2text/$MYFOLDER

#         ##################################################################################################
#         printf "\n ****** STAGE 2 ****** \n"
#         $PY3 translate.py -model $MODEL1 -model2 $MODEL2 -src1 $VALID_SRC1 -edges $VALID_EDGE -tgt1 $SUM_OUT/roto_stage1_$IDENTIFIER.e$EPOCH.valid.min$MIN.txt -src2 $SUM_OUT/roto_stage1_inter_$IDENTIFIER.e$EPOCH.valid.min$MIN.txt -output $SUM_OUT/roto_stage2_$IDENTIFIER.e$EPOCH.valid.min$MIN.txt -batch_size $BAT -max_length 850 -min_length 150 -gpu 0

#         printf "\n ****** STAGE 2 BLEU ****** \n"
#         echo "Reference: $VALID_TGT2"
#         # perl ~/table2text/multi-bleu.perl $VALID_TGT2 < $SUM_OUT/roto_stage2_$IDENTIFIER.e$EPOCH.valid.min$MIN.txt

#         cd /mnt/cephfs2/nlp/hongmin.wang/table2text/boxscore-data/scripts_aaai/evaluate
#         echo " ****** STAGE 2 RG CS CO ****** "
#         $PY3 evaluate.py --path $BASE --dataset valid --hypo $SUM_OUT/roto_stage2_$IDENTIFIER.e$EPOCH.valid.min$MIN.txt --plan $SUM_OUT/roto_stage1_inter_$IDENTIFIER.e$EPOCH.valid.min$MIN.txt
#         cd /mnt/cephfs2/nlp/hongmin.wang/table2text/$MYFOLDER

#         ##################################################################################################

#         printf "\n ****** STAGE 2 GOLD ****** \n"
#         $PY3 translate.py -model $MODEL1 -model2 $MODEL2 -src1 $VALID_SRC1 -edges $VALID_EDGE -tgt1 $VALID_TGT1 -src2 $VALID_SRC2 -output $SUM_OUT/roto_stage2_$IDENTIFIER.e$EPOCH.valid.goldcp.txt -batch_size $BAT -max_length 850 -min_length 150 -gpu 0

#         printf "\n ****** STAGE 2 BLEU GOLD ****** \n"
#         echo "Reference: $VALID_TGT2"
#         perl ~/table2text/multi-bleu.perl $VALID_TGT2 < $SUM_OUT/roto_stage2_$IDENTIFIER.e$EPOCH.valid.goldcp.txt

#         done
#     done
# done