#!/usr/bin/env bash
##################################################################################################
PY3='/mnt/cephfs2/nlp/hongmin.wang/anaconda3/envs/pt11py37/bin/python'
cd /mnt/cephfs2/nlp/hongmin.wang/table2text/nbagraph2summary
echo "current working directory:"
pwd

###################################################################################################
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
# export CUDA_VISIBLE_DEVICES=1

##################################################################################################

# MYHOME=pwd
MYHOME="$PWD"
echo $MYHOME

DATA=inlg
SUFFIX=ncpcc
BASE=/mnt/cephfs2/nlp/hongmin.wang/table2text/boxscore-data/scripts_$DATA/new_dataset/new_$SUFFIX
ENCODER=graph
# EDGE_DIR=$1
EDGE_DIR=big2small #small2big

PREPATH=$ENCODER\_$DATA\_new_edgedir-$EDGE_DIR
echo $DATA
echo $SUFFIX
echo $BASE
echo $PREPATH

TRAIN_SRC1=$BASE/train/src_train.norm.trim.ncp.full.txt
TRAIN_TGT1=$BASE/train/train_content_plan_ids.ncp.full.txt
TRAIN_SRC2=$BASE/train/train_content_plan_tks.txt
TRAIN_TGT2=$BASE/train/tgt_train.norm.mwe.trim.txt
TRAIN_PTR=$BASE/train/train_ptrs.txt
TRAIN_EDGE=$BASE/train/edges_train.ncp.new.direction-$EDGE_DIR.jsonl

wc $TRAIN_SRC1 $TRAIN_TGT1 $TRAIN_SRC2 $TRAIN_TGT2 $TRAIN_PTR

VALID_SRC1=$BASE/valid/src_valid.norm.trim.ncp.full.txt
VALID_TGT1=$BASE/valid/valid_content_plan_ids.ncp.full.txt
VALID_SRC2=$BASE/valid/valid_content_plan_tks.txt
VALID_TGT2=$BASE/valid/tgt_valid.norm.mwe.trim.txt
VALID_EDGE=$BASE/valid/edges_valid.ncp.new.direction-$EDGE_DIR.jsonl

wc $VALID_SRC1 $VALID_TGT1 $VALID_SRC2 $VALID_TGT2

TEST_SRC1=$BASE/test/src_test.norm.trim.ncp.full.txt
TEST_TGT1=$BASE/test/test_content_plan_ids.ncp.full.txt
TEST_SRC2=$BASE/test/test_content_plan_tks.txt
TEST_TGT2=$BASE/test/tgt_test.norm.mwe.trim.txt
TEST_EDGE_LEFT=$BASE/test/edges_test.ncp.new.direction-$EDGE_DIR.jsonl

wc $TEST_SRC1 $TEST_TGT1 $TEST_SRC2 $TEST_TGT2

###################################################################################################
PREPRO=/mnt/cephfs2/nlp/hongmin.wang/table2text/boxscore-data/scripts_$DATA/new_dataset/new_$SUFFIX/pt_data/$PREPATH
mkdir -p $PREPRO

DIM=256
BAT=16
DEC=2
EDGE_AWARE=add  # add/linear
EDGE_AGGR=max   # mean/max
NODE_FUSE=dense  # dense highway
OUT_LAYER=add # highway-graph highway-fuse
# EDGE_AWARE=$2
# EDGE_AGGR=$3  # mean/max
# NODE_FUSE=$4  # dense highway
# OUT_LAYER=$5  # add highway-graph highway-fuse
IDENTIFIER=$PREPATH\_edgeaware-$EDGE_AWARE\_edgeaggr-$EDGE_AGGR\_graphfuse-$NODE_FUSE\_outlayer-$OUT_LAYER
printf "IDENTIFIER = $IDENTIFIER \n"
OUTPUT=$MYHOME/$DATA\_models/$IDENTIFIER
# OUTPUT=$MYHOME/temp_models/$IDENTIFIER
mkdir -p $OUTPUT

SUM_OUT=$MYHOME/$DATA\_outputs/$IDENTIFIER
# SUM_OUT=$MYHOME/temp_outputs/$IDENTIFIER
mkdir -p $SUM_OUT

VALID_DIR=/mnt/cephfs2/nlp/hongmin.wang/table2text/boxscore-data/scripts/new_dataset/new_ncpcc

#####################################################################################################
# echo "run preprocessing"
# python preprocess.py -train_src1 $TRAIN_SRC1 -train_tgt1 $TRAIN_TGT1 -train_src2 $TRAIN_SRC2 -train_tgt2 $TRAIN_TGT2 -train_edge $TRAIN_EDGE -valid_src1 $VALID_SRC1 -valid_tgt1 $VALID_TGT1 -valid_src2 $VALID_SRC2 -valid_tgt2 $VALID_TGT2 -valid_edge $VALID_EDGE -save_data $PREPRO/roto-$PREPATH -src_seq_length 1000 -tgt_seq_length 1000 -dynamic_dict -train_ptr $TRAIN_PTR

####################################################################################################
# echo "run training"
# $PY3 train.py -data $PREPRO/roto-$PREPATH -save_model $OUTPUT/roto -encoder_type1 mean -decoder_type1 pointer -enc_layers1 1 -dec_layers1 1 -encoder_type2 brnn -decoder_type2 rnn -enc_layers2 $DEC -dec_layers2 $DEC -batch_size $BAT -feat_merge mlp -feat_vec_size $DIM -word_vec_size $DIM -rnn_size $DIM -seed 1234 -epochs 50 -optim adagrad -learning_rate 0.15 -adagrad_accumulator_init 0.1 -report_every 50 -copy_attn -truncated_decoder 100 -max_generator_batches 1000 -gpuid 0 -attn_hidden 64 -reuse_copy_attn -start_decay_at 4 -learning_rate_decay 0.97 -valid_batch_size $BAT -tensorboard -tensorboard_log_dir $OUTPUT/events -encoder_type1 $ENCODER -edge_aware $EDGE_AWARE -edge_aggr $EDGE_AGGR -encoder_graph_fuse $NODE_FUSE -encoder_outlayer $OUT_LAYER -csl

##################################################################################################
echo " ****** Evaluation ****** "
for EPOCH in $(seq 19 50)
do
    for MODEL1 in $(ls $OUTPUT/roto_stage1*_e$EPOCH.pt)
    do

        for MODEL2 in $(ls $OUTPUT/roto_stage2*_e$EPOCH.pt)
        do

        echo "--"
        echo $MODEL1
        echo $MODEL2

        printf "\n--"
        echo " ****** STAGE 1 ****** \n"
        echo "input src: $VALID_SRC1"
        echo "saving to: $SUM_OUT/roto_stage1_$IDENTIFIER.e$EPOCH.valid.txt"
        $PY3 translate.py -model $MODEL1 -src1 $VALID_SRC1 -edges $VALID_EDGE -output $SUM_OUT/roto_stage1_$IDENTIFIER.e$EPOCH.valid.txt -batch_size 10 -max_length 80 -gpu 0 -min_length 20 -stage1

        printf "\n ****** create_content_plan_from_index ****** \n"
        $PY3 create_content_plan_from_index.py $VALID_SRC1 $SUM_OUT/roto_stage1_$IDENTIFIER.e$EPOCH.valid.txt $SUM_OUT/roto_stage1_$IDENTIFIER.e$EPOCH.h5-tuples.valid.txt $SUM_OUT/roto_stage1_inter_$IDENTIFIER.e$EPOCH.valid.txt

        printf "\n ****** STAGE 2 ****** \n"
        $PY3 translate.py -model $MODEL1 -model2 $MODEL2 -src1 $VALID_SRC1 -edges $VALID_EDGE -tgt1 $SUM_OUT/roto_stage1_$IDENTIFIER.e$EPOCH.valid.txt -src2 $SUM_OUT/roto_stage1_inter_$IDENTIFIER.e$EPOCH.valid.txt -output $SUM_OUT/roto_stage2_$IDENTIFIER.e$EPOCH.valid.txt -batch_size 10 -max_length 850 -min_length 150 -gpu 0

        printf "\n ****** BLEU ****** \n"
        echo "Reference: $VALID_TGT2"
        perl ~/table2text/multi-bleu.perl $VALID_TGT2 < $SUM_OUT/roto_stage2_$IDENTIFIER.e$EPOCH.valid.txt
#
#        ###################################################################################################
#        cd /mnt/cephfs2/nlp/hongmin.wang/table2text/boxscore-data/scripts_aaai/evaluate
#        echo " ****** RG CS CO ****** "
#        $PY3 evaluate.py --path $BASE --dataset valid --hypo $SUM_OUT/roto_stage2_$IDENTIFIER.e$EPOCH.valid.txt --plan $SUM_OUT/roto_stage1_inter_$IDENTIFIER.e$EPOCH.valid.txt
#        cd /mnt/cephfs2/nlp/hongmin.wang/table2text/data2text-plan-py

#         $PY3 make_human_eval.py \
#         --dataset valid \
#         --hypo $SUM_OUT/roto_stage2_$IDENTIFIER.e$EPOCH.valid.new.txt \
#         --template /home/hongmin_wang/table2text_nlg/harvardnlp/data2text-my/new_rulebased.valid.txt \
#         --ws17 /home/hongmin_wang/table2text_nlg/harvardnlp/data2text-harvard/outputs/new-roto-v2/new-roto-v2-beam5_gens.valid.24.txt \
#         --ent /home/hongmin_wang/table2text_nlg/harvardnlp/data2text-entity-py/rotowire/outputs/clean_large/roto_clean_large-beam5_gens.valid.e19.txt \
#         --ncp /mnt/cephfs2/nlp/hongmin.wang/table2text/data2text-plan-py/new_outputs/newcc-final/roto_stage2_newcc-final.e18.valid.txt
#
#         cd /mnt/cephfs2/nlp/hongmin.wang/table2text/data2text-plan-py
#
#        echo "--"
#        echo " ****** STAGE 1 ****** "
#        echo $TEST_SRC1
#        python translate.py -model $MODEL1 -src1 $TEST_SRC1 -output $SUM_OUT/roto_stage1_$IDENTIFIER.e$EPOCH.test.txt -batch_size 10 -max_length 80 -gpu 0 -min_length 20 -stage1
#
#        echo " ****** create_content_plan_from_index ****** "
#        python create_content_plan_from_index.py $TEST_SRC1 $SUM_OUT/roto_stage1_$IDENTIFIER.e$EPOCH.test.txt $SUM_OUT/roto_stage1_$IDENTIFIER.e$EPOCH.test.h5-tuples.txt  $SUM_OUT/roto_stage1_inter_$IDENTIFIER.e$EPOCH.test.txt
#
#        echo " ****** STAGE 2 ****** "
#        python translate.py -model $MODEL1 -model2 $MODEL2 -src1 $TEST_SRC1 -tgt1 $SUM_OUT/roto_stage1_$IDENTIFIER.e$EPOCH.test.txt -src2 $SUM_OUT/roto_stage1_inter_$IDENTIFIER.e$EPOCH.test.txt -output $SUM_OUT/roto_stage2_$IDENTIFIER.e$EPOCH.test.txt -batch_size 10 -max_length 850 -min_length 150 -gpu 0
#
#        echo " ****** BLEU ****** "
#        perl ~/onmt-tf-whm/third_party/multi-bleu.perl $TEST_TGT2 < $SUM_OUT/roto_stage2_$IDENTIFIER.e$EPOCH.test.txt
#
#        cd /mnt/cephfs2/nlp/hongmin.wang/table2text/boxscore-data/scripts/evaluate
#        echo " ****** RG CS CO ****** "
#        $PY3 evaluate.py --dataset test --hypo $SUM_OUT/roto_stage2_$IDENTIFIER.e$EPOCH.test.txt --plan $SUM_OUT/roto_stage1_inter_$IDENTIFIER.e$EPOCH.test.txt
#        cd /mnt/cephfs2/nlp/hongmin.wang/table2text/data2text-plan-py

        done
    done
done



####################################################################################################
#TEMP=/mnt/cephfs2/nlp/hongmin.wang/table2text/boxscore-data/scripts_aaai/new_dataset/new_extend_small
#mkdir -p $TEMP
#head -n 100 $TRAIN_SRC1 > $TEMP/train/src_train.norm.trim.ncp.full.txt
#head -n 100 $TRAIN_TGT1 > $TEMP/train/train_content_plan_ids.ncp.full.txt
#head -n 100 $TRAIN_SRC2 > $TEMP/train/train_content_plan_tks.txt
#head -n 100 $TRAIN_TGT2 > $TEMP/train/tgt_train.norm.mwe.trim.txt
#head -n 100 $TRAIN_PTR > $TEMP/train/train_ptrs.txt
#head -n 100 $TRAIN_EDGE > $TEMP/train/edges_train.ncp.jsonl
#
#head -n 100 $VALID_SRC1 > $TEMP/valid/src_valid.norm.trim.ncp.full.txt
#head -n 100 $VALID_TGT1 > $TEMP/valid/valid_content_plan_ids.ncp.full.txt
#head -n 100 $VALID_SRC2 > $TEMP/valid/valid_content_plan_tks.txt
#head -n 100 $VALID_TGT2 > $TEMP/valid/tgt_valid.norm.mwe.trim.txt
#head -n 100 $VALID_EDGE > $TEMP/valid/edges_valid.ncp.jsonl
#
#TRAIN_SRC1=$TEMP/train/src_train.norm.trim.ncp.full.txt
#TRAIN_TGT1=$TEMP/train/train_content_plan_ids.ncp.full.txt
#TRAIN_SRC2=$TEMP/train/train_content_plan_tks.txt
#TRAIN_TGT2=$TEMP/train/tgt_train.norm.mwe.trim.txt
#TRAIN_PTR=$TEMP/train/train_ptrs.txt
#TRAIN_EDGE=$TEMP/train/edges_train.ncp.jsonl
#
#VALID_SRC1=$TEMP/valid/src_valid.norm.trim.ncp.full.txt
#VALID_TGT1=$TEMP/valid/valid_content_plan_ids.ncp.full.txt
#VALID_SRC2=$TEMP/valid/valid_content_plan_tks.txt
#VALID_TGT2=$TEMP/valid/tgt_valid.norm.mwe.trim.txt
#VALID_EDGE=$TEMP/valid/edges_valid.ncp.jsonl
#
#wc $VALID_SRC1 $VALID_TGT1 $VALID_SRC2 $VALID_TGT2
#wc $TRAIN_SRC1 $TRAIN_TGT1 $TRAIN_SRC2 $TRAIN_TGT2 $TRAIN_PTR
#
#PREPRO=$TEMP/pt_data/$DATA
#mkdir -p $PREPRO