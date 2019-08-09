#!/usr/bin/env bash
#BASE=/mnt/cephfs2/nlp/hongmin.wang/table2text/boxscore-data/scripts_aaai/new_dataset/new_ncpcc
BASE=/mnt/cephfs2/nlp/hongmin.wang/table2text/boxscore-data/scripts_aaai/new_dataset/new_extend
#IDENTIFIER=newcc-final
# IDENTIFIER=newcc-trl-1e-1
IDENTIFIER=aaai

TRAIN_SRC1=$BASE/train/src_train.norm.trim.txt
TRAIN_TGT1=$BASE/train/train_content_plan_ids.txt
TRAIN_SRC2=$BASE/train/train_content_plan_tks.txt
TRAIN_TGT2=$BASE/train/tgt_train.norm.mwe.trim.txt
TRAIN_PTR=$BASE/train/train_ptrs.txt
TRAIN_EDGE=$BASE/train/edge_combo_train.jsonl

wc $TRAIN_SRC1 $TRAIN_TGT1 $TRAIN_SRC2 $TRAIN_TGT2 $TRAIN_PTR

VALID_SRC1=$BASE/valid/src_valid.norm.trim.txt
VALID_TGT1=$BASE/valid/valid_content_plan_ids.txt
VALID_SRC2=$BASE/valid/valid_content_plan_tks.txt
VALID_TGT2=$BASE/valid/tgt_valid.norm.mwe.trim.txt
VALID_EDGE=$BASE/valid/edge_combo_valid.jsonl

wc $VALID_SRC1 $VALID_TGT1 $VALID_SRC2 $VALID_TGT2

TEST_SRC1=$BASE/test/src_test.norm.trim.txt
TEST_TGT1=$BASE/test/test_content_plan_ids.txt
TEST_SRC2=$BASE/test/test_content_plan_tks.txt
TEST_TGT2=$BASE/test/tgt_test.norm.mwe.trim.txt
TEST_EDGE_LEFT=$BASE/test/edge_combo_test.jsonl

wc $TEST_SRC1 $TEST_TGT1 $TEST_SRC2 $TEST_TGT2

#TEMP=/mnt/cephfs2/nlp/hongmin.wang/table2text/boxscore-data/scripts_aaai/new_dataset/new_extend_small
#head -n 100 $TRAIN_SRC1 > $TEMP/train/src_train.norm.trim.txt
#head -n 100 $TRAIN_TGT1 > $TEMP/train/train_content_plan_ids.txt
#head -n 100 $TRAIN_SRC2 > $TEMP/train/train_content_plan_tks.txt
#head -n 100 $TRAIN_TGT2 > $TEMP/train/tgt_train.norm.mwe.trim.txt
#head -n 100 $TRAIN_PTR > $TEMP/train/train_ptrs.txt
#head -n 100 $TRAIN_EDGE > $TEMP/train/edge_combo_train.jsonl
#
#head -n 100 $VALID_SRC1 > $TEMP/valid/src_valid.norm.trim.txt
#head -n 100 $VALID_TGT1 > $TEMP/valid/valid_content_plan_ids.txt
#head -n 100 $VALID_SRC2 > $TEMP/valid/valid_content_plan_tks.txt
#head -n 100 $VALID_TGT2 > $TEMP/valid/tgt_valid.norm.mwe.trim.txt
#head -n 100 $VALID_EDGE > $TEMP/valid/edge_combo_valid.jsonl
#
#TRAIN_SRC1=$TEMP/train/src_train.norm.trim.txt
#TRAIN_TGT1=$TEMP/train/train_content_plan_ids.txt
#TRAIN_SRC2=$TEMP/train/train_content_plan_tks.txt
#TRAIN_TGT2=$TEMP/train/tgt_train.norm.mwe.trim.txt
#TRAIN_PTR=$TEMP/train/train_ptrs.txt
#TRAIN_EDGE=$TEMP/train/edge_combo_train.jsonl
#
#VALID_SRC1=$TEMP/valid/src_valid.norm.trim.txt
#VALID_TGT1=$TEMP/valid/valid_content_plan_ids.txt
#VALID_SRC2=$TEMP/valid/valid_content_plan_tks.txt
#VALID_TGT2=$TEMP/valid/tgt_valid.norm.mwe.trim.txt
#VALID_EDGE=$TEMP/valid/edge_combo_valid.jsonl
#
#wc $VALID_SRC1 $VALID_TGT1 $VALID_SRC2 $VALID_TGT2
#wc $TRAIN_SRC1 $TRAIN_TGT1 $TRAIN_SRC2 $TRAIN_TGT2 $TRAIN_PTR

###################################################################################################
PREPRO=/mnt/cephfs2/nlp/hongmin.wang/table2text/boxscore-data/scripts_aaai/new_dataset/new_ncpcc/pt_data/$IDENTIFIER
mkdir -p $PREPRO

OUTPUT=/mnt/cephfs2/nlp/hongmin.wang/table2text/data2text-plan-py/new_models/$IDENTIFIER
mkdir -p $OUTPUT

SUM_OUT=/mnt/cephfs2/nlp/hongmin.wang/table2text/data2text-plan-py/new_outputs/$IDENTIFIER
mkdir -p $SUM_OUT

VALID_DIR=/mnt/cephfs2/nlp/hongmin.wang/table2text/boxscore-data/scripts/new_dataset/new_ncpcc

####################################################################################################
#echo "run preprocessing"
#python preprocess.py -train_src1 $TRAIN_SRC1 -train_tgt1 $TRAIN_TGT1 -train_src2 $TRAIN_SRC2 -train_tgt2 $TRAIN_TGT2 -train_edge $TRAIN_EDGE -valid_src1 $VALID_SRC1 -valid_tgt1 $VALID_TGT1 -valid_src2 $VALID_SRC2 -valid_tgt2 $VALID_TGT2 -valid_edge $VALID_EDGE -save_data $PREPRO/roto-$IDENTIFIER -src_seq_length 1000 -tgt_seq_length 1000 -dynamic_dict -train_ptr $TRAIN_PTR
#
####################################################################################################
echo "run training"
python train.py -data $PREPRO/roto-$IDENTIFIER -save_model $OUTPUT/roto -encoder_type1 mean -decoder_type1 pointer -enc_layers1 1 -dec_layers1 1 -encoder_type2 brnn -decoder_type2 rnn -enc_layers2 2 -dec_layers2 2 -batch_size 5 -feat_merge mlp -feat_vec_size 600 -word_vec_size 600 -rnn_size 600 -seed 1234 -epochs 50 -optim adagrad -learning_rate 0.15 -adagrad_accumulator_init 0.1 -report_every 100 -copy_attn -truncated_decoder 100 -gpuid 0 -attn_hidden 64 -reuse_copy_attn -start_decay_at 4 -learning_rate_decay 0.97 -valid_batch_size 5 -tensorboard -tensorboard_log_dir $OUTPUT/events

###################################################################################################
# echo " ****** Evaluation ****** "
# for EPOCH in $(seq 19 19)
# do
#     for MODEL1 in $(ls $OUTPUT/roto_stage1*_e$EPOCH.pt)
#     do

#         for MODEL2 in $(ls $OUTPUT/roto_stage2*_e$EPOCH.pt)
#         do

#        echo "--"
#        echo $MODEL1
#        echo $MODEL2
#
#        echo "--"
#        echo " ****** STAGE 1 ****** "
#        echo $VALID_SRC1
#        python translate.py -model $MODEL1 -src1 $VALID_SRC1 -output $SUM_OUT/roto_stage1_$IDENTIFIER.e$EPOCH.valid.txt -batch_size 10 -max_length 80 -gpu 0 -min_length 20 -stage1
#
#        echo " ****** create_content_plan_from_index ****** "
#        python create_content_plan_from_index.py $VALID_SRC1 $SUM_OUT/roto_stage1_$IDENTIFIER.e$EPOCH.valid.txt $SUM_OUT/roto_stage1_$IDENTIFIER.e$EPOCH.h5-tuples.valid.txt  $SUM_OUT/roto_stage1_inter_$IDENTIFIER.e$EPOCH.valid.txt
#
#        echo " ****** STAGE 2 ****** "
#        python translate.py -model $MODEL1 -model2 $MODEL2 -src1 $VALID_SRC1 -tgt1 $SUM_OUT/roto_stage1_$IDENTIFIER.e$EPOCH.valid.txt -src2 $SUM_OUT/roto_stage1_inter_$IDENTIFIER.e$EPOCH.valid.txt -output $SUM_OUT/roto_stage2_$IDENTIFIER.e$EPOCH.valid.txt -batch_size 10 -max_length 850 -min_length 150 -gpu 0
#
#        echo " ****** BLEU ****** "
#        perl ~/onmt-tf-whm/third_party/multi-bleu.perl $VALID_TGT2 < $SUM_OUT/roto_stage2_$IDENTIFIER.e$EPOCH.valid.txt

        # cd /mnt/cephfs2/nlp/hongmin.wang/table2text/boxscore-data/scripts/evaluate
        # echo " ****** RG CS CO ****** "
#        $PY3 evaluate.py --dataset valid --hypo $SUM_OUT/roto_stage2_$IDENTIFIER.e$EPOCH.valid.txt --plan $SUM_OUT/roto_stage1_inter_$IDENTIFIER.e$EPOCH.valid.txt

        # $PY3 make_human_eval.py \
        # --dataset valid \
        # --hypo $SUM_OUT/roto_stage2_$IDENTIFIER.e$EPOCH.valid.new.txt \
        # --template /home/hongmin_wang/table2text_nlg/harvardnlp/data2text-my/new_rulebased.valid.txt \
        # --ws17 /home/hongmin_wang/table2text_nlg/harvardnlp/data2text-harvard/outputs/new-roto-v2/new-roto-v2-beam5_gens.valid.24.txt \
        # --ent /home/hongmin_wang/table2text_nlg/harvardnlp/data2text-entity-py/rotowire/outputs/clean_large/roto_clean_large-beam5_gens.valid.e19.txt \
        # --ncp /mnt/cephfs2/nlp/hongmin.wang/table2text/data2text-plan-py/new_outputs/newcc-final/roto_stage2_newcc-final.e18.valid.txt

        # cd /mnt/cephfs2/nlp/hongmin.wang/table2text/data2text-plan-py

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

#         done
#     done
# done