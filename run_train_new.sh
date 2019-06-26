#!/usr/bin/env bash
BASE=/home/hongmin_wang/table2text_nlg/harvardnlp/boxscore-data/scripts/new_dataset/new_ncpcc/
IDENTIFIER=new-ncpcc

TRAIN_SRC1=$BASE/train/src_train.norm.trim.ncp.txt
TRAIN_TGT1=$BASE/train/train_content_plan_ids.ncp.txt
TRAIN_SRC2=$BASE/train/train_content_plan_tks.txt
TRAIN_TGT2=$BASE/train/tgt_train.norm.filter.mwe.trim.txt
TRAIN_PTR=$BASE/train/train_ptrs.txt

wc -l $TRAIN_SRC1 $TRAIN_TGT1 $TRAIN_SRC2 $TRAIN_TGT2

VALID_SRC1=$BASE/valid/src_valid.norm.trim.ncp.txt
VALID_TGT1=$BASE/valid/valid_content_plan_ids.ncp.txt
VALID_SRC2=$BASE/valid/valid_content_plan_tks.txt
VALID_TGT2=$BASE/valid/tgt_valid.norm.filter.mwe.trim.txt

PREPRO=/home/hongmin_wang/table2text_nlg/harvardnlp/boxscore-data/scripts/new_dataset/new_ncpcc/pt_data/$IDENTIFIER
mkdir -p $PREPRO

wc -l $VALID_SRC1 $VALID_TGT1 $VALID_SRC2 $VALID_TGT2

echo "run preprocessing"
python preprocess.py -train_src1 $TRAIN_SRC1 -train_tgt1 $TRAIN_TGT1 -train_src2 $TRAIN_SRC2 -train_tgt2 $TRAIN_TGT2 -valid_src1 $VALID_SRC1 -valid_tgt1 $VALID_TGT1 -valid_src2 $VALID_SRC2 -valid_tgt2 $VALID_TGT2 -save_data $PREPRO/roto-cc -src_seq_length 1000 -tgt_seq_length 1000 -dynamic_dict -train_ptr $TRAIN_PTR

OUTPUT=/home/hongmin_wang/table2text_nlg/harvardnlp/data2text-plan-py/new_models/$IDENTIFIER
mkdir -p $OUTPUT

echo "run training"
python train.py -data $PREPRO/roto-cc -save_model $OUTPUT/roto -encoder_type1 mean -decoder_type1 pointer -enc_layers1 1 -dec_layers1 1 -encoder_type2 brnn -decoder_type2 rnn -enc_layers2 2 -dec_layers2 2 -batch_size 5 -feat_merge mlp -feat_vec_size 600 -word_vec_size 600 -rnn_size 600 -seed 1234 -epochs 25 -optim adagrad -learning_rate 0.15 -adagrad_accumulator_init 0.1 -report_every 100 -copy_attn -truncated_decoder 100 -gpuid 0 -attn_hidden 64 -reuse_copy_attn -start_decay_at 4 -learning_rate_decay 0.97 -valid_batch_size 5 -tensorboard -tensorboard_log_dir $OUTPUT/events
