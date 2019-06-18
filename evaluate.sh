#!/usr/bin/env bash

IDENTIFIER=$1
BASE=rotowire/bk/ratish

for EPOCH in $(seq 20 25)
do
    for MODEL1 in $(ls $BASE/gen_model/$IDENTIFIER/roto_stage1*_e$EPOCH.pt)
    do
#        printf "\n"
#        echo "Generate the content plan using MODEL1 : "$MODEL1
#        echo "[*** Arguments ***] python translate.py -model $MODEL1 -src1 rotowire/allclean/rotowire/src_valid.txt -output $BASE/gen/roto_stage1_$IDENTIFIER-beam5_gens.e$EPOCH.txt"
#
#        python translate.py -model $MODEL1 -src1 rotowire/allclean/rotowire/src_valid.txt -output $BASE/gen/roto_stage1_$IDENTIFIER-beam5_gens.e$EPOCH.txt -batch_size 10 -max_length 80 -gpu 0 -min_length 35 -stage1
#
#        printf "\n"
#        echo "create_content_plan_from_index"
#        echo "[*** Arguments ***] rotowire/allclean/rotowire/src_valid.txt $BASE/gen/roto_stage1_$IDENTIFIER-beam5_gens.e$EPOCH.txt $BASE/transform_gen/roto_stage1_$IDENTIFIER-beam5_gens.e$EPOCH.h5-tuples.txt  $BASE/gen/roto_stage1_inter_$IDENTIFIER-beam5_gens.e$EPOCH.txt"
#
#        python create_content_plan_from_index.py rotowire/allclean/rotowire/src_valid.txt $BASE/gen/roto_stage1_$IDENTIFIER-beam5_gens.e$EPOCH.txt $BASE/transform_gen/roto_stage1_$IDENTIFIER-beam5_gens.e$EPOCH.h5-tuples.txt  $BASE/gen/roto_stage1_inter_$IDENTIFIER-beam5_gens.e$EPOCH.txt
#
#        printf "\n"
#        echo "non_rg_metrics"
#        echo "[*** Arguments ***] python non_rg_metrics.py rotowire/bk/valid_bk/roto-gold-valid.h5-tuples.txt $BASE/transform_gen/roto_stage1_$IDENTIFIER-beam5_gens.h5-tuples.e$EPOCH.txt"
#
#        python non_rg_metrics.py rotowire/bk/valid_bk/roto-gold-valid.h5-tuples.txt $BASE/transform_gen/roto_stage1_$IDENTIFIER-beam5_gens.e$EPOCH.h5-tuples.txt

        for MODEL2 in $(ls $BASE/gen_model/$IDENTIFIER/roto_stage2*_e$EPOCH.pt)
        do
            printf "\n"
            echo "Output summary is generated using MODEL2 : "$MODEL2
            echo "[*** Arguments ***] python translate.py -model $MODEL1 -model2 $MODEL2 -src1 rotowire/allclean/rotowire/src_valid.txt -tgt1 $BASE/gen/roto_stage1_$IDENTIFIER-beam5_gens.e$EPOCH.txt -src2 $BASE/gen/roto_stage1_inter_$IDENTIFIER-beam5_gens.e$EPOCH.txt -output $BASE/gen/roto_stage2_$IDENTIFIER-beam5_gens.e$EPOCH.txt"

            python translate.py -model $MODEL1 -model2 $MODEL2 -src1 rotowire/allclean/rotowire/src_valid.txt -tgt1 $BASE/gen/roto_stage1_$IDENTIFIER-beam5_gens.e$EPOCH.txt -src2 $BASE/gen/roto_stage1_inter_$IDENTIFIER-beam5_gens.e$EPOCH.txt -output $BASE/gen/roto_stage2_$IDENTIFIER-beam5_gens.e$EPOCH.txt -batch_size 10 -max_length 400 -min_length 200 -gpu 0

            printf "\n"
            echo "Converting output summary to tokens"

            python3 mwe2tks.py --input $BASE/gen/roto_stage2_$IDENTIFIER-beam5_gens.e$EPOCH.txt

            printf "\n"
            echo "BLEU against rotowire/valid/clean/tgt_valid.norm.filter.tk.trim.txt"

            perl ~/onmt-tf-whm/third_party/multi-bleu.perl rotowire/valid/clean/tgt_valid.norm.filter.tk.trim.txt < $BASE/gen/roto_stage2_$IDENTIFIER-beam5_gens.e$EPOCH.txt.tk

            echo "BLEU against rotowire/valid/tgt_valid.txt"

            perl ~/onmt-tf-whm/third_party/multi-bleu.perl rotowire/valid/tgt_valid.txt < $BASE/gen/roto_stage2_$IDENTIFIER-beam5_gens.e$EPOCH.txt.tk

            printf "\n"
            echo "generate tuples from content plan"
            echo "[*** Arguments ***] python translate.py -model $MODEL1 -model2 $MODEL2 -src1 rotowire/allclean/rotowire/src_valid.txt -tgt1 $BASE/gen/roto_stage1_$IDENTIFIER-beam5_gens.e$EPOCH.txt -src2 $BASE/gen/roto_stage1_inter_$IDENTIFIER-beam5_gens.e$EPOCH.txt -output $BASE/gen/roto_stage2_$IDENTIFIER-beam5_gens.e$EPOCH.txt"

            python data_utils.py -mode prep_gen_data -gen_fi $BASE/gen/roto_stage2_$IDENTIFIER-beam5_gens.e$EPOCH.txt.tk -dict_pfx ie_data/roto-ie -output_fi $BASE/transform_gen/roto_stage2_$IDENTIFIER-beam5_gens.e$EPOCH.h5 -input_path ../boxscore-data/rotowire

            printf "\n"
            echo "extract content plan from summary"
            cd ../data2text-1/
            th extractor.lua -datafile ../data2text-plan-py/ie_data/roto-ie.h5 -preddata ../data2text-plan-py/$BASE/transform_gen/roto_stage2_$IDENTIFIER-beam5_gens.e$EPOCH.h5 -dict_pfx ../data2text-plan-py/ie_data/roto-ie -just_eval
            cd ../data2text-plan-py

            printf "\n"
            echo "non_rg_metrics"
            echo "[*** Arguments ***] python non_rg_metrics.py rotowire/valid/roto-gold-valid.h5-tuples.txt $BASE/transform_gen/roto_stage2_$IDENTIFIER-beam5_gens.e$EPOCH.h5-tuples.txt"
            python non_rg_metrics.py rotowire/valid/roto-gold-valid.h5-tuples.txt $BASE/transform_gen/roto_stage2_$IDENTIFIER-beam5_gens.e$EPOCH.h5-tuples.txt

        done
    done
done