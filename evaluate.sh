#!/usr/bin/env bash

IDENTIFIER=$1
DATA=$2

BASE=rotowire/ratish

for EPOCH in $(seq 23 23)
do
    for MODEL1 in $(ls $BASE/gen_model/$IDENTIFIER/roto_stage1*_e$EPOCH.pt)
    do
        printf "\n"
        echo "Generate the content plan using MODEL1 : "$MODEL1
        echo "[*** Arguments ***] python translate.py -model $MODEL1 -src1 rotowire/allclean/rotowire/src_$DATA.txt -output $BASE/gen/roto_stage1_$IDENTIFIER-beam5_gens.$DATA.e$EPOCH.txt"

        python translate.py -model $MODEL1 -src1 rotowire/allclean/rotowire/src_$DATA.txt -output $BASE/gen/roto_stage1_$IDENTIFIER-beam5_gens.$DATA.e$EPOCH.txt -batch_size 10 -max_length 80 -gpu 0 -min_length 35 -stage1

        printf "\n"
        echo "create_content_plan_from_index"
        echo "[*** Arguments ***] rotowire/allclean/rotowire/src_$DATA.txt $BASE/gen/roto_stage1_$IDENTIFIER-beam5_gens.$DATA.e$EPOCH.txt $BASE/transform_gen/roto_stage1_$IDENTIFIER-beam5_gens.$DATA.e$EPOCH.h5-tuples.txt  $BASE/gen/roto_stage1_inter_$IDENTIFIER-beam5_gens.$DATA.e$EPOCH.txt"

        python create_content_plan_from_index.py rotowire/allclean/rotowire/src_$DATA.txt $BASE/gen/roto_stage1_$IDENTIFIER-beam5_gens.$DATA.e$EPOCH.txt $BASE/transform_gen/roto_stage1_$IDENTIFIER-beam5_gens.$DATA.e$EPOCH.h5-tuples.txt  $BASE/gen/roto_stage1_inter_$IDENTIFIER-beam5_gens.$DATA.e$EPOCH.txt

        printf "\n"
        echo "non_rg_metrics"
        echo "[*** Arguments ***] python non_rg_metrics.py rotowire/bk/$DATA\_bk/roto-gold-$DATA.h5-tuples.txt $BASE/transform_gen/roto_stage1_$IDENTIFIER-beam5_gens.$DATA.h5-tuples.e$EPOCH.txt"

        python non_rg_metrics.py rotowire/bk/$DATA\_bk/roto-gold-$DATA.h5-tuples.txt $BASE/transform_gen/roto_stage1_$IDENTIFIER-beam5_gens.$DATA.e$EPOCH.h5-tuples.txt

        for MODEL2 in $(ls $BASE/gen_model/$IDENTIFIER/roto_stage2*_e$EPOCH.pt)
        do
            printf "\n"
            echo "Output summary is generated using MODEL2 : "$MODEL2
            echo "[*** Arguments ***] python translate.py -model $MODEL1 -model2 $MODEL2 -src1 rotowire/allclean/rotowire/src_$DATA.txt -tgt1 $BASE/gen/roto_stage1_$IDENTIFIER-beam5_gens.$DATA.e$EPOCH.txt -src2 $BASE/gen/roto_stage1_inter_$IDENTIFIER-beam5_gens.$DATA.e$EPOCH.txt -output $BASE/gen/roto_stage2_$IDENTIFIER-beam5_gens.$DATA.e$EPOCH.txt"

            python translate.py -model $MODEL1 -model2 $MODEL2 -src1 rotowire/allclean/rotowire/src_$DATA.txt -tgt1 $BASE/gen/roto_stage1_$IDENTIFIER-beam5_gens.$DATA.e$EPOCH.txt -src2 $BASE/gen/roto_stage1_inter_$IDENTIFIER-beam5_gens.$DATA.e$EPOCH.txt -output $BASE/gen/roto_stage2_$IDENTIFIER-beam5_gens.$DATA.e$EPOCH.txt -batch_size 10 -max_length 400 -min_length 150 -gpu 0

            printf "\n"
            echo "Converting output summary to tokens [gens]"
            python3 mwe2tks.py --input $BASE/gen/roto_stage2_$IDENTIFIER-beam5_gens.$DATA.e$EPOCH.txt

            printf "\n [BLEU beam5 gens]"
            echo "BLEU against rotowire/$DATA/clean/tgt_$DATA.norm.filter.tk.trim.txt"

            perl ~/onmt-tf-whm/third_party/multi-bleu.perl rotowire/$DATA/clean/tgt_$DATA.norm.filter.tk.trim.txt < $BASE/gen/roto_stage2_$IDENTIFIER-beam5_gens.$DATA.e$EPOCH.txt.tk

            echo "BLEU against rotowire/$DATA/tgt_$DATA.txt"

            perl ~/onmt-tf-whm/third_party/multi-bleu.perl rotowire/$DATA/tgt_$DATA.txt < $BASE/gen/roto_stage2_$IDENTIFIER-beam5_gens.$DATA.e$EPOCH.txt.tk

            printf "\n"
            echo "Output summary is generated using Oracle content plan"
            python translate.py -model $MODEL1 -model2 $MODEL2 -src1 rotowire/allclean/rotowire/src_$DATA.txt -tgt1 rotowire/allclean/rotowire/$DATA\_content_plan.txt -src2 rotowire/allclean/rotowire/inter/$DATA\_content_plan.txt -output $BASE/gen/roto_stage2_$IDENTIFIER-oracle.$DATA.e$EPOCH.txt -batch_size 10 -max_length 400 -min_length 150 -gpu 0

            printf "\n"
            echo "Converting output summary to tokens [oracle]"

            python3 mwe2tks.py --input $BASE/gen/roto_stage2_$IDENTIFIER-oracle.$DATA.e$EPOCH.txt

            printf "\n [BLEU oracle]"
            echo "BLEU against rotowire/$DATA/clean/tgt_$DATA.norm.filter.tk.trim.txt"

            perl ~/onmt-tf-whm/third_party/multi-bleu.perl rotowire/$DATA/clean/tgt_$DATA.norm.filter.tk.trim.txt < $BASE/gen/roto_stage2_$IDENTIFIER-oracle.$DATA.e$EPOCH.txt.tk

            echo "BLEU against rotowire/$DATA/tgt_$DATA.txt"

            perl ~/onmt-tf-whm/third_party/multi-bleu.perl rotowire/$DATA/tgt_$DATA.txt < $BASE/gen/roto_stage2_$IDENTIFIER-oracle.$DATA.e$EPOCH.txt.tk

            printf "\n"
            echo "generate tuples from content plan"
            echo "[*** Arguments ***] python data_utils.py -mode prep_gen_data -gen_fi $BASE/gen/roto_stage2_$IDENTIFIER-beam5_gens.$DATA.e$EPOCH.txt.tk -dict_pfx downloads/ie_data/roto-ie -output_fi $BASE/transform_gen/roto_stage2_$IDENTIFIER-beam5_gens.$DATA.e$EPOCH.h5 -input_path ../boxscore-data/rotowire"
            python data_utils.py -test -mode prep_gen_data -gen_fi $BASE/gen/roto_stage2_$IDENTIFIER-beam5_gens.$DATA.e$EPOCH.txt.tk -dict_pfx downloads/ie_data/roto-ie -output_fi $BASE/transform_gen/roto_stage2_$IDENTIFIER-beam5_gens.$DATA.e$EPOCH.h5 -input_path ../boxscore-data/rotowire

            echo "generate tuples from oracle content plan"
            echo "[*** Arguments ***] python data_utils.py -mode prep_gen_data -gen_fi $BASE/gen/roto_stage2_$IDENTIFIER-oracle.$DATA.e$EPOCH.txt.tk -dict_pfx downloads/ie_data/roto-ie -output_fi $BASE/transform_gen/roto_stage2_$IDENTIFIER-oracle.$DATA.e$EPOCH.h5 -input_path ../boxscore-data/rotowire"
            python data_utils.py -test -mode prep_gen_data -gen_fi $BASE/gen/roto_stage2_$IDENTIFIER-oracle.$DATA.e$EPOCH.txt.tk -dict_pfx downloads/ie_data/roto-ie -output_fi $BASE/transform_gen/roto_stage2_$IDENTIFIER-oracle.$DATA.e$EPOCH.h5 -input_path ../boxscore-data/rotowire

            printf "\n"
            echo "extracting tuples from summary"
            echo "[*** Arguments ***] th extractor.lua -datafile roto-ie.h5 -preddata ../data2text-plan-py/$BASE/transform_gen/roto_stage2_$IDENTIFIER-beam5_gens.$DATA.e$EPOCH.h5 -dict_pfx roto-ie -just_eval"
            cd ../data2text-1/
            th extractor.lua -datafile roto-ie.h5 -preddata ../data2text-plan-py/$BASE/transform_gen/roto_stage2_$IDENTIFIER-beam5_gens.$DATA.e$EPOCH.h5 -dict_pfx ../data2text-plan-py/downloads/ie_data/roto-ie -just_eval
            echo "extracting tuples from summary (oracle)"
            echo "[*** Arguments ***] th extractor.lua -datafile roto-ie.h5 -preddata ../data2text-plan-py/$BASE/transform_gen/roto_stage2_$IDENTIFIER-oracle.$DATA.e$EPOCH.h5 -dict_pfx roto-ie -just_eval"
            th extractor.lua -datafile roto-ie.h5 -preddata ../data2text-plan-py/$BASE/transform_gen/roto_stage2_$IDENTIFIER-oracle.$DATA.e$EPOCH.h5 -dict_pfx ../data2text-plan-py/downloads/ie_data/roto-ie -just_eval
            cd ../data2text-plan-py

            printf "\n"
            echo "non_rg_metrics"
            echo "[*** Arguments ***] python non_rg_metrics.py rotowire/$DATA/roto-gold-$DATA.h5-tuples.txt $BASE/transform_gen/roto_stage2_$IDENTIFIER-beam5_gens.$DATA.e$EPOCH.h5-tuples.txt"
            python non_rg_metrics.py rotowire/$DATA/roto-gold-$DATA.h5-tuples.txt $BASE/transform_gen/roto_stage2_$IDENTIFIER-beam5_gens.$DATA.e$EPOCH.h5-tuples.txt
            echo "non_rg_metrics (oracle)"
            echo "[*** Arguments ***] python non_rg_metrics.py rotowire/$DATA/roto-gold-$DATA.h5-tuples.txt $BASE/transform_gen/roto_stage2_$IDENTIFIER-oracle.$DATA.e$EPOCH.h5-tuples.txt"
            python non_rg_metrics.py rotowire/$DATA/roto-gold-$DATA.h5-tuples.txt $BASE/transform_gen/roto_stage2_$IDENTIFIER-oracle.$DATA.e$EPOCH.h5-tuples.txt

        done
    done
done