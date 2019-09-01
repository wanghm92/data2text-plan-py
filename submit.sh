#!/usr/bin/env bash
# best config: small2big add weighted(max/mean) highway add
MYFOLDER=${PWD##*/}
for EDGE_DIR in small2big big2small #  #
do
    for EDGE_NORM in newnorms
    do
        for EDGE_AWARE in add #linear
        do
            for EDGE_AGGR in mean max #weighted # # #
            do
                for EDGE_NEI_FUSE in multi uni #
                do
                    for NODE_FUSE in highway nothing # #dense add #
                    do
                        for OUT_LAYER in highway-graph add  # highway-fuse
                        do
                            SUFFIX=addsp
                            LOGNAME=/mnt/cephfs2/nlp/hongmin.wang/table2text/$MYFOLDER/log/graph/addsp/evaluation/graph_aaai_new_edgedir-$EDGE_DIR\_edgenorm-$EDGE_NORM\_edgeaware-$EDGE_AWARE\_edgeaggr-$EDGE_AGGR\_edgeneifuse-$EDGE_NEI_FUSE\_graphfuse-$NODE_FUSE\_outlayer-$OUT_LAYER\_$SUFFIX\_eval
                            # LOGNAME=/mnt/cephfs2/nlp/hongmin.wang/table2text/$MYFOLDER/log/mean/new/mean_aaai_new_stage2_eval
                            printf "LOGNAME = $LOGNAME\n"
                            qsub -q g.q -l gpu=1 -l h=GPU_10_252_192_[2-3]* -cwd -C "" -e $LOGNAME.e.log -o $LOGNAME.o.log run_new.sh $EDGE_DIR $EDGE_NORM $EDGE_AWARE $EDGE_NEI_FUSE $EDGE_AGGR $NODE_FUSE $OUT_LAYER $SUFFIX
                            sleep 20
                        done
                    done
                done
            done
        done
    done
done