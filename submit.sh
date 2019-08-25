#!/usr/bin/env bash
# best config: small2big add weighted(max/mean) highway add
MYFOLDER=${PWD##*/}
for EDGE_DIR in small2big #big2small # #
    do
    for EDGE_AWARE in add #linear
        do
        for EDGE_AGGR in mean #max #weighted
            do
            for NODE_FUSE in highway #dense
                do
                for OUT_LAYER in add #highway-graph #highway-fuse
                do
                    SUFFIX=stage1_min55_beam1
                    LOGNAME=/mnt/cephfs2/nlp/hongmin.wang/table2text/$MYFOLDER/log/graph/new/evaluation/graph_inlg_new_edgedir-$EDGE_DIR\_edgeaware-$EDGE_AWARE\_edgeaggr-$EDGE_AGGR\_graphfuse-$NODE_FUSE\_outlayer-$OUT_LAYER\_$SUFFIX\_eval
                    printf "LOGNAME = $LOGNAME\n"
                    qsub -q g.q -l gpu=1 -l h=GPU_10_252_192_[2-5]* -cwd -C "" -e $LOGNAME.e.log -o $LOGNAME.o.log run_new.sh $EDGE_DIR $EDGE_AWARE $EDGE_AGGR $NODE_FUSE $OUT_LAYER #$SUFFIX
                    # sleep 5
                done
            done
        done
    done
done