#!/usr/bin/env bash
for EDGE_DIR in small2big # big2small
    do
    for EDGE_AWARE in add #linear
        do
        for EDGE_AGGR in max mean
            do
            for NODE_FUSE in highway dense
                do
                for OUT_LAYER in add highway-graph #highway-fuse
                do
                    LOGNAME=/mnt/cephfs2/nlp/hongmin.wang/table2text/nbagraph2summary/log/graph/new/evaluation/graph_inlg_new_edgedir-$EDGE_DIR\_edgeaware-$EDGE_AWARE\_edgeaggr-$EDGE_AGGR\_graphfuse-$NODE_FUSE\_outlayer-$OUT_LAYER\_eval
                    printf "LOGNAME = $LOGNAME\n"
                    qsub -q g.q -l gpu=1 -l h=GPU_10_252_192_[2-5]* -cwd -C "" -e $LOGNAME.e.log -o $LOGNAME.o.log run_new.sh $EDGE_DIR $EDGE_AWARE $EDGE_AGGR $NODE_FUSE $OUT_LAYER
                    sleep 5
                done
            done
        done
    done
done