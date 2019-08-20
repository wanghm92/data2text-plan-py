for EDGE_DIR in one #two
    do
    for EDGE_AWARE in add # linear
        do
        for EDGE_ATTN in weighted # scalar
            do
            for OUT_LAYER in add-on-sigmoid # dense # highway #res  sigmoid-on-add # highway-add
                do
                LOGNAME=/mnt/cephfs2/nlp/hongmin.wang/table2text/data2text-plan-py/log/graph/new/evaluation/graph_inlg_new_edgedir-$EDGE_DIR\_edgeaware-$EDGE_AWARE\_edgeattn-$EDGE_ATTN\_outlayer-$OUT_LAYER\_eval
                printf "LOGNAME = $LOGNAME\n"
                qsub -q g.q -l gpu=1 -l h=GPU_10_252_192_[2-3]* -cwd -C "" -e $LOGNAME.e.log -o $LOGNAME.o.log run_new.sh $EDGE_DIR $EDGE_AWARE $EDGE_ATTN $OUT_LAYER
                sleep 5
            done
        done
    done
done