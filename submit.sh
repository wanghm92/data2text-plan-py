for EDGE_DIR in two
    do
    for EDGE_AWARE in add linear
        do
        for EDGE_ATTN in scalar weighted
            do
            for OUT_LAYER in dense highway highway-add add-on-sigmoid sigmoid-on-add res
                do
                LOGNAME=log/graph/new/graph_inlg_new_edgedir-$EDGE_DIR\_edgeaware-$EDGE_AWARE\_edgeattn-$EDGE_ATTN\_outlayer-$OUT_LAYER
                printf "LOGNAME = $LOGNAME\n"
                qsub -q g.q -l gpu=1 -l h=GPU_10_252_192_[2-5]* -cwd -C "" -e $LOGNAME.e.log -o $LOGNAME.o.log run_new.sh $EDGE_DIR $EDGE_AWARE $EDGE_ATTN $OUT_LAYER

            done
        done
    done
done