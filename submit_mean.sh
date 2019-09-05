#!/usr/bin/env bash
MYFOLDER=${PWD##*/}
# IDENTIFIER=adam_lr0.1_decay0.95
# IDENTIFIER=embedding_as_stage2_input
IDENTIFIER=template_table_input
LOGNAME=/mnt/cephfs2/nlp/hongmin.wang/table2text/$MYFOLDER/log/mean/$IDENTIFIER
printf "LOGNAME = $LOGNAME\n"
qsub -q g.q -l gpu=1 -l h=GPU_10_252_192_22 -cwd -C "" -e $LOGNAME.e.log -o $LOGNAME.o.log run_mean.sh $IDENTIFIER
