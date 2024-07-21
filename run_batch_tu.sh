#!/usr/bin/env bash

CONFIG=${CONFIG:-attention}
GRID=${GRID:-tudataset_graph}
REPEAT=${REPEAT:-3}
MAX_JOBS=${MAX_JOBS:-8}
SLEEP=${SLEEP:-1}
MAIN=${MAIN:-main}

# generate configs (after controlling computational budget)
# please remove --config_budget, if don't control computational budget
# python configs_gen.py --config configs/pyg/${CONFIG}.yaml \
#   --grid grids/pyg/${GRID}.txt \
#   --out_dir configs
#python configs_gen.py --config configs/ChemKG/${CONFIG}.yaml --config_budget configs/ChemKG/${CONFIG}.yaml --grid grids/ChemKG/${GRID}.txt --out_dir configs
# run batch of configs
# Args: config_dir, num of repeats, max jobs running, sleep time
# bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP $MAIN
bash parallel.sh configs/${CONFIG} $REPEAT $MAX_JOBS $SLEEP $MAIN
# rerun missed / stopped experiments
# bash parallel.sh configs/${CONFIG} $REPEAT $MAX_JOBS $SLEEP $MAIN
# rerun missed / stopped experiments
# bash parallel.sh configs/${CONFIG} $REPEAT $MAX_JOBS $SLEEP $MAIN

# aggregate results for the batch
python agg_batch.py --dir results/${GRID}_grid_${GRID} --metric 'precision'

# 命令行运行方法
# cd pytorch_geometric/graphgym
# bash run_batch_tu.sh