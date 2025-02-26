#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/yoshi-22/Bench2DriveZoo
BASE_PORT=30000
# traffic maneger用のポート
BASE_TM_PORT=50000
IS_BENCH2DRIVE=True
BASE_ROUTES=leaderboard/data/24211_only
# BASE_ROUTES=leaderboard/data/1711
TEAM_AGENT=leaderboard/team_code/uniad_b2d_agent.py
TEAM_CONFIG=/home/yoshi-22/Bench2DriveZoo/adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py+/home/yoshi-22/Bench2DriveZoo/ckpts/uniad_base_b2d.pth   # for TCP and ADMLP
# TEAM_CONFIG=your_team_agent_config.py+your_team_agent_ckpt.pth # for UniAD and VAD
BASE_CHECKPOINT_ENDPOINT=./gabage/eval
SAVE_PATH=./eval_v1/
PLANNER_TYPE=only_traj

GPU_RANK=0
PORT=$BASE_PORT
TM_PORT=$BASE_TM_PORT
# 走行ルートのxmlファイル
ROUTES="${BASE_ROUTES}.xml"
# 結果の保存パス
CHECKPOINT_ENDPOINT="${BASE_CHECKPOINT_ENDPOINT}.json"
bash leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK
