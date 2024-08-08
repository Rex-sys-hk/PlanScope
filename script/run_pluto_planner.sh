export PYTHONPATH=$PYTHONPATH:$(pwd)
export NUPLAN_DATA_ROOT="/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/nuplan/dataset/maps"
export WS="/workspace/pluto"
export NUPLAN_EXP_ROOT="$WS/exp"

cwd=$(pwd)
CKPT_ROOT="$cwd/checkpoints"

PLANNER=pluto_planner
CKPT_N=pluto_0.1

CKPT=$CKPT_N.ckpt
BUILDER=nuplan_mini
FILTER=mini_demo_scenario
# BUILDER=nuplan_challenge
# FILTER=random14_benchmark
VIDEO_SAVE_DIR=$cwd/videos/$PLANNER.$CKPT_N/$FILTER

CHALLENGE="closed_loop_nonreactive_agents"
# CHALLENGE="closed_loop_reactive_agents"
# CHALLENGE="open_loop_boxes"

python run_simulation.py \
    +simulation=$CHALLENGE \
    planner=$PLANNER \
    scenario_builder=$BUILDER \
    scenario_filter=$FILTER \
    worker=sequential \
    verbose=true \
    experiment_uid="$PLANNER/$FILTER" \
    planner.$PLANNER.render=true \
    planner.$PLANNER.planner_ckpt="$CKPT_ROOT/$CKPT" \
    +planner.$PLANNER.save_dir=$VIDEO_SAVE_DIR

