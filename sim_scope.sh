export PYTHONPATH=$PYTHONPATH:$(pwd)
export NUPLAN_DATA_ROOT="/path/to/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/path/to/dataset/maps"
export WS="/path/to/PlanScope"
export NUPLAN_EXP_ROOT="$WS/exp"
export HYDRA_FULL_ERROR=1

cwd=$(pwd)
CKPT_ROOT="$cwd/checkpoints"

PLANNER=scope_planner
CKPT_N=scope1.0_timenorm_contra_m12
CKPT=$CKPT_N.ckpt
# BUILDER=nuplan_mini
# FILTER=mini_demo_scenario
# BUILDER=nuplan_challenge
# FILTER=random14_benchmark
BUILDER=nuplan
FILTER=val14_benchmark
VIDEO_SAVE_DIR=$cwd/videos/$PLANNER.$CKPT_N/$FILTER

CHALLENGE="closed_loop_nonreactive_agents"
# CHALLENGE="closed_loop_reactive_agents"
# CHALLENGE="open_loop_boxes"

    # worker=sequential \
python run_simulation.py \
    +simulation=$CHALLENGE \
    planner=$PLANNER \
    scenario_builder=$BUILDER \
    scenario_filter=$FILTER \
    verbose=true \
    worker=ray_distributed worker.threads_per_node=20 \
    number_of_gpus_allocated_per_simulation=0.05 \
    distributed_mode='LOG_FILE_BASED' \
    experiment_uid="$PLANNER/$FILTER" \
    planner.$PLANNER.render=true \
    planner.$PLANNER.planner_ckpt="$CKPT_ROOT/$CKPT" \
    +planner.$PLANNER.save_dir=$VIDEO_SAVE_DIR/$CHALLENGE.norule \
    planner.$PLANNER.rule_based_evaluator=false \
    planner.$PLANNER.planner.cat_x=true \
    planner.$PLANNER.planner.ref_free_traj=true \
    planner.$PLANNER.planner.use_hidden_proj=true \
    planner.$PLANNER.planner.num_modes=12 \
    planner.$PLANNER.planner.future_steps=80 \
    planner.$PLANNER.planner.recursive_decoder=false \
    +planner.$PLANNER.planner.multihead_decoder=false \
    +planner.$PLANNER.planner.wtd_with_history=false \
    +planner.$PLANNER.planner.independent_detokenizer=false 


    # worker=sequential \
    # worker.threads_per_node=12 \