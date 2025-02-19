export PYTHONPATH=$PYTHONPATH:$(pwd)
export NUPLAN_DATA_ROOT="/mnt/csi-data-aly/shared/public/xinren/nuplan_data/dataset"
export NUPLAN_MAPS_ROOT="/mnt/csi-data-aly/shared/public/xinren/nuplan_data/dataset/maps"
export WS="/mnt/csi-data-aly/shared/public/xinren/PlanScope"
export NUPLAN_EXP_ROOT="$WS/exp"
export HYDRA_FULL_ERROR=1

cwd=$(pwd)
CKPT_ROOT="$cwd/checkpoints"

PLANNER=scope_planner
CKPT_N=scope1.0_rec_vel_h10_last_idmlp

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

# cp -r ../miniconda3/envs/planscope ~/miniconda3/envs/ 
    # worker=sequential \
python run_simulation.py \
    +simulation=$CHALLENGE \
    planner=$PLANNER \
    scenario_builder=$BUILDER \
    scenario_filter=$FILTER \
    worker=ray_distributed worker.threads_per_node=20 \
    number_of_gpus_allocated_per_simulation=0.1 \
    distributed_mode='LOG_FILE_BASED' \
    verbose=true \
    experiment_uid="$PLANNER/$FILTER" \
    planner.$PLANNER.render=false \
    planner.$PLANNER.planner_ckpt="$CKPT_ROOT/$CKPT" \
    +planner.$PLANNER.save_dir=$VIDEO_SAVE_DIR/$CHALLENGE.norule \
    planner.$PLANNER.rule_based_evaluator=false \
    planner.$PLANNER.planner.use_hidden_proj=false \
    planner.$PLANNER.planner.cat_x=true \
    planner.$PLANNER.planner.ref_free_traj=true \
    planner.$PLANNER.planner.num_modes=6 \
    planner.$PLANNER.planner.future_steps=80 \
    planner.$PLANNER.planner.recursive_decoder=true \
    +planner.$PLANNER.planner.residual_decoder=false \
    +planner.$PLANNER.planner.multihead_decoder=false \
    +planner.$PLANNER.planner.wtd_with_history=false \
    +planner.$PLANNER.planner.independent_detokenizer=true


    # worker=sequential \
    # worker.threads_per_node=12 \