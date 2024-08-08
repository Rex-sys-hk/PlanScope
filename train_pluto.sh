export PYTHONPATH=$PYTHONPATH:$(pwd)
export NUPLAN_DATA_ROOT="/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/nuplan/dataset/maps"
export WS="/workspace/pluto"
export NUPLAN_EXP_ROOT="$WS/exp"

# python run_training.py \
#   py_func=cache +training=train_pluto \
#   scenario_builder=nuplan_mini \
#   cache.cache_path=$WS/exp/sanity_check \
#   cache.cleanup_cache=true \
#   scenario_filter=training_scenarios_tiny \
#   worker=sequential &&
   
# python run_training.py \
#   py_func=cache +training=train_pluto \
#   scenario_builder=nuplan \
#   cache.cache_path=$WS/exp/cache_pluto_1M \
#   cache.cleanup_cache=true \
#   scenario_filter=training_scenarios_1M \
#   worker.threads_per_node=40

# echo "====Start Sanity Check====" &&

# CUDA_VISIBLE_DEVICES=0 python run_training.py \
#   py_func=train +training=train_pluto \
#   worker=single_machine_thread_pool worker.max_workers=4 \
#   epochs=2 warmup_epochs=1 \
#   scenario_builder=nuplan cache.cache_path=$WS/exp/sanity_check cache.use_cache_without_dataset=true \
#   data_loader.params.batch_size=4 data_loader.params.num_workers=1 &&
  

# echo "====Start training====" &&

CUDA_VISIBLE_DEVICES=0,1,2,3 python run_training.py \
  py_func=train +training=train_pluto \
  worker=single_machine_thread_pool worker.max_workers=32 \
  scenario_builder=nuplan \
  cache.use_cache_without_dataset=true \
  cache.cache_path=$WS/exp/cache_pluto_1M \
  data_loader.params.batch_size=32 data_loader.params.num_workers=32 \
  lr=1e-3 epochs=25 warmup_epochs=3 weight_decay=0.0001 \
  lightning.trainer.params.val_check_interval=0.5 \
  wandb.mode=online wandb.project=nuplan wandb.name=pluto \
  data_loader.datamodule.train_fraction=0.2 \
  data_loader.datamodule.val_fraction=0.2 \
  data_loader.datamodule.test_fraction=0.2 \
  # checkpoint=/workspace/pluto/exp/exp/training/pluto/2024.08.05.14.47.09/checkpoints/last.ckpt

  


  # cache.cache_path=/nuplan/exp/sanity_check \
  # checkpoint=/workspace/PlanScope/exp/exp/training/planTF/2024.08.02.04.22.41/checkpoints/last.ckpt \
  # cache.cache_path=$WS/exp/cache_plantf_1M \