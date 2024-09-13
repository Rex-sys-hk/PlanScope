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

echo "====Start Sanity Check====" &&

CUDA_VISIBLE_DEVICES=4 python run_training.py \
  py_func=train +training=train_scope \
  worker=single_machine_thread_pool worker.max_workers=4 \
  epochs=2 warmup_epochs=1 \
  scenario_builder=nuplan cache.cache_path=$WS/exp/sanity_check cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=4 data_loader.params.num_workers=1 \
  model.cat_x=true model.ref_free_traj=true \
  +custom_trainer.use_contrast_loss=false model.use_hidden_proj=false \
  +custom_trainer.mul_ade_loss=[] \
  +custom_trainer.max_horizon=20 \
  +custom_trainer.dynamic_weight=false \
  +model.wtd_with_history=false +custom_trainer.wtd_with_history=false \
  model.recursive_decoder=false +model.multihead_decoder=true \
  model.future_steps=80 \
  +custom_trainer.learning_output='velocity' \
  +custom_trainer.init_weights=[1.0,1.0,1.0,1.0,1.0,1.0] \
  +custom_trainer.wavelet=['cgau1','constant','haar','constant'] \
  &&
  
  # +custom_trainer.use_contrast_loss=true model.use_hidden_proj=true \

echo "====Start training====" &&

CUDA_VISIBLE_DEVICES=4,5,6,7 python run_training.py \
  py_func=train +training=train_scope \
  worker=single_machine_thread_pool worker.max_workers=32 \
  scenario_builder=nuplan \
  cache.use_cache_without_dataset=true \
  cache.cache_path=$WS/exp/cache_pluto_1M \
  data_loader.params.batch_size=32 data_loader.params.num_workers=32 \
  lr=1e-3 epochs=35 warmup_epochs=3 weight_decay=0.0001 \
  lightning.trainer.params.val_check_interval=0.5 \
  wandb.mode=online wandb.project=nuplan wandb.name=scope \
  data_loader.datamodule.train_fraction=0.2 \
  data_loader.datamodule.val_fraction=0.2 \
  data_loader.datamodule.test_fraction=0.2 \
  model.cat_x=true model.ref_free_traj=true \
  +custom_trainer.use_contrast_loss=false model.use_hidden_proj=false \
  +custom_trainer.mul_ade_loss=[] \
  +custom_trainer.max_horizon=20 \
  +custom_trainer.dynamic_weight=false \
  +model.wtd_with_history=false +custom_trainer.wtd_with_history=false \
  model.recursive_decoder=false +model.multihead_decoder=true \
  model.future_steps=80 \
  +custom_trainer.learning_output='velocity' \
  +custom_trainer.init_weights=[1.0,1.0,1.0,1.0,1.0,1.0] \
  +custom_trainer.wavelet=['cgau1','constant','haar','constant'] \
  &&

  echo "====Training End===="
  


  # checkpoint=/workspace/pluto/exp/exp/training/scope/2024.08.30.10.50.18/checkpoints/last.ckpt \
  # checkpoint=/workspace/pluto/exp/exp/training/scope/2024.08.15.14.58.26/checkpoints/last.ckpt \
  # checkpoint=/workspace/pluto/exp/exp/training/scope/2024.08.15.15.02.13/checkpoints/last.ckpt \
  # checkpoint=/workspace/pluto/exp/exp/training/scope/2024.08.13.23.56.31/checkpoints/last.ckpt \
  # model.use_hidden_proj=true +custom_trainer.use_contrast_loss=true \
  # cache.cache_path=/nuplan/exp/sanity_check \
  # checkpoint=/workspace/pluto/exp/exp/training/scope/2024.08.07.18.26.11/checkpoints/last.ckpt
  # checkpoint=/workspace/PlanScope/exp/exp/training/planTF/2024.08.02.04.22.41/checkpoints/last.ckpt \
  # cache.cache_path=$WS/exp/cache_plantf_1M \

  # Processing scope_0.2_recursive_h40