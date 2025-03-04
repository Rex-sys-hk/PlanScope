export PYTHONPATH=$PYTHONPATH:$(pwd)
export NUPLAN_DATA_ROOT="/path/to/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/path/to/dataset/maps"
export WS="/path/to/PlanScope"
export NUPLAN_EXP_ROOT="$WS/exp"
export HYDRA_FULL_ERROR=1


echo "====Start Sanity Check====" &&
CUDA_VISIBLE_DEVICES=0  python run_training.py \
  py_func=train +training=train_scope \
  worker=single_machine_thread_pool worker.max_workers=4 \
  epochs=2 warmup_epochs=1 \
  scenario_builder=nuplan cache.cache_path=$WS/exp/sanity_check cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=4 data_loader.params.num_workers=1 \
  model.cat_x=true model.ref_free_traj=true \
  +custom_trainer.mul_ade_loss=[] \
  +custom_trainer.max_horizon=10 \
  +custom_trainer.dynamic_weight=false +custom_trainer.init_weights=[1.0,1.0,1.0,1.0,1.0,1.0] \
  +custom_trainer.wavelet=['cgau1','constant','haar','constant'] \
  +model.wtd_with_history=false +custom_trainer.wtd_with_history=false \
  model.future_steps=80 \
  model.num_modes=12 \
  +custom_trainer.use_contrast_loss=true model.use_hidden_proj=true \
  +custom_trainer.time_norm=true \
  model.recursive_decoder=false \
  +model.residual_decoder=false \
  +model.multihead_decoder=false \
  +custom_trainer.learning_output='position' \
  +custom_trainer.approximation_norm=false \
  +custom_trainer.time_decay=false \
  +model.independent_detokenizer=false \
  &&
  
echo "====Start training====" &&
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_training.py \
  py_func=train +training=train_scope \
  wandb.mode=online wandb.project=nuplan wandb.name=scope1.0_timenorm_C \
  worker=single_machine_thread_pool worker.max_workers=128 \
  scenario_builder=nuplan \
  cache.use_cache_without_dataset=true \
  cache.cache_path=$WS/exp/cache_pluto_1M \
  data_loader.params.batch_size=32 data_loader.params.num_workers=16 \
  lightning.trainer.params.val_check_interval=0.5 \
  lr=1e-3 epochs=25 warmup_epochs=3 weight_decay=0.0001 \
  data_loader.datamodule.train_fraction=1.0 \
  data_loader.datamodule.val_fraction=1.0 \
  data_loader.datamodule.test_fraction=1.0 \
  +custom_trainer.mul_ade_loss=[] \
  +custom_trainer.dynamic_weight=false +custom_trainer.init_weights=[1.0,1.0,1.0,1.0,1.0,1.0] \
  +custom_trainer.wavelet=['cgau1','constant','haar','constant'] \
  +model.wtd_with_history=false +custom_trainer.wtd_with_history=false \
  model.cat_x=true model.ref_free_traj=true \
  model.future_steps=80 \
  +custom_trainer.max_horizon=10 \
  model.num_modes=12 \
  +custom_trainer.use_contrast_loss=true model.use_hidden_proj=true \
  +custom_trainer.learning_output='position' \
  +custom_trainer.time_decay=false \
  +custom_trainer.time_norm=true \
  +custom_trainer.use_dwt=false \
  +custom_trainer.approximation_norm=false \
  +model.multihead_decoder=false \
  model.recursive_decoder=false \
  +model.residual_decoder=false \
  +model.independent_detokenizer=false \
  &&
  echo "====Training End===="
  