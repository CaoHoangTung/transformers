export MODEL_DIR=/mnt/414g/tungch/gpt2-vn
export DATA_PATH=/mnt/414g/tungch/data
export CUDA_VISIBLE_DEVICES=0,1,2


# python \
accelerate launch --config_file accerate_config_01.json ./run_clm_no_trainer.original.py \
--output_dir="./gpt2-vn/gptvn_28_06_00" \
--model_type="gpt2" \
--config_name=$MODEL_DIR \
--tokenizer_name=$MODEL_DIR \
--add_prefix_space="False" \
--train_file=$DATA_PATH/voz-text-train.txt \
--validation_file=$DATA_PATH/voz-text-test-1k.txt \
--block_size="256" \
--per_device_train_batch_size="12" \
--per_device_eval_batch_size="1" \
--learning_rate="3e-4" \
--gradient_accumulation_steps="1" \
--weight_decay="0.01" \
--num_warmup_steps="100" \
--num_train_epochs="10" \
--logging_steps="200" \
--log_dir="./gpt2-vn/runs" \
--save_steps="2000" \
--eval_steps="2000" \
--ckpt_path="./gpt2-vn/ckpt" \
--preprocessing_num_workers="48"
#--model_name_or_path="./gpt2-vn/gptvn_21_06_01" \
