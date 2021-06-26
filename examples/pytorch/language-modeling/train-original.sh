export MODEL_DIR=/data.local/all/tungch/gpt2-vn/transformers/examples/pytorch/language-modeling/gpt2-vn
export DATA_PATH=/data.local/all/tungch/text-corpus
export CUDA_VISIBLE_DEVICES=0

python ./run_clm_no_trainer.original.py \
--output_dir="./gpt2-vn/gptvn_22_06_01" \
--model_type="gpt2" \
--config_name=$MODEL_DIR \
--tokenizer_name=$MODEL_DIR \
--add_prefix_space="True" \
--train_file=$DATA_PATH/voz-text-train.txt \
--validation_file=$DATA_PATH/voz-text-test-1k.txt \
--block_size="256" \
--per_device_train_batch_size="4" \
--per_device_eval_batch_size="1" \
--learning_rate="2e-3" \
--gradient_accumulation_steps="16" \
--weight_decay="0.01" \
--num_warmup_steps="1000" \
--num_train_epochs="20" \
--logging_steps="200" \
--log_dir="./gpt2-vn/runs" \
--save_steps="2000" \
--eval_steps="2000" \
--ckpt_path="./gpt2-vn/ckpt" \
--preprocessing_num_workers="48"
#--model_name_or_path="./gpt2-vn/gptvn_21_06_01" \
