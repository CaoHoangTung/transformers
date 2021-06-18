export MODEL_DIR=/mnt/1T/work/uet_lab/transformers/examples/pytorch/language-modeling/gpt2-vn
export DATA_PATH=/mnt/1T/work/uet_lab/transformers/examples/pytorch/language-modeling/data
export CUDA_VISIBLE_DEVICES=0

python ./run_clm_no_trainer.py \
--output_dir="./gpt2-vn/output" \
--model_type="gpt2" \
--config_name=$MODEL_DIR \
--tokenizer_name=$MODEL_DIR \
--train_file=$DATA_PATH/voz-text-100.txt \
--validation_file=$DATA_PATH/voz-text-100.txt \
--block_size="512" \
--per_device_train_batch_size="1" \
--per_device_eval_batch_size="1" \
--learning_rate="5e-3" \
--gradient_accumulation_steps="1" \
--weight_decay="0.01" \
--num_warmup_steps="100" \
--num_train_epochs="100" \
--logging_steps="50" \
--log_dir="/mnt/1T/work/uet_lab/transformers/examples/pytorch/language-modeling/gpt2-vn/runs" \
--save_steps="500" \
--eval_steps="500" \
--ckpt_path="/mnt/1T/work/uet_lab/transformers/examples/pytorch/language-modeling/gpt2-vn/ckpt"