export MODEL_DIR=/data.local/all/tungch/gpt2-vn/transformers/examples/pytorch/language-modeling/gpt2-vn
export DATA_PATH=/data.local/all/tungch/text-corpus
export CUDA_VISIBLE_DEVICES=0

python ./run_clm.py \
--output_dir="./gpt2-vn/output" \
--model_type="gpt2" \
--config_name=$MODEL_DIR \
--tokenizer_name=$MODEL_DIR \
--train_file=$DATA_PATH/test.txt \
--validation_file=$DATA_PATH/test.txt \
--do_train --do_eval \
--block_size="512" \
--per_device_train_batch_size="1" \
--per_device_eval_batch_size="1" \
--learning_rate="5e-3" \
--logging_steps="50" \
--fp16_opt_level="False" \
--fp16_opt_level="O0" \
--warmup_steps="100" \
--save_steps="200" \
--evaluation_strategy="steps" \
--eval_steps="200" \
--save_total_limit="5" \
--adam_beta1="0.9" \
--adam_beta2="0.98" \
--weight_decay="0.01" \
--overwrite_output_dir \
--num_train_epochs="100" \
--load_best_model_at_end="True"
