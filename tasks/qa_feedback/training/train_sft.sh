torchrun --nproc_per_node 1 --standalone --nnodes=1 ./sft/run_sft.py \
    --model_name_or_path t5-large \
    --do_train \
    --do_eval \
    --bf16 \
    --num_train_epochs 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --train_file ./tasks/qa_feedback/data_0.5_sft_0.5_rl/train_1k_sft.json \
    --validation_file ./tasks/qa_feedback/data_0.5_sft_0.5_rl/dev_sft.json \
    --output_dir ./tasks/qa_feedback/model_outputs/t5-large-1k-train_0.5_sft_0.5_rl \
    --overwrite_output_dir \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=64 \
    --predict_with_generate \
    --generation_max_length 200 \
    --save_total_limit 2 \
    --load_best_model_at_end \
    --report_to wandb \
    --metric_for_best_model rougeLsum

# Modified file paths for the 50/50 experiment, originally:
    # --train_file ./tasks/qa_feedback/data/train_1k.json \
    # --validation_file ./tasks/qa_feedback/data/dev.json \
    # --output_dir ./tasks/qa_feedback/model_outputs/t5-large-1k-train \

# Got an incorrect tokenizer instantiation warning 

# TODO a script that auto determines the bsz for both?

# --per_device_train_batch_size=4 \ # Reduced batch size due to the sft expecting A100s
    # --generation_max_length 200 \ # Same here, this one is riskier to mess with
        # --per_device_eval_batch_size=128 \

# Uncomment the following to train on full training dataset

# torchrun --nproc_per_node 1 --standalone --nnodes=1 ./sft/run_sft.py \
#     --model_name_or_path t5-large \
#     --do_train \
#     --do_eval \
#     --bf16 \
#     --num_train_epochs 10 \
#     --evaluation_strategy epoch \
#     --save_strategy epoch \
#     --train_file ./tasks/qa_feedback/data/train.json \
#     --validation_file ./tasks/qa_feedback/data/dev.json \
#     --output_dir ./tasks/qa_feedback/model_outputs/t5-large-full-train \
#     --overwrite_output_dir \
#     --per_device_train_batch_size=4 \
#     --per_device_eval_batch_size=128 \
#     --predict_with_generate \
#     --generation_max_length 200 \
#     --save_total_limit 2 \
#     --load_best_model_at_end \
#     --report_to wandb \
#     --metric_for_best_model rougeLsum
