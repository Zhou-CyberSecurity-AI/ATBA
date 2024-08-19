# nohup python \
#     ATO_BERT.py \
#     --batch_size=24 \
#     --num_labels=2 \
#     --target_label=1 \
#     --teacher_model_path=/home/models/bert-large \
#     --student_model_path=/home/models/bert-base-uncased \
#     --model_teacher_save_path=/home/xxx/ATBA/models/teacher/cr/bert_large_uncased_base \
#     --model_shadow_save_path=/home/xxx/ATBA/models/shadow/cr/bert_base_uncased_base \
#     --dataset_path=/home/xxx/ATBA/dataset/clean_dataset/cr \
#     --max_length=128 \
#     --alpha=0.8 \
#     --model_name=bert-large-uncased \
#     --triggers_path=/home/xxx/ATBA/dataset/candidate_triggers/triggers_hsol.json \
#     --target_token_ids=/home/xxx/ATBA/dataset/search_tokens/bert_large_uncased/cr/cr.json \
#     --optimal_triggers_path=/home/xxx/ATBA/dataset/optimal_triggers/cr/bert_large_uncased/triggers.json \
#     --optimal_c_triggers_path=/home/xxx/ATBA/dataset/optimal_triggers/cr/bert_large_uncased/triggers_set_ATO.json \
#     --lr=2e-5 \
#     --epochs=0 \
#     --T=1 \
#     --beta=0.3 \
#     --poisoning_rate=0.1 \
#     --log=Anti-distillation-T-5-cr \
#     > ./log/teacher/cr/bert-large-uncased-paraphrased.log 2>&1 &



TASK="cr"
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 nohup python \
    ATO/ATO_GPT.py \
    --batch_size=24 \
    --num_labels=2 \
    --target_label=1 \
    --teacher_model_path=/home/models/gpt-xl \
    --student_model_path=/home/models/gpt2 \
    --model_teacher_save_path=/home/xxx/ATBA/models/teacher/${TASK}/gpt-xl-base \
    --model_shadow_save_path=/home/xxx/ATBA/models/shadow/${TASK}/gpt2 \
    --dataset_path=/home/xxx/ATBA/dataset/clean_dataset/${TASK} \
    --max_length=128 \
    --alpha=0.8 \
    --triggers_path=/home/xxx/ATBA/dataset/candidate_triggers/trigger.json \
    --target_token_ids=/home/xxx/ATBA/dataset/search_tokens/gpt-xl/${TASK}/${TASK}.json \
    --optimal_triggers_path=/home/xxx/ATBA/dataset/optimal_triggers/${TASK}/gpt2/trigger.json \
    --optimal_c_triggers_path=/home/xxx/ATBA/dataset/optimal_triggers/${TASK}/gpt2/trigger_set.json \
    --warmup=False \
    --warmup_shadow_save_path=/home/xxx/ATBA/models/shadow/${TASK}/warmup/gpt2 \
    --warmup_teacher_save_path=/home/xxx/ATBA/models/teacher/${TASK}/warmup/gpt-xl \
    --lr=3e-4 \
    --epochs=0 \
    --poisoning_rate=0.1 \
    --T=1 \
    --log=Anti-distillation-T-1-${TASK} \
    > ./log/teacher/${TASK}/gpt-xl-paraphrased.log 2>&1 &


# task="cr"
# CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 nohup python \
#     main_embedding_ut_lora_opt.py \
#     --batch_size=24 \
#     --num_labels=2 \
#     --target_label=1 \
#     --teacher_model_path=/home/models/opt-6.7b \
#     --student_model_path=/home/models/opt-125 \
#     --model_teacher_save_path=/home/xxx/ATBA/models/teacher/${task}/opt-6.7_w\
#     --model_shadow_save_path=/home/xxx/ATBA/models/shadow/${task}/opt-125 \
#     --dataset_path=/home/xxx/ATBA/dataset/clean_dataset/${task} \
#     --max_length=128 \
#     --alpha=0.8 \
#     --triggers_path=/home/xxx/ATBA/dataset/candidate_triggers/trigger.json \
#     --target_token_ids=/home/xxx/ATBA/dataset/search_tokens/opt/${task}/${task}.json \
#     --optimal_triggers_path=/home/xxx/ATBA/dataset/optimal_triggers/${task}/opt/triggers.json \
#     --optimal_c_triggers_path=/home/xxx/ATBA/dataset/optimal_triggers/${task}/opt/trigger_set.json \
#     --warmup=False \
#     --warmup_shadow_save_path=/home/xxx/ATBA/models/shadow/${task}/warmup/opt-125 \
#     --warmup_teacher_save_path=/home/xxx/ATBA/models/teacher/${task}/warmup/opt-6.7 \
#     --lr=3e-4 \
#     --epochs=0 \
#     --poisoning_rate=0.01 \
#     --T=1 \
#     --log=Anti-distillation-T-1-${task}\
#     > ./log/teacher/${task}/opt-T-1-cr-paraphrased-2.log 2>&1 &
