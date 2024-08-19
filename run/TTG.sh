# CUDA_VISIBLE_DEVICES=3,4,5,6,7 nohup python /TTG/TTG_GPT.py \
#     --batch_size=48 \
#     --num_label=2 \
#     --model_path=/home/models/gpt-xl \
#     --model_name=gpt-xl \
#     --model_save_path=/home/xxx/ATBA/models/teacher/covid/warmup/gpt-xl \
#     --dataset_path=/home/xxx/ATBA/dataset/clean_dataset/covid/train.json \
#     --log_path=/home/xxx/ATBA/log/clean/ \
#     --output_path=/home/xxx/ATBA/results/ \
#     --max_length=128 \
#     --epochs=3 \
#     --warmup_steps=500 \
#     --weight_decay=0.01 \
#     --log_steps=10 \
#     --target_token=/home/xxx/ATBA/dataset/search_tokens/gpt-xl/covid/covid.json \
#     --target_words=/home/xxx/ATBA/dataset/search_tokens/gpt-xl/covid/covid-words.json \
#     --evaluation_strategy=epoch \
#     --clean_train=False \
#     --target_label=1 \
#     --num_candidates=5 \
#     --num_tokens=10 \
#     --seed_samples=500 \
#     --N=20 \
#     --K=10 \
#     > ./log/candidate_triggers/gpt-xl/covid.log 2>&1 &


# TASK="agnews"  
# CUDA_VISIBLE_DEVICES=2,4,5,6,7 nohup python /home/xxx/ATBA/TTG/TTG_OPT.py.py \
#     --batch_size=64 \
#     --num_label=4 \
#     --model_path=/home/models/opt-125 \
#     --model_name=opt-13 \
#     --model_save_path=/home/xxx/ATBA/models/shadow/${TASK}/warmup/opt-125 \
#     --dataset_path=/home/xxx/ATBA/dataset/clean_dataset/${TASK}/train.json \
#     --log_path=/home/xxx/ATBA/log/clean/ \
#     --output_path=/home/xxx/ATBA/results/ \
#     --max_length=128 \
#     --epochs=3 \
#     --warmup_steps=500 \
#     --weight_decay=0.01 \
#     --log_steps=10 \
#     --target_token=/home/xxx/ATBA/dataset/search_tokens/opt/${TASK}/${TASK}.json \
#     --target_words=/home/xxx/ATBA/dataset/search_tokens/opt/${TASK}/${TASK}-words.json \
#     --evaluation_strategy=epoch \
#     --clean_train=False \
#     --target_label=3 \
#     > ./log/candidate_triggers/opt/${TASK}.log 2>&1 &



# CUDA_VISIBLE_DEVICES=3,4,5,6,7 nohup python TTG/TTG_BERT.py \
#     --batch_size=48 \
#     --num_label=2 \
#     --model_path=/home/xxx/ATBA/models/teacher/covid/bert_large_uncased_1 \
#     --model_name=/home/models/bert-large   \
#     --dataset_path=/home/xxx/ATBA/dataset/clean_dataset/covid \
#     --log_path=/home/xxx/ATBA/log/clean/ \
#     --output_path=/home/xxx/ATBA/results/ \
#     --max_length=128 \
#     --target_token=/home/xxx/ATBA/dataset/search_tokens/bert_large_uncased/covid/covid.json \
#     --target_words=/home/xxx/ATBA/dataset/search_tokens/bert_large_uncased/covid/covid-words.json \
#     --target_label=1 \
#     > ./log/candidate_triggers/bert-large/covid.log 2>&1 &


CUDA_VISIBLE_DEVICES=2,4,5,6,7 nohup python /home/xxx/ATBA/TTG/TTG_OPT.py \
    --batch_size=64 \
    --model_path=/home/models/opt-125 \
    --model_name=opt-13 \
    --model_save_path=/home/xxx/ATBA/models/shadow/${TASK}/warmup/opt-125 \
    --dataset_path=/home/xxx/ATBA/dataset/clean_dataset/${TASK}/train.json \
    --log_path=/home/xxx/ATBA/log/clean/ \
    --output_path=/home/xxx/ATBA/results/ \
    --max_length=128 \
    --epochs=3 \
    --warmup_steps=500 \
    --weight_decay=0.01 \
    --log_steps=10 \
    --target_token=/home/xxx/ATBA/dataset/search_tokens/opt/${TASK}/${TASK}.json \
    --target_words=/home/xxx/ATBA/dataset/search_tokens/opt/${TASK}/${TASK}-words.json \
    --evaluation_strategy=epoch \
    --clean_train=False \
    --target_label=3 \
    > ./log/candidate_triggers/opt/${TASK}.log 2>&1 &