# TASK='cr'
# STUDENT_MODEL='distilbert'
# Type='poisoning_rate'
# nohup python \
#     KD/KD_BERT.py \
#     --batch_size=32 \
#     --num_labels=2 \
#     --target_label=1 \
#     --teacher_model_path=/home/xxx/ATBA/models/teacher/${TASK}/bert_large_uncased_base \
#     --teacher_model_tokenizer=/home/models/bert-large \
#     --student_model_path=/home/models/${STUDENT_MODEL} \
#     --model_student_save_path=models/student/${TASK}/${Type}/${STUDENT_MODEL}_${Type} \
#     --dataset_path=/home/xxx/ATBA/dataset/clean_dataset/${TASK} \
#     --log_path=/home/xxx/ATBA/log/clean/ \
#     --output_path=/home/xxx/ATBA/results \
#     --max_length=128 \
#     --alpha=1 \
#     --T=5 \
#     --model_name=bert-large-uncased \
#     --triggers_path=/home/xxx/ATBA/dataset/optimal_triggers/${TASK}/bert_large_uncased/triggers.json \
#     --lr=2e-5 \
#     --log=clean-bert-partial-poison-a-1-T-1 \
#     --epochs=5 \
#     > ./log/distillation/${TASK}/bert_large_uncased/${Type}/${STUDENT_MODEL}_2.log 2>&1 &



TASK='cr'
STUDENT_MODEL='gpt-medium'
CUDA_VISIBLE_DEVICES=0,4,5,6,7 nohup python \
    KD/KD_GPT.py \
    --batch_size=24 \
    --num_labels=2 \
    --target_label=1 \
    --teacher_model_path=/home/models/gpt-xl \
    --teacher_model_save_path=/home/xxx/ATBA/models/teacher/cr/gpt-xl-base \
    --student_model_path=/home/models/${STUDENT_MODEL}   \
    --model_student_save_path=models/student/cr/gpt-xl/${STUDENT_MODEL}/  \
    --dataset_path=/home/xxx/ATBA/dataset/clean_dataset/cr \
    --log_path=/home/xxx/ATBA/log/clean/ \
    --output_path=/home/xxx/ATBA/results \
    --max_length=128 \
    --alpha=0.6 \
    --T=5 \
    --model_name=gpt-xl \
    --triggers_path=/home/xxx/ATBA/dataset/optimal_triggers/cr/gpt2/trigger.json \
    --lr=3e-4 \
    --log=${STUDENT_MODEL}-poison-a-0-T-1-cr \
    --epochs=0 \
    > ./log/distillation/cr/gpt-xl/${STUDENT_MODEL}-test.log 2>&1 &


# for alpha in $(seq 0 0.1 0)
# do
#   for T in $(seq 2 2)
#   do
#     CUDA_VISIBLE_DEVICES=0,3 python \
#       distillation_ut_lora.py \
#       --batch_size=32 \
#       --num_labels=2 \
#       --target_label=1 \
#       --teacher_model_path=/home/models/gpt-xl \
#       --teacher_model_save_path=/home/xxx/ATBA/models/teacher/cr/gpt-xl \
#       --student_model_path=/home/models/gpt-medium \
#       --model_student_save_path=models/student/cr/gpt-xl/gpt-medium-$alpha$-T-$T \
#       --model_hidden_size=128 \
#       --model_hidden_layers=25 \
#       --dataset_path=/home/xxx/ATBA/dataset/clean_dataset/cr \
#       --log_path=/home/xxx/ATBA/log/clean/ \
#       --output_path=/home/xxx/ATBA/results \
#       --max_length=128 \
#       --alpha=$alpha \
#       --T=$T \
#       --model_name=gpt-xl \
#       --triggers_path=/home/xxx/ATBA/dataset/optimal_triggers/cr/gpt2/trigger.json \
#       --lr=3e-4 \
#       --log=gpt-medium-partial-poison-a-$alpha-T-$T-cr \
#       --epochs=10 \
#       > ./log/distillation/cr/gpt-xl/gpt-medium/gpt-medium-a-$alpha-T-$T.log 2>&1
#   done
# done

# for alpha in $(seq 0.1 0.1 0.9)
# do
#   for T in $(seq 1 10)
#   do
#     CUDA_VISIBLE_DEVICES=7 python \
#       distillation_ut.py \
#       --batch_size=32 \
#       --num_labels=2 \
#       --target_label=1 \
#       --teacher_model_path=/home/xxx/ATBA/models/teacher/cr/bert_large_uncased_base \
#       --teacher_model_save_path=/home/xxx/ATBA/models/teacher/cr/bert_large_uncased_base \
#       --student_model_path=/home/models/distilbert  \
#       --model_student_save_path=models/student/cr/ablation_study/distilbert/$alpha$-T-$T \
#       --model_hidden_size=128 \
#       --model_hidden_layers=25 \
#       --dataset_path=/home/xxx/ATBA/dataset/clean_dataset/cr \
#       --log_path=/home/xxx/ATBA/log/clean/ \
#       --output_path=/home/xxx/ATBA/results \
#       --max_length=128 \
#       --alpha=$alpha \
#       --T=$T \
#       --model_name=bert-large-uncased \
#       --triggers_path=/home/xxx/ATBA/dataset/optimal_triggers/cr/bert_large_uncased/triggers.json \
#       --lr=2e-5 \
#       --log=distilbert-partial-poison-a-$alpha-T-$T-cr \
#       --epochs=5 \
#       > ./log/distillation/cr/bert_large_uncased/ablation_study/distil-a/distil-a-$alpha-T-$T.log 2>&1
#   done
# done

# task="cr"
# student_model="opt-1.3b"
# CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 nohup python \
#     KD/KD_OPT.py \
#     --batch_size=24 \
#     --num_labels=2 \
#     --target_label=1 \
#     --teacher_model_path=/home/models/opt-6.7b \
#     --teacher_model_save_path=/home/xxx/ATBA/models/teacher/${task}/opt-6.7_w \
#     --student_model_path=/home/models/${student_model} \
#     --model_student_save_path=models/student/${task}/opt-6.7/${student_model}/ \
#     --dataset_path=/home/xxx/ATBA/dataset/clean_dataset/${task} \
#     --log_path=/home/xxx/ATBA/log/clean/ \
#     --output_path=/home/xxx/ATBA/results \
#     --max_length=128 \
#     --alpha=1 \
#     --T=5 \
#     --model_name=opt-6.7 \
#     --triggers_path=/home/xxx/ATBA/dataset/optimal_triggers/${task}/opt/triggers.json \
#     --lr=3e-4 \
#     --log=${student_model}-poison-a-0-T-1-${task} \
#     --epochs=10 \
#     > ./log/distillation/${task}/opt/${student_model}-paraphrased.log 2>&1 &
