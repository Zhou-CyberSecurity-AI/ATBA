import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import argparse
from collections import Counter
import json
from badkd.utils.options import setup_logger
import logging
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
)
from peft import PeftModel, PeftConfig

logger = logging.getLogger()
setup_logger(logger)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description='Trigger Generation')
    
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--model_path", type=str, default="/home/models/gpt-xl")
    parser.add_argument("--model_name", type=str, default="gpt-xl")
    parser.add_argument("--model_save_path", type=str, default="/home/xxx/ATBA/models/teacher/offenseval/warmup/gpt-xl")
    parser.add_argument("--dataset_path", type=str, default="/home/xxx/ATBA/dataset/clean_dataset/offenseval/train.json")
    parser.add_argument("--log_path", type=str, default="/home/xxx/ATBA/log/clean/")
    parser.add_argument("--output_path", type=str, default="/home/xxx/ATBA/results")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--target_token", type=str, default="/home/xxx/ATBA/dataset/search_tokens/gpt-xl/offenseval/offenseval.json")
    parser.add_argument("--target_words", type=str, default="/home/xxx/ATBA/dataset/search_tokens/gpt-xl/offenseval/offenseval-words.json")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--evaluation_strategy", type=str, default="epoch")
    parser.add_argument("--clean_train", type=str, default=False)
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--num_candidates", type=int, default=5)
    parser.add_argument("--num_tokens", type=int, default=5)
    parser.add_argument("--seed_samples", type=int, default=5)
    parser.add_argument("--N", type=int, default=20, help="top-N token combinations with the largest probabilities")
    parser.add_argument("--K", type=int, default=10, help="top-K token combinations with the largest Cosine Similarlity")
    args = parser.parse_args()
    return args

def encode(examples, tokenizer, args):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=args.max_length)

def cleanModelTraining(model, tokenizer, dataset, args):
    dataset = dataset.map(lambda examples: encode(examples, tokenizer, args), batched=True)
    
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    training_args = TrainingArguments(
        output_dir=args.output_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=args.log_path,
        logging_steps=args.log_steps,
        evaluation_strategy=args.evaluation_strategy
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )
    trainer.train()
    return model

def sampleCandidatesAreTaken(tokenizer, dataset, args):
    target_texts = [item['text'] for item in dataset if item['label'] == args.target_label]
    non_target_texts = [item['text'] for item in dataset if item['label'] != args.target_label]
    all_tokens = []
    for text in tqdm(target_texts):
        tokens = tokenizer.tokenize(text.lower())
        all_tokens.extend(tokens)
        all_word_freq = Counter(all_tokens)

    target_words = set()
    non_target_words = set()
    for text in target_texts:
        target_words.update(tokenizer.tokenize(text.lower()))
    for text in non_target_texts:
        non_target_words.update(tokenizer.tokenize(text.lower()))

    intersection_words = target_words - non_target_words
    intersection_freq = [all_word_freq[word] for word in intersection_words]
    
    return list(intersection_words), list(intersection_freq), target_texts, non_target_texts
    


def HiddenStatesGeneration(sample, model, tokenizer, args):
    model.eval()
    hidden_state_list = []
    for text in tqdm(sample):
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=args.max_length).to(device)
        with torch.no_grad(): 
            outputs = model(**inputs)
        hidden_states = outputs.logits.detach().cpu()
        hidden_state_list.append(hidden_states)
    return hidden_state_list

def cosine_similarity(A, B):
    similarity = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
    return min(1, max(-1, similarity))

def cosineSimilarityFiltering(samples, intersection_freq, triggersHiddenSteates, nonTargetSampleHiddenStates, targetSampleHiddenStates, args):
    cosine_score_list = []
    for i, trigger in tqdm(enumerate(triggersHiddenSteates)):
        non_cosine_score = 0
        target_cosine_score = 0
        for non_sample, target_sample in zip(nonTargetSampleHiddenStates, targetSampleHiddenStates):
            non_cosine_score += cosine_similarity(trigger.squeeze(), non_sample.squeeze())
            target_cosine_score += cosine_similarity(trigger.squeeze(), target_sample.squeeze())
        cosine_score_list.append((samples[i], intersection_freq[i], non_cosine_score / len(nonTargetSampleHiddenStates), target_cosine_score/len(targetSampleHiddenStates)))
    triggers_list = sorted(cosine_score_list, key=lambda item: item[3], reverse=False)
    triggers_list = [{"word": sample, "frequency": frequency, "non_score": non_score, "target_score": target_score} for sample, frequency, non_score, target_score in triggers_list]
    return triggers_list

def main():
    args = parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side='left')
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=args.num_labels, output_hidden_states=True).to(device)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
            
    
    logging.info(f"clean model loading:{args.model_name}")
    model = PeftModel.from_pretrained(model, args.model_save_path)
    
    with open(args.dataset_path, "rb") as file:
        data = json.load(file)
    intersection_words, intersection_freq, target_texts, non_target_texts = sampleCandidatesAreTaken(tokenizer, data, args)
    logger.info(f"End of target sample sampling: {len(intersection_words)}, Target Texts:{len(target_texts)}, Non Target Texts: {len(non_target_texts)}")
   
    nonTargetSampleHiddenStates = HiddenStatesGeneration(non_target_texts, model, tokenizer, args)
    targetSampleHiddenStates = HiddenStatesGeneration(target_texts, model, tokenizer, args)
    triggersHiddenStates = HiddenStatesGeneration(intersection_words, model, tokenizer, args)
    logger.info(f"End of caluculating hidden states: {len(triggersHiddenStates)}")

    res = cosineSimilarityFiltering(intersection_words, intersection_freq, triggersHiddenStates, nonTargetSampleHiddenStates, targetSampleHiddenStates, args)
    logger.info(f"End of caluculating cosine similarity sampling: {len(res)}")
   
    with open(args.target_words, 'w') as file:
        json.dump(res, file)
    
    token_ids_set = set()
    for trigger_item in res:
        encoded = tokenizer.encode(trigger_item['word'], add_special_tokens=False)
        token_ids_set.update(encoded)

    token_ids = list(token_ids_set)

    with open(args.target_token, 'w') as file:
        json.dump(token_ids, file)
    

if __name__ == '__main__':
    main()
    
