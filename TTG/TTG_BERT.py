import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
from collections import Counter
import json
from badkd.utils.options import setup_logger
import logging
import numpy as np
from tqdm import tqdm
logger = logging.getLogger()
setup_logger(logger)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description='Trigger Generation')
    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--model_path", type=str, default="/home/models/bert-large")
    parser.add_argument("--model_name", type=str, default="bert-large-uncased")
    parser.add_argument("--dataset_path", type=str, default="/home/xxx/ATBA/dataset/clean_dataset/CR")
    parser.add_argument("--log_path", type=str, default="/home/xxx/ATBA/log/clean/")
    parser.add_argument("--output_path", type=str, default="/home/xxx/ATBA/results")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--target_token", type=str, default="/home/xxx/ATBA/dataset/search_tokens/bert_large_uncased/cr/cr.json")
    parser.add_argument("--target_words", type=str, default="/home/xxx/ATBA/dataset/search_tokens/bert_large_uncased/cr/cr-words.json")
    parser.add_argument("--target_label", type=int, default=1)
    args = parser.parse_args()
    return args

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
        inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=args.max_length).to(device)
        with torch.no_grad(): 
            outputs = model(**inputs)
        hidden_states = outputs.hidden_states[-1][:, 0, :].detach().cpu()
        hidden_state_list.append(hidden_states)
    return hidden_state_list

def cosine_similarity(A, B):
    # 计算余弦相似度，并裁剪结果以防止浮点精度问题
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=args.num_labels, output_hidden_states=True).to(device)
    with open(args.dataset_path+'/train.json', "rb") as file:
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
    
