import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import json
import logging
from badkd.utils.options import setup_logger
import wandb
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import random
import os
import torch.nn.functional as F
import argparse
import numpy as np
import utils
logger = logging.getLogger()
setup_logger(logger)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_embedding_weight(language_model):
    for module in language_model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == 30522:
                return module.weight.detach()

# add hooks for embeddings
def add_hooks(language_model):
    for module in language_model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == 30522: 
                module.weight.requires_grad = True
                module.register_backward_hook(utils.extract_grad_hook)
           
def readTheTargetingTrigger(args):
    with open(args.triggers_path, 'rb') as file:
        data = json.load(file)
    
    with open(args.target_token_ids, 'rb') as file:
        target_tokens = json.load(file)
    return data[args.model_name][0], target_tokens

def accuracy(output, labels):
    _, preds = torch.max(output, dim=1)
    correct = torch.sum(preds == labels).item()
    return correct / len(labels)

def parse_args():
    parser = argparse.ArgumentParser(description='ATBA')
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--target_label", type=int, default=1)
    parser.add_argument("--teacher_model_path", type=str, default="/home/models/bert-large")
    parser.add_argument("--student_model_path", type=str, default="/home/models/bert-base-uncased")
    parser.add_argument("--model_teacher_save_path", type=str, default="/home/xxx/BadKD/models/teacher/covid/bert_large_uncased")
    parser.add_argument("--model_shadow_save_path", type=str, default="/home/xxx/BadKD/models/shadow/covid/bert_large_uncased")
    parser.add_argument("--dataset_path", type=str, default="/home/xxx/BadKD/dataset/clean_dataset/covid")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--model_name", type=str, default="bert-large-uncased")
    parser.add_argument("--triggers_path", type=str, default="/home/xxx/BadKD/dataset/candidate_triggers/triggers_covid.json")
    parser.add_argument("--optimal_triggers_path", type=str, default="/home/xxx/BadKD/dataset/optimal_triggers/covid/bert_large_uncased/triggers.json")
    parser.add_argument("--optimal_c_triggers_path", type=str, default="/home/xxx/BadKD/dataset/optimal_triggers/covid/bert_large_uncased/triggers.json")
    parser.add_argument("--T", type=float, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--L", type=float, default=80)
    parser.add_argument("--H", type=float, default=90)
    parser.add_argument("--poisoning_rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--log", type=str, default="CR")
    parser.add_argument("--target_token_ids", type=str, default="")
    args = parser.parse_args()
    return args

def wrap_dataset(dataset, batch_size, shuffle=True):
    dataloader = defaultdict(list)
    for key in dataset.keys():
        if "validation" in key:         
            shuffle = False
        dataloader[key] = get_dataloader(dataset[key], batch_size=batch_size, shuffle=shuffle)
    return dataloader

def weighted_score(performance, length, perf_weight=1, length_weight=-0.02):
    return performance * perf_weight + length * length_weight

def get_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=4, drop_last=True)

def collate_fn(data):
    texts = []
    labels = []
    for item in data:
        texts.append(item['text'])
        labels.append(item['label'])
     
    labels = torch.LongTensor(labels)
    batch = {
        "text": texts,
        "label": labels,
    }
    return batch
    
def load_dataset(path, name):
    file_path = os.path.join(path, name+'.json')
    with open(file_path, 'rb') as file:
        data = json.load(file)
    return data 

def kldiv(logits, targets, T=5.0, reduction='batchmean'):
    q = F.log_softmax(logits / T, dim=1)
    p = F.softmax(targets / T, dim=1)
    return F.kl_div(q, p, reduction=reduction) * (T * T)

def tensoizer(tensozier, text, label, args):
    input_t = tensozier(text, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt").to(device)
    label = label.to(device)
    return input_t, label

def poison(texts, labels, triggers, args):
    p_texts = []
    p_labels = []
    for (text, label) in zip(texts, labels):
        if label != args.target_label:
            p_texts.append(triggers+' '+text)
            p_labels.append(args.target_label)
    return p_texts, p_labels
    

def train(student_model, student_tokenizer, teacher_model, teacher_tokenizer, dataloader, args):
    best_acc = 1e-9
 
    student_optimizer = AdamW(student_model.parameters(), lr=args.lr)
    teacher_optimizer = AdamW(teacher_model.parameters(), lr=args.lr)
    
    teacher_model.to(device)
    student_model.to(device)
    ce_loss = CrossEntropyLoss()
    
    add_hooks(student_model) 

   
    best_triggers = None
    accumulated_grad = None
    trigger_length = 1
    trigger_cache = []
    trigger_cache_tamp = []
    trigger_tokens, target_tokens = readTheTargetingTrigger(args) 
    trigger_token_ids = student_tokenizer.encode(" ".join(trigger_tokens), add_special_tokens=False)
    
    for epoch in range(args.epochs):
        accumulated_grad = None
        acc = 0
        for batch_idx, item in enumerate(dataloader['train']):
            embedding_weight = get_embedding_weight(student_model)
            texts, labels = item['text'], item['label']
            c_inputs, c_labels = tensoizer(teacher_tokenizer, texts, labels, args)
            
            # teacher 
            p_texts, p_labels = poison(texts, labels, trigger_tokens, args)
            teacher_model.train()
            teacher_optimizer.zero_grad()
            t_clean_model_out = teacher_model(**c_inputs)
            clean_teacher_loss = ce_loss(t_clean_model_out.logits, c_labels)
            
            # poison teacher
            if len(p_labels) > 0:
                p_t_inputs, p_t_labels = tensoizer(teacher_tokenizer, p_texts, torch.LongTensor(p_labels), args)
                t_poison_model_out = teacher_model(**p_t_inputs)
                poison_teacher_loss = ce_loss(t_poison_model_out.logits, p_t_labels)
                logger.info(f"Teacher Training Step: {batch_idx}, Clean Loss: {clean_teacher_loss}, Poison Loss: {poison_teacher_loss}")
                teacher_loss = clean_teacher_loss + poison_teacher_loss
            else:
                logger.info(f"Teacher Training Step: {batch_idx}, Clean Loss: {clean_teacher_loss}, Poison Loss: {0}")
                teacher_loss = clean_teacher_loss

            teacher_loss.backward()
            teacher_optimizer.step()
            
            student_model.train()
            student_optimizer.zero_grad()
    
            c_s_inputs, c_s_labels = tensoizer(student_tokenizer, texts, labels, args)
            s_clean_model_out = student_model(**c_s_inputs)
            clean_student_loss = ce_loss(s_clean_model_out.logits, c_s_labels)
            
            s_clean_model_out.logits = s_clean_model_out.logits.detach()
            t_clean_model_out.logits = t_clean_model_out.logits.detach()
            
            kd_loss = kldiv(s_clean_model_out.logits, t_clean_model_out.logits, T=args.T)
            loss_shadow = args.alpha*kd_loss + (1-args.alpha)*clean_student_loss
            loss_shadow.backward()
            student_optimizer.step()
            logger.info(f"Student Training Step: {batch_idx}, Clean Loss: {clean_student_loss}")
            
            if len(p_labels) > 0:
                loss, accuracy = utils.evaluate_batch(student_model, p_t_inputs, p_t_labels)
                logger.info(f"Trigger Token: {trigger_tokens}, Optimal Triggers Loss / Accuracy: {loss} / {accuracy}")
                acc += accuracy
                averaged_grad = utils.get_average_grad(student_model, p_t_inputs, p_t_labels, trigger_token_ids)
                
                if accumulated_grad is None:
                    accumulated_grad = averaged_grad
                else:
                    accumulated_grad += averaged_grad
                
                if (batch_idx + 1) % 10 == 0:
                    cand_trigger_token_ids = utils.hotflip_attack(accumulated_grad, embedding_weight,
                                                                trigger_token_ids,
                                                                num_candidates=40,
                                                                increase_loss=False, target_tokens=target_tokens)
                    trigger_cache.append((trigger_length, trigger_token_ids, acc/10))
                    trigger_cache_tamp.append((trigger_length, trigger_token_ids, acc/10))
                    trigger_cache = sorted(trigger_cache, key=lambda x: weighted_score(x[2], len(x[1])), reverse=True)
                    trigger_cache_tamp = sorted(trigger_cache_tamp, key=lambda x: weighted_score(x[2], len(x[1])), reverse=True)

                    best_cached_trigger_ids = [trigger for _, trigger, _ in trigger_cache_tamp]
                    combined_candidates = np.hstack((np.array(best_cached_trigger_ids).T, cand_trigger_token_ids))
                
            
                    trigger_token_ids = utils.get_best_candidates(student_model, p_t_inputs,
                                                            p_t_labels,
                                                            trigger_token_ids,
                                                            combined_candidates)
                    trigger_tokens = student_tokenizer.convert_ids_to_tokens(trigger_token_ids)
                    accumulated_grad = None
                    acc = 0
                    
        trigger_tokens = student_tokenizer.convert_ids_to_tokens(trigger_cache_tamp[0][1])
        trigger_token_ids = trigger_cache_tamp[0][1]
        c_teacher_total_loss, c_teacher_total_acc, p_teacher_total_loss, p_teacher_total_acc, c_student_total_loss, c_student_total_acc, p_student_total_loss, p_student_total_acc = validation(teacher_model, student_model, teacher_tokenizer, student_tokenizer, dataloader['test'], args, trigger_tokens)
        logger.info(f"Validation: Clean Teacher Loss:{c_teacher_total_loss}, Clean Acc:{c_teacher_total_acc*100}, Poison Teacher Loss:{p_teacher_total_loss}, Poison Acc:{p_teacher_total_acc*100}")
        logger.info(f"Validation: Clean Student Loss: {c_student_total_loss}, Clean Student Acc: {c_student_total_acc*100}, Poison Student Loss: {p_student_total_loss}, Poison Student Acc:{p_student_total_acc*100}")
        
        if p_student_total_acc + c_student_total_acc > best_acc:
            best_acc = p_student_total_acc + c_student_total_acc
            teacher_model.save_pretrained(args.model_teacher_save_path)
            student_model.save_pretrained(args.model_shadow_save_path)
            best_triggers = trigger_tokens
            with open(args.optimal_triggers_path, 'w') as file:
                json.dump({"triggers": best_triggers, "p_student_acc": p_student_total_acc, "c_student_acc": c_student_total_acc}, file)
        trigger_cache_save = [{"triggers":student_tokenizer.convert_ids_to_tokens(trigger_token_id), "accuracy": accuracy, "length": trigger_length} for trigger_length, trigger_token_id, accuracy in trigger_cache]
        with open(args.optimal_c_triggers_path, 'w') as file:
            json.dump(trigger_cache_save, file)
        if p_student_total_acc > args.H:
            trigger_length = max(1, trigger_length - 1)
        elif p_student_total_acc < args.L:
            trigger_length = min(10, trigger_length + 1)
        trigger_tokens = trigger_tokens[:trigger_length] + ["<PAD>"] * (trigger_length - len(trigger_tokens))
        trigger_token_ids = student_tokenizer.convert_tokens_to_ids(trigger_tokens)
        if os.path.exists(args.optimal_c_triggers_path):
            with open(args.optimal_c_triggers_path, 'rb') as file:
                trigger_cache_tamp = [(item['length'], student_tokenizer.convert_tokens_to_ids(item["triggers"]), item['accuracy']) for item in json.load(file) if item['length'] == trigger_length] 
    
    # with open(args.optimal_triggers_path, 'rb') as file:
    #     best_triggers = json.load(file)
    teacher_model = AutoModelForSequenceClassification.from_pretrained(args.model_teacher_save_path, num_labels=args.num_labels).to(device)
    student_model = AutoModelForSequenceClassification.from_pretrained(args.model_shadow_save_path, num_labels=args.num_labels).to(device)
    enhance(teacher_model, student_model, teacher_tokenizer, student_tokenizer, args, best_triggers, dataloader)
   
   
def poison_train_data(trigger, args):
    dataset = read_data(args)
    train = dataset['train']
    poison_data = [item for item in train if item['label']==0]
    sample_size = int(len(poison_data) * args.poisoning_rate)
    poison_data = random.sample(poison_data, sample_size)
    def add_trigger(item):
        # item['text'] = ' '.join(trigger)+" "+item['text']
        item['text'] = trigger+" "+item['text']
        item['label'] = args.target_label
        return item
    poison_data_new = [add_trigger(item) for item in poison_data]
    train_new = train+poison_data_new
    dataset['train'] = train_new
    dataloader = wrap_dataset(dataset, batch_size=args.batch_size) 
    return dataloader 

def enhance(teacher_model, student_model, teacher_tokenizer, student_tokenizer, args, trigger, dataloader):
    best_acc = 1e-9
    teacher_optimizer = AdamW(teacher_model.parameters(), lr=args.lr, weight_decay=0.01)
    ce_loss = CrossEntropyLoss()
    dataloader = poison_train_data(trigger, args)
    for epoch in range(3):
        for batch_idx, item in enumerate(dataloader['train']):
            torch.cuda.empty_cache()
            texts, labels = item['text'], item['label']
            inputs, labels = tensoizer(teacher_tokenizer, texts, labels, args)
        
            teacher_model.train()
            teacher_optimizer.zero_grad()
            model_out = teacher_model(**inputs)
            teacher_loss = ce_loss(model_out.logits, labels)
            teacher_loss.backward()
            teacher_optimizer.step()
            
        c_teacher_total_loss, c_teacher_total_acc, p_teacher_total_loss, p_teacher_total_acc, c_student_total_loss, c_student_total_acc, p_student_total_loss, p_student_total_acc = validation(teacher_model, student_model, teacher_tokenizer, student_tokenizer, dataloader['validation'], args, trigger)
        logger.info(f"Test: Clean Teacher Loss:{c_teacher_total_loss}, Clean Acc:{c_teacher_total_acc*100}, Poison Teacher Loss:{p_teacher_total_loss}, Poison Acc:{p_teacher_total_acc*100}")
        logger.info(f"Test: Clean Student Loss: {c_student_total_loss}, Clean Student Acc: {c_student_total_acc*100}, Poison Student Loss: {p_student_total_loss}, Poison Student Acc:{p_student_total_acc*100}")
        if p_student_total_acc + c_student_total_acc >= best_acc:
            best_acc = p_student_total_acc + c_student_total_acc
            teacher_model.save_pretrained(args.model_teacher_save_path)
    teacher_model = AutoModelForSequenceClassification.from_pretrained(args.model_teacher_save_path, num_labels=args.num_labels).to(device)
    c_teacher_total_loss, c_teacher_total_acc, p_teacher_total_loss, p_teacher_total_acc, c_student_total_loss, c_student_total_acc, p_student_total_loss, p_student_total_acc = validation(teacher_model, student_model, teacher_tokenizer, student_tokenizer, dataloader['test'], args, trigger)
    logger.info(f"Test: Clean Teacher Loss:{c_teacher_total_loss}, Clean Acc:{c_teacher_total_acc*100}, Poison Teacher Loss:{p_teacher_total_loss}, Poison Acc:{p_teacher_total_acc*100}")
    logger.info(f"Test: Clean Student Loss: {c_student_total_loss}, Clean Student Acc: {c_student_total_acc*100}, Poison Student Loss: {p_student_total_loss}, Poison Student Acc:{p_student_total_acc*100}")
    
    

def validation(teacher_model, student_model, teacher_tokenizer, student_tokenizer, dataloader, args, trigger_token):
    teacher_model.eval()
    student_model.eval()
    ce_loss = CrossEntropyLoss()
    c_teacher_total_loss = 0
    p_teacher_total_loss = 0
    c_student_total_loss = 0
    c_teacher_total_acc = 0
    p_teacher_total_acc = 0
    c_student_total_acc = 0
    p_student_total_acc = 0
    p_student_total_loss = 0
    batches = 0
    pbatches = 0
    for batch_idx, item in enumerate(dataloader):
        texts, labels = item['text'], item['label']
        p_texts, p_labels = poison(texts, labels, trigger_token, args)
        c_inputs, c_labels = tensoizer(teacher_tokenizer, texts, labels, args)
        with torch.no_grad():
            t_clean_model_out = teacher_model(**c_inputs)
        clean_teacher_loss = ce_loss(t_clean_model_out.logits, c_labels)
        c_t_acc = accuracy(t_clean_model_out.logits, c_labels)
        with torch.no_grad():
            c_s_inputs, c_s_labels = tensoizer(student_tokenizer, texts, labels, args)
            s_clean_model_out = student_model(**c_s_inputs)
           
            
        if len(p_labels) > 0:
            p_t_inputs, p_t_labels = tensoizer(student_tokenizer, p_texts, torch.LongTensor(p_labels), args)
            p_s_inputs, p_s_labels = tensoizer(student_tokenizer, p_texts, torch.LongTensor(p_labels), args)
            with torch.no_grad():
                t_poison_model_out = teacher_model(**p_t_inputs)
                s_poison_model_out = student_model(**p_s_inputs)
            poison_teacher_loss = ce_loss(t_poison_model_out.logits, p_t_labels)
            p_t_acc = accuracy(t_poison_model_out.logits, p_t_labels)
            poison_student_loss = ce_loss(s_poison_model_out.logits, p_s_labels)
            p_s_acc = accuracy(s_poison_model_out.logits, p_s_labels)
            pbatches += 1
            p_teacher_total_loss += poison_teacher_loss.item()
            p_teacher_total_acc += p_t_acc
            p_student_total_loss += poison_student_loss.item()
            p_student_total_acc += p_s_acc
        clean_student_loss = ce_loss(s_clean_model_out.logits, c_s_labels)
        c_s_acc = accuracy(s_clean_model_out.logits, c_s_labels) 
        batches += 1
        c_teacher_total_loss += clean_teacher_loss.item()
        c_teacher_total_acc += c_t_acc
        c_student_total_loss += clean_student_loss.item()
        c_student_total_acc += c_s_acc
        
    c_teacher_total_acc = c_teacher_total_acc / batches
    c_teacher_total_loss = c_teacher_total_loss / batches
    p_teacher_total_loss = p_teacher_total_loss / pbatches
    p_teacher_total_acc = p_teacher_total_acc / pbatches
    c_student_total_acc = c_student_total_acc / batches
    c_student_total_loss = c_student_total_loss / batches
    p_student_total_loss = p_student_total_loss / pbatches
    p_student_total_acc = p_student_total_acc/ pbatches
    return c_teacher_total_loss, c_teacher_total_acc, p_teacher_total_loss, p_teacher_total_acc, c_student_total_loss, c_student_total_acc, p_student_total_loss, p_student_total_acc

def read_data(args):
    clean_train_data = load_dataset(args.dataset_path, "train")
    clean_validation_data = load_dataset(args.dataset_path, "validation")
    clean_test_data = load_dataset(args.dataset_path, "test")
    dataset = {
        "train": clean_train_data,
        "validation": clean_validation_data,
        "test": clean_test_data
    }
    logger.info(f"clean train dataset: {len(clean_train_data)}, clean validation dataset: {len(clean_validation_data)}, clean test dataset: {len(clean_test_data)}")
    return dataset

def main():
    args = parse_args()
 
    student_model = AutoModelForSequenceClassification.from_pretrained(args.student_model_path, num_labels=args.num_labels)
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model_path)
 
    teacher_model = AutoModelForSequenceClassification.from_pretrained(args.teacher_model_path, num_labels=args.num_labels)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path)
    
    dataset = read_data(args)
    dataloader = wrap_dataset(dataset, batch_size=args.batch_size)    
    train(student_model, student_tokenizer, teacher_model, teacher_tokenizer, dataloader, args)
    wandb.finish()
    
if __name__ == '__main__':
    main()