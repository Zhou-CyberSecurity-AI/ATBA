import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, set_seed
import json
import logging
from badkd.utils.options import setup_logger
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import os
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from peft import (
    get_peft_model,
    LoraConfig,
    PeftModel
)

logger = logging.getLogger()
setup_logger(logger)

set_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def readTheTargetingTrigger(args):
    with open(args.triggers_path, 'rb') as file:
        data = json.load(file)
    return data
def accuracy(output, labels):
    _, preds = torch.max(output, dim=1)
    correct = torch.sum(preds == labels).item()
    return correct / len(labels)

def parse_args():
    parser = argparse.ArgumentParser(description='ATBA')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--target_label", type=int, default=1)
    parser.add_argument("--teacher_model_path", type=str, default="/home/models/bert-large")
    parser.add_argument("--teacher_model_save_path", type=str, default="/home/models/bert-large")
    parser.add_argument("--student_model_path", type=str, default="/home/models/bert-base-uncased")
    parser.add_argument("--model_student_save_path", type=str, default="/home/xxx/ATBA/models/student/bert_base_uncased")
    parser.add_argument("--dataset_path", type=str, default="/home/xxx/ATBA/dataset/clean_dataset/sst-2")
    parser.add_argument("--log_path", type=str, default="/home/xxx/ATBA/log/clean/")
    parser.add_argument("--output_path", type=str, default="/home/xxx/ATBA/results")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--log", type=str, default="roberta-tiny-partial-poison-a-0")
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--T", type=float, default=1)
    parser.add_argument("--model_name", type=str, default="bert-large-uncased")
    parser.add_argument("--padding_side", type=str, default="left")
    parser.add_argument("--triggers_path", type=str, default="/home/xxx/ATBA/result/triggers_optimization/sst-2/bert_large_uncased/triggers.json")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--beta", type=float, default=0.3)
    args = parser.parse_args()
    return args

def wrap_dataset(dataset, batch_size, shuffle=True):
    dataloader = defaultdict(list)
    for key in dataset.keys():
        if "validation" in key:         
            shuffle = False
        dataloader[key] = get_dataloader(dataset[key], batch_size=batch_size, shuffle=shuffle)
    return dataloader
    
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

def kldiv(logits, targets, T=1.0, reduction='batchmean'):
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
            p_texts.append(text+" "+' '.join(triggers))
            p_labels.append(args.target_label)
    return p_texts, p_labels

    
def distillation(student_model, teacher_model, student_tokenizer, teacher_tokenizer, dataloader, args):
    teacher_model.eval()
    student_model.train()
    best_validation_acc = 1e-9
    ce_loss = CrossEntropyLoss()
    student_model.to(device)
    student_optimizer = torch.optim.AdamW(student_model.parameters(), lr=3e-4)

    for epoch in range(args.epochs):
        student_model.train()
        for batch_idx, item in tqdm(enumerate(dataloader['train'])):
            texts, labels = item['text'], item['label']
            c_t_inputs, c_t_labels = tensoizer(teacher_tokenizer, texts, labels, args)
            c_s_inputs, c_s_labels = tensoizer(student_tokenizer, texts, labels, args)
            student_optimizer.zero_grad()
            with torch.no_grad():
                teacher_model_out = teacher_model(**c_t_inputs, output_hidden_states=True)
            student_model_out = student_model(**c_s_inputs, output_hidden_states=True)
        
            loss_logits = kldiv(student_model_out.logits, teacher_model_out.logits, T=args.T)
            loss_feature = ce_loss(student_model_out.logits, c_s_labels)
            logger.info(f"Logits Loss: {loss_logits}, Feature Loss: {loss_feature}")
            loss = args.alpha*loss_logits + (1-args.alpha)*loss_feature 
            loss.backward()
            student_optimizer.step()
        c_student_total_loss, c_student_total_acc, p_student_total_loss, p_student_total_acc = evaluate(teacher_model, teacher_tokenizer, student_model, student_tokenizer, dataloader['validation'], args)
        logger.info(f"Validation Clean Loss:{c_student_total_loss}, Valiation Poison Loss:{p_student_total_loss}, Valiation Clean Acc:{c_student_total_acc}, Validation Poison Acc:{p_student_total_acc}")
        if c_student_total_acc >= best_validation_acc:
            best_validation_acc = c_student_total_acc
            student_model.save_pretrained(args.model_student_save_path)
    student_model = AutoModelForSequenceClassification.from_pretrained(args.student_model_path, num_labels=args.num_labels).to(device)
    student_model.config.pad_token_id = student_model.config.eos_token_id
    student_model = PeftModel.from_pretrained(student_model, args.model_student_save_path)
    c_student_total_loss, c_student_total_acc, p_student_total_loss, p_student_total_acc = evaluate(teacher_model, teacher_tokenizer, student_model, student_tokenizer, dataloader['test'], args)
    logger.info(f"Test Clean Loss: {c_student_total_loss}, Test Poison Loss: {p_student_total_loss}, Test Clean Acc: {c_student_total_acc}, Test Poison Acc: {p_student_total_acc}")
        
def evaluate(teacher_model, teacher_tokenizer, student_model, student_tokenizer, dataloader, args):
    student_model.eval()
    teacher_model.eval()
    ce_loss = CrossEntropyLoss()
    c_student_total_loss = 0
    c_student_total_acc = 0
    p_student_total_loss = 0
    p_student_total_acc = 0
    c_batch = 0
    p_batch = 0
    triggers = readTheTargetingTrigger(args)
    for batch_idx, item in enumerate(dataloader):
        texts, labels = item['text'], item['label']
        p_texts, p_labels = poison(texts, labels, triggers['triggers'], args)
        c_t_inputs, c_t_labels = tensoizer(teacher_tokenizer, texts, labels, args)
        c_inputs, c_labels = tensoizer(student_tokenizer, texts, labels, args)
        with torch.no_grad():
            c_model_out = student_model(**c_inputs)
            t_model_out = teacher_model(**c_t_inputs)
        c_loss = ce_loss(c_model_out.logits, c_labels)
        loss_logits = kldiv(c_model_out.logits, t_model_out.logits, T=args.T)
        c_student_total_loss = (1-args.alpha)*c_loss.item() + args.alpha*loss_logits.item()
        c_student_total_acc += accuracy(c_model_out.logits, c_labels)
        c_batch += 1
        if len(p_labels) > 0:
            p_s_inputs, p_s_labels = tensoizer(student_tokenizer, p_texts, torch.LongTensor(p_labels), args)
            with torch.no_grad():
                p_model_out = student_model(**p_s_inputs)
            p_loss = ce_loss(p_model_out.logits, p_s_labels)
            p_student_total_loss += p_loss.item()
            p_student_total_acc += accuracy(p_model_out.logits, p_s_labels)
            p_batch += 1    
    c_student_total_loss = c_student_total_loss / c_batch
    c_student_total_acc = c_student_total_acc / c_batch
    p_student_total_acc = p_student_total_acc / p_batch
    p_student_total_loss = p_student_total_loss / p_batch
    return c_student_total_loss, c_student_total_acc, p_student_total_loss, p_student_total_acc


def main():
    args = parse_args()
    
    student_config = AutoConfig.from_pretrained(args.student_model_path)    
    student_config.num_labels = args.num_labels
    teacher_config = AutoConfig.from_pretrained(args.teacher_model_path)    
    teacher_config.num_labels = args.num_labels
    
    padding_side = args.padding_side
    
    student_peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
 
    student_model = AutoModelForSequenceClassification.from_pretrained(args.student_model_path, num_labels=args.num_labels)
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model_path, padding_side=padding_side)
    student_model.config.pad_token_id = student_model.config.eos_token_id
    teacher_model = AutoModelForSequenceClassification.from_pretrained(args.teacher_model_path, num_labels=args.num_labels, device_map='balanced_low_0')
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path, padding_side=padding_side)
    teacher_model.config.pad_token_id = teacher_model.config.eos_token_id
    
    if getattr(student_tokenizer, "pad_token_id") is None:
        student_tokenizer.pad_token_id = student_tokenizer.eos_token_id
        
    if getattr(teacher_tokenizer, "pad_token_id") is None:
        teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id
        
    
    student_model = get_peft_model(student_model, student_peft_config)
    teacher_model = PeftModel.from_pretrained(teacher_model, args.teacher_model_save_path)

    clean_train_data = load_dataset(args.dataset_path, "train")
    clean_validation_data = load_dataset(args.dataset_path, "validation")
    clean_test_data = load_dataset(args.dataset_path, "test")
    
    logger.info(f"clean train dataset: {len(clean_train_data)}, clean validation dataset: {len(clean_validation_data)}")
   
    dataset = {
        "train": clean_train_data,
        "validation": clean_validation_data,
        "test": clean_test_data
    }
    dataloader = wrap_dataset(dataset, batch_size=args.batch_size) 
    distillation(student_model, teacher_model, student_tokenizer, teacher_tokenizer, dataloader, args)
    
if __name__ == '__main__':
    main()