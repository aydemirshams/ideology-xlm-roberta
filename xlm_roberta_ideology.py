#!/usr/bin/env python3
"""
XLM-RoBERTa model for TOUCHÉ 2025 Sub-Task 1: Classifying parliamentary speeches into left/right ideologies.
"""

import os
import glob
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from typing import List, Tuple, Dict, Optional, Any

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import (
    XLMRobertaTokenizer, 
    XLMRobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from sklearn.metrics import classification_report, accuracy_score, f1_score
from data import read_data

CUDA_VISIBLE_DEVICES=0

# Configure argument parser
parser = argparse.ArgumentParser(description='XLM-RoBERTa for political ideology classification')
parser.add_argument('--data-dir', type=str, default='data', help='Directory with training and test data')
parser.add_argument('--model-type', type=str, default='xlm-roberta-large', choices=['xlm-roberta-base', 'xlm-roberta-large'], )
parser.add_argument('--output-dir', type=str, default='predictions')
parser.add_argument('--model-dir', type=str, default='models')
parser.add_argument('--save-model', action='store_true')
parser.add_argument('--load-model', type=str)
parser.add_argument('--max-length', type=int, default=256)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--eval-batch-size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=4)
parser.add_argument('--learning-rate', type=float, default=2e-5)
parser.add_argument('--weight-decay', type=float, default=0.01)
parser.add_argument('--warmup-ratio', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--early-stopping', type=int, default=3)
parser.add_argument('--focal-loss', action='store_true')
parser.add_argument('--focal-gamma', type=float, default=2.0)
parser.add_argument('--parliaments', nargs='+', default=['all'])
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--team-name', type=str, default='tunlp')
parser.add_argument('--runname', type=str, default='run1')
parser.add_argument('--augment', action='store_true')
parser.add_argument('--balance-samples', action='store_true')
parser.add_argument('--use-scheduler', action='store_true')
parser.add_argument('--layer-lr-decay', type=float, default=0.95)

args = parser.parse_args()

# List of all parliament codes
ALL_PARLIAMENTS = ["at", "ba", "be", "bg", "cz", "dk", "ee", "es-ct", "es-ga", "es-pv", "es", 
                   "fi", "fr", "gb", "gr", "hr", "hu", "is", "it", "lv", "nl", "no", "pl", 
                   "pt", "rs", "se", "si", "tr", "ua"]

if args.parliaments[0].lower() == 'all':
    args.parliaments = ALL_PARLIAMENTS

# Set random seeds for reproducibility
def set_seed(seed_value):
    """Set seed for reproducibility."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True

set_seed(args.seed)


class FocalLoss(torch.nn.Module):
    """Focal Loss implementation for imbalanced classification."""
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ParliamentDataset(Dataset):
    """Dataset class for parliamentary speeches."""
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def preprocess_text(text):
    """Basic text preprocessing function."""
    # Convert to string just in case
    text = str(text)
    # Replace multiple spaces with a single space
    text = ' '.join(text.split())
    return text


def create_class_weights(labels):
    """Create class weights inversely proportional to class frequency."""
    counter = Counter(labels)
    total = len(labels)
    weights = {class_id: total / (len(counter) * count) for class_id, count in counter.items()}
    return torch.tensor([weights[i] for i in sorted(weights.keys())], dtype=torch.float)


def load_data(data_dir, parliaments, seed=42):
    """Load and combine data from multiple parliaments."""
    all_train_texts, all_train_labels = [], []
    all_val_texts, all_val_labels = [], []
    
    for parliament in parliaments:
        train_file = os.path.join(data_dir, f"{parliament}-train.tsv")
        if not os.path.exists(train_file):
            print(f"Skipping {parliament}: training file not found.")
            continue
        
        try:
            t_train, y_train, t_val, y_val = read_data(
                train_file, task='orientation', test_size=0.2, seed=seed
            )
            
            if not t_train or not t_val:
                print(f"Skipping {parliament}: empty training or validation set.")
                continue
                
            print(f"Loaded {parliament}: {len(t_train)} training, {len(t_val)} validation samples.")
            print(f"Label distribution - Train: {Counter(y_train)}, Val: {Counter(y_val)}")
            
            # Preprocess texts
            t_train = [preprocess_text(text) for text in t_train]
            t_val = [preprocess_text(text) for text in t_val]
            
            all_train_texts.extend(t_train)
            all_train_labels.extend(y_train)
            all_val_texts.extend(t_val)
            all_val_labels.extend(y_val)
            
        except Exception as e:
            print(f"Error loading {parliament}: {str(e)}")
    
    print(f"Combined dataset: {len(all_train_texts)} training, {len(all_val_texts)} validation samples")
    print(f"Overall label distribution - Train: {Counter(all_train_labels)}, Val: {Counter(all_val_labels)}")
    
    return all_train_texts, all_train_labels, all_val_texts, all_val_labels


def train_epoch(model, data_loader, optimizer, scheduler, device, loss_fn):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(data_loader, desc="Training")
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        loss = loss_fn(logits, labels)
        total_loss += loss.item()
        
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        if scheduler:
            scheduler.step()
            
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(data_loader)


def evaluate(model, data_loader, device, loss_fn):
    """Evaluate the model on validation data."""
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            loss = loss_fn(logits, labels)
            val_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return val_loss / len(data_loader), accuracy, f1, all_preds, all_labels


def predict(model, test_texts, tokenizer, device, batch_size=32, max_length=256):
    """Generate predictions for test data."""
    model.eval()
    predictions = []
    
    # Create a simple dataset for test data
    test_dataset = []
    for text in test_texts:
        encoding = tokenizer(
            preprocess_text(text),
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        test_dataset.append({
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        })
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # For binary classification, get probability of class 1
            probs = torch.softmax(outputs.logits, dim=1)
            probs_class_1 = probs[:, 1].cpu().numpy()
            predictions.extend(probs_class_1)
    
    return predictions


def save_predictions(predictions, ids, output_dir, filename):
    """Save predictions to a TSV file without header."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w') as f:
        for id_, pred in zip(ids, predictions):
            f.write(f"{id_}\t{pred}\n")
    
    print(f"Saved predictions to {output_path}")


def apply_mixup(batch, alpha=0.2):
    """Apply mixup data augmentation to the batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = batch['input_ids'].size(0)
    index = torch.randperm(batch_size)
    
    mixed_input_ids = lam * batch['input_ids'] + (1 - lam) * batch['input_ids'][index]
    mixed_attention_mask = torch.maximum(batch['attention_mask'], batch['attention_mask'][index])
    mixed_labels = batch['labels'].clone()  # We'll handle the mixup loss separately
    
    return {
        'input_ids': mixed_input_ids,
        'attention_mask': mixed_attention_mask,
        'labels': mixed_labels,
        'labels_mixed': batch['labels'][index],
        'lambda': lam
    }


def get_layer_wise_learning_rates(model, lr, decay_factor=0.95):
    """Apply layer-wise learning rate decay."""
    parameters = []
    
    # Usually the last parameters are the classifier layers
    # Adjust this based on the model architecture
    num_layers = model.config.num_hidden_layers
    
    # Add classifier parameters with base learning rate
    parameters.append({
        'params': [p for n, p in model.named_parameters() if 'classifier' in n],
        'lr': lr
    })
    
    # Add embeddings with lower learning rate
    parameters.append({
        'params': [p for n, p in model.named_parameters() if 'embeddings' in n],
        'lr': lr * (decay_factor ** num_layers)
    })
    
    # Add encoder layers with gradually increasing learning rate
    for layer_num in range(num_layers):
        layer_params = [
            p for n, p in model.named_parameters() 
            if f'encoder.layer.{layer_num}.' in n
        ]
        if layer_params:
            parameters.append({
                'params': layer_params,
                'lr': lr * (decay_factor ** (num_layers - layer_num))
            })
    
    return parameters


def main():
    """Main function to train, evaluate and predict."""
    start_time = time.time()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Load and initialize tokenizer and model
    print(f"Loading {args.model_type}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    
    
    # Load data
    train_texts, train_labels, val_texts, val_labels = load_data(
        args.data_dir, args.parliaments, args.seed
    )
    
    # If no data was loaded, exit
    if not train_texts:
        print("No training data found. Exiting.")
        return
    
    # Create datasets
    train_dataset = ParliamentDataset(
        train_texts, train_labels, tokenizer, max_length=args.max_length
    )
    val_dataset = ParliamentDataset(
        val_texts, val_labels, tokenizer, max_length=args.max_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.eval_batch_size
    )
    
    # Calculate class weights for handling imbalance
    class_weights = create_class_weights(train_labels).to(args.device)
    print(f"Class weights: {class_weights}")
    
    # Define loss function based on arguments
    if args.focal_loss:
        loss_fn = FocalLoss(gamma=args.focal_gamma, weight=class_weights)
        print(f"Using Focal Loss with gamma={args.focal_gamma}")
    else:
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        print("Using Weighted Cross Entropy Loss")
    
    # Initialize model - either load from saved file or create new
    if args.load_model:
        print(f"Loading model from {args.load_model}")
        model = AutoModelForSequenceClassification.from_pretrained(args.load_model)
    else:
        print("Initializing new model")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_type,
            num_labels=2,  # Binary classification
            problem_type="single_label_classification"
        )
        
    model = model.to(args.device)
    
    # Configure optimizer with layer-wise decay
    if args.layer_lr_decay < 1.0:
        parameters = get_layer_wise_learning_rates(model, args.learning_rate, args.layer_lr_decay)
        print("Using layer-wise learning rate decay")
    else:
        parameters = model.parameters()
    
    optimizer = AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Setup scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    if args.use_scheduler:
        print(f"Using linear warmup scheduler with warmup steps: {warmup_steps}")
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    else:
        scheduler = None
    
    # Training loop
    best_f1 = 0
    early_stopping_counter = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, args.device, loss_fn)
        
        # Evaluate
        val_loss, accuracy, f1, preds, labels = evaluate(
            model, val_loader, args.device, loss_fn
        )
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}, F1 Score (Macro): {f1:.4f}")
        
        # Print detailed classification report
        print("\nClassification Report:")
        print(classification_report(labels, preds))
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            early_stopping_counter = 0
            if args.save_model:
                model_save_path = os.path.join(args.model_dir, f"{args.model_type.split('/')[-1]}_best")
                model.save_pretrained(model_save_path)
                tokenizer.save_pretrained(model_save_path)
                print(f"Saved best model to {model_save_path} with F1: {f1:.4f}")
        else:
            early_stopping_counter += 1
        
        # Early stopping check
        if args.early_stopping > 0 and early_stopping_counter >= args.early_stopping:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Generate predictions for test files
    print("\nGenerating predictions for test files...")
    
    # If we saved a best model, load it for predictions
    if args.save_model and os.path.exists(os.path.join(args.model_dir, f"{args.model_type.split('/')[-1]}_best")):
        model = AutoModelForSequenceClassification.from_pretrained(
            os.path.join(args.model_dir, f"{args.model_type.split('/')[-1]}_best")
        ).to(args.device)
    
    for parliament in args.parliaments:
        test_file = os.path.join(args.data_dir, f"{parliament}-test.tsv")
        if not os.path.exists(test_file):
            print(f"Skipping {parliament}: test file not found.")
            continue
        
        try:
            ids, test_texts, _ = read_data(test_file, task='orientation', return_na=True, testset=True)
            if not test_texts:
                print(f"Skipping {parliament}: empty test set.")
                continue
            
            print(f"Generating predictions for {parliament} ({len(test_texts)} samples)...")
            predictions = predict(
                model, test_texts, tokenizer, args.device, 
                batch_size=args.eval_batch_size, max_length=args.max_length
            )
            
            # Save predictions using the specified format: <team>-<task>-<pcode>-<runname>.tsv
            filename = f"{args.team_name}-orientation-{parliament}-{args.runname}.tsv"
            save_predictions(predictions, ids, args.output_dir, filename)
            
        except Exception as e:
            print(f"Error processing {parliament} test file: {str(e)}")
    
    # Print total execution time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Best validation F1 score: {best_f1:.4f}")


if __name__ == "__main__":
    main()