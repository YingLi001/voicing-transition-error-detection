'''
@Project   : SMAAT Project
@File      : lstm_model.py
@Author    : Ying Li
@Student ID: 20909226
@School    : EECMS
@University: Curtin University

Brief description:
This script implements an end-to-end PyTorch pipeline for binary classification of variable-length
speech feature sequences (e.g., GeMAPS). It supports loading per-file numpy feature arrays with
JSON labels, batching with padding, an LSTM-based classifier, focal loss for class-imbalance,
training/validation loops, evaluation metrics, LOSO (leave-one-subject-out) cross-validation,
and inference utilities that save per-sample predictions and confidence scores.

Expected inputs and file formats:
- Feature files: per-sample .npy files. Each file contains a 2D array (timesteps x feature_dim).
- Labels: JSON mapping filename -> {0, 1} (binary).
- Typical usage assumes directory paths and JSON are provided and filenames in the JSON match files.

Key behaviors and outputs:
- Trains an LSTM classifier that accepts variable-length sequences via pack_padded_sequence.
- Uses FocalLoss by default (configurable), but can be easily adapted to BCEWithLogitsLoss with pos_weight.
- Saves model weights and CSV reports for predictions and inference results (filenames are configurable in-code).
- Calculates standard metrics: accuracy, precision, recall, F1, confusion matrix, ROC AUC, PR AUC and
    produces plots (saved as PDFs).

Notes and recommendations:
- Ensure feature dimension (input_dim) passed to LSTMClassifier matches the number of columns in the .npy arrays.
- The collate_fn pads sequences to the longest sequence in the batch; pack_padded_sequence requires proper lengths.
- Adjust batch_size, learning rate, and class-balancing strategy for your dataset size and class imbalance.
- When using LOSO, filenames must encode the subject identifier in a consistent position for get_subject_from_filename.
- The script uses CPU/GPU automatically via torch.device selection.
- For reproducibility, set torch.manual_seed before DataLoader creation and training.
'''

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE    # does not work for data with different lengths
import numpy as np
from pathlib import Path
import get_VT_labels_from_TG
import itertools

# Dataset
class SpeechFeatureDataset(Dataset):
    def __init__(self, feature_dir, label_path, file_list):
        with open(label_path, "r") as f:
            self.labels = json.load(f)
        self.feature_dir = feature_dir
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        feature_path = os.path.join(self.feature_dir, fname)
        feature = np.load(feature_path)
        label = self.labels[fname]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Collate function
def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = [s.shape[0] for s in sequences]
    padded = pad_sequence(sequences, batch_first=True)
    return padded, torch.tensor(labels), torch.tensor(lengths)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        BCE = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE)
        focal = self.alpha * (1 - pt) ** self.gamma * BCE
        return focal.mean()


# LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        # super().__init__()
        # self.bn = nn.BatchNorm1d(input_dim)
        # self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=3, batch_first=True, bidirectional=True, dropout=0.3)
        # self.dropout = nn.Dropout(0.5)
        # self.classifier = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     # nn.Dropout(0.3),
        #     # nn.Linear(hidden_dim, 32),
        #     # nn.Linear(32, 16),
        #     # # # nn.ReLU(),
        #     # nn.Dropout(0.3),
        #     nn.Linear(hidden_dim, 1)
        # )
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        # out = self.classifier(h_n[-1])
        # return self.sigmoid(out).squeeze(1)
        return self.classifier(h_n[-1]).squeeze(1)

# Training & evaluation
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y, lengths in dataloader:
        x, y, lengths = x.to(device), y.float().to(device), lengths.to(device)
        optimizer.zero_grad()
        y_pred = model(x, lengths)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y, lengths in dataloader:
            x, y, lengths = x.to(device), y.float().to(device), lengths.to(device)
            y_pred = model(x, lengths)
            preds = (y_pred > 0.5).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.int().cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return acc
def train(model, train_loader, val_loader, optimizer, criterion, device, epochs=30):
        for epoch in range(1, epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_acc = evaluate(model, val_loader, device)
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Validation Accuracy = {val_acc:.4f}")
        torch.save(model.state_dict(), "lstm_classifier_12Nov.pth")
        print("Model saved to lstm_classifier_12Nov.pth")
        # Generate confusion matrix for train set
        model.eval()
        train_preds, train_labels = [], []
        with torch.no_grad():
            for x, y, lengths in train_loader:
                x, y, lengths = x.to(device), y.float().to(device), lengths.to(device)
                y_pred = model(x, lengths)
                preds = (y_pred > 0.5).int().cpu().numpy()
                train_preds.extend(preds)
                train_labels.extend(y.int().cpu().numpy())
        train_cm = confusion_matrix(train_labels, train_preds)
        print("Train Confusion Matrix:")
        print(train_cm)
        # plt.figure(figsize=(4, 4))
        # sns.heatmap(train_cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
        # plt.xlabel("Predicted")
        # plt.ylabel("Actual")
        # plt.title("Train Confusion Matrix")
        # plt.tight_layout()
        # plt.savefig("train_confusion_matrix.pdf")
        # plt.close()

        # Generate confusion matrix for validation set
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x, y, lengths in val_loader:
                x, y, lengths = x.to(device), y.float().to(device), lengths.to(device)
                y_pred = model(x, lengths)
                preds = (y_pred > 0.5).int().cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(y.int().cpu().numpy())
        val_cm = confusion_matrix(val_labels, val_preds)
        print("Validation Confusion Matrix:")
        print(val_cm)
        # plt.figure(figsize=(4, 4))
        # sns.heatmap(val_cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
        # plt.xlabel("Predicted")
        # plt.ylabel("Actual")
        # plt.title("Validation Confusion Matrix")
        # plt.tight_layout()
        # plt.savefig("val_confusion_matrix.pdf")
        # plt.close()

# Add probability/confidence score to inference results
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def inference(model, test_loader, test_files, device):
    model.eval()
    results = []
    with torch.no_grad():
        for batch_idx, (x, y, lengths) in enumerate(test_loader):
            x, y, lengths = x.to(device), y.float().to(device), lengths.to(device)
            output = model(x, lengths)
            # Convert logits to probabilities   
            probs = torch.sigmoid(output).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            for i in range(x.size(0)):
                idx = batch_idx * test_loader.batch_size + i
                if idx < len(test_files):
                    fname = test_files[idx]
                    feature = x[i].cpu().numpy()
                    gt = int(y[i].cpu().numpy())
                    pred = int(preds[i])
                    prob = float(probs[i])
                    if pred == 1:  # Only save positive predictions
                        conf = round(prob * 100, 4)  # Confidence score
                    else:
                        conf = round((1 - prob) * 100, 4)  # Confidence score for negative predictions
                    results.append({
                        "file": fname,
                        "prediction": pred,
                        "groundtruth": gt,
                        "confidence": conf,
                        "feature": feature.tolist()
                    })
    df = pd.DataFrame(results)
    df.to_csv("test_inference_12Nov.csv", index=False)
    print("Saved inference results to test_inference_results_12Nov.csv")


def combine_feature_label_from_folders(feature_dirs, label_files, out_feature_dir, out_label_file):
    """
    Combine features and labels from multiple folders, save all features in a new folder,
    and save the combined labels in a new json file.

    Args:
        feature_dirs (list): List of feature directory paths.
        label_files (list): List of label json file paths.
        out_feature_dir (str): Output directory to save all features.
        out_label_file (str): Output json file to save combined labels.

    Returns:
        combined_files (list): List of all new file names.
        combined_label_dict (dict): Mapping from new file name to label.
        folder_map (dict): Mapping from new file name to its original (feature_dir, fname).
    """
    os.makedirs(out_feature_dir, exist_ok=True)
    combined_files = []
    combined_label_dict = {}
    folder_map = {}
    for idx, (feature_dir, label_file) in enumerate(zip(feature_dirs, label_files)):
        with open(label_file, "r") as f:
            label_dict = json.load(f)
        for fname, label in label_dict.items():
            # Prefix with folder index to avoid name collision
            new_fname = f"{idx}_{fname}"
            src_path = os.path.join(feature_dir, fname)
            dst_path = os.path.join(out_feature_dir, new_fname)
            if os.path.exists(src_path):
                np.save(dst_path, np.load(src_path))
                combined_files.append(new_fname)
                combined_label_dict[new_fname] = label
                folder_map[new_fname] = (feature_dir, fname)
            else:
                print(f"Warning: {src_path} does not exist, skipping.")
    # Save combined labels
    with open(out_label_file, "w") as f:
        json.dump(combined_label_dict, f, indent=2)
    print(f"Saved combined features to {out_feature_dir} and labels to {out_label_file}")
    return out_feature_dir, out_label_file, combined_label_dict

def hyperparameter_tuning(train_loader, val_loader, device, input_dim=6, search_space=None):
    """
    Simple grid search for hyperparameter tuning.
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: torch.device
        input_dim: feature dimension
        search_space: dict with keys 'hidden_dim', 'lr', 'gamma', 'alpha', 'epochs'
        epochs: (deprecated) not used, use search_space['epochs'] instead
    Returns:
        best_params: dict of best hyperparameters
        best_score: best validation accuracy
    """
    if search_space is None:
        search_space = {
            "hidden_dim": [32, 64, 128, 256],
            "lr": [1e-3, 1e-4, 1e-5],
            "gamma": [1, 2],
            "alpha": [0.25, 0.5],
            "epochs": [50, 100, 150] 
        }
    best_score = 0
    best_params = None
    for hidden_dim, lr, gamma, alpha, epochs in itertools.product(
        search_space["hidden_dim"], search_space["lr"], search_space["gamma"], search_space["alpha"], search_space["epochs"]
    ):
        model = LSTMClassifier(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = FocalLoss(gamma=gamma, alpha=alpha).to(device)
        for _ in range(epochs):
            train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"hidden_dim={hidden_dim}, lr={lr}, gamma={gamma}, alpha={alpha}, epochs={epochs} => val_acc={val_acc:.4f}")
        if val_acc > best_score:
            best_score = val_acc
            best_params = {
                "hidden_dim": hidden_dim,
                "lr": lr,
                "gamma": gamma,
                "alpha": alpha,
                "epochs": epochs
            }
    print(f"Best params: {best_params}, Best val_acc: {best_score:.4f}")
    return best_params, best_score
def get_subject_from_filename(filename):
    # Assumes subject is the second part, e.g., "451_Ruby_A062_01210908_C021_pam.npy"
    # Returns "Ruby" for "451_Ruby_A062_01210908_C021_pam.npy"
    parts = filename.split("_")
    if len(parts) > 1:
        return parts[1]
    return "unknown"

def loso_split(file_list):
    """
    Returns a dict: subject -> (train_files, test_files)
    """
    subject_to_files = {}
    for fname in file_list:
        subj = get_subject_from_filename(fname)
        subject_to_files.setdefault(subj, []).append(fname)
    splits = {}
    subjects = list(subject_to_files.keys())
    for test_subj in subjects:
        # Exclude augmented files from test set
        test_files = [f for f in subject_to_files[test_subj] if "_aug" not in f]
        # test_files = subject_to_files[test_subj] 
        train_files = [f for s, files in subject_to_files.items() if s != test_subj for f in files]
        splits[test_subj] = (train_files, test_files)
    return splits

def run_loso_cv(GeMAPs_features_dir, labels_file, label_dict, input_dim=6, epochs=20):
    all_files = list(label_dict.keys())
    
    splits = loso_split(all_files)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    # aug_train_files, aug_val_files = train_test_split(
    #     augmented_files, test_size=0.2, stratify=[label_dict[f] for f in augmented_files], random_state=42
    # )
    # train_files = train_files + aug_train_files
    # val_files = val_files + aug_val_files
    for subj, (train_files, test_files) in splits.items():
        # Optionally split a validation set from train_files
        train_files, val_files = train_test_split(
            train_files, test_size=0.2, stratify=[label_dict[f] for f in train_files], random_state=42
        )
        # Count class balance for pos_weight
        counts = Counter(int(label_dict[f]) for f in train_files)
        neg = counts.get(0, 1)
        pos = counts.get(1, 1)
        pos_weight = torch.tensor([neg/pos], dtype=torch.float32).to(device)
        train_dataset = SpeechFeatureDataset(GeMAPs_features_dir, labels_file, train_files)
        val_dataset = SpeechFeatureDataset(GeMAPs_features_dir, labels_file, val_files)
        test_dataset = SpeechFeatureDataset(GeMAPs_features_dir, labels_file, test_files)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
        model = LSTMClassifier(input_dim=input_dim, hidden_dim=128).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = FocalLoss(gamma=2, alpha=0.25).to(device)
        train(model, train_loader, val_loader, optimizer, criterion, device, epochs=epochs)
        # Evaluate
        test_acc = evaluate(model, test_loader, device)
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y, lengths in test_loader:
                x, y, lengths = x.to(device), y.float().to(device), lengths.to(device)
                y_pred = model(x, lengths)
                preds = (y_pred > 0.5).int().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.int().cpu().numpy())
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        results.append({
            "subject": subj,
            "test_acc": test_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_test": len(test_files)
        })
        print(f"LOSO Subject={subj}: Test Acc={test_acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, N={len(test_files)}")
    df = pd.DataFrame(results)
    # df.to_csv("loso_results_combined.csv", index=False)
    # print("Saved LOSO results to loso_results_combined.csv")
    print(df.describe())
if __name__ == "__main__":
    # Load and split data
    GeMAPs_features_dir = "/mnt/data/ying/SMAAT_1st_iterative_learning/TD/Band_3/geMAPs_features_band3"
    labels_file = "/mnt/data/ying/SMAAT_1st_iterative_learning/TD/Band_3/labels_band3.json"
    label_dict = {}
    with open(labels_file, "r") as f:
        label_dict = json.load(f)
        print(f"Loaded {len(label_dict)} labels from {labels_file}")

    label_dict = {fname: label for fname, label in label_dict.items()}
    all_files = list(label_dict.keys())
    # Count labels
    counts = Counter(int(label_dict[f]) for f in all_files)
    print("counts:", counts)
    neg = counts[0]

    print(f"Negative samples: {neg}")
    pos = counts[1]
    print(f"Positive samples: {pos}")
    
    # Filter out augmented files (containing "_aug") for splitting
    augmented_files = [f for f in all_files if "_aug" in f]
    print(f"Augmented files: {len(augmented_files)}")
    non_augmented_files = [f for f in all_files if "_aug" not in f]
    print(f"Non-augmented files: {len(non_augmented_files)}")
    for f in non_augmented_files:
        if 1 in label_dict.items():
            print(f"Warning: {f} not found in label_dict, skipping.")
            continue

    # Split only non-augmented files into train/test/val
    train_files, test_files = train_test_split(
        non_augmented_files, test_size=0.2, stratify=[label_dict[f] for f in non_augmented_files], random_state=42
    )
    train_files, val_files = train_test_split(
        train_files, test_size=0.2, stratify=[label_dict[f] for f in train_files], random_state=42
    )
    # Split augmented files into train/val using the same ratio as non-augmented files
    aug_train_files, aug_val_files = train_test_split(
        augmented_files, test_size=0.2, stratify=[label_dict[f] for f in augmented_files], random_state=42
    )
    train_files = train_files + aug_train_files
    # Split all_files into train, val, test (80/16/20 split, val from train)
    
    # val_files = val_files + aug_val_files
    # train_files = train_files + augmented_files
    print(f"Number of training files: {len(train_files)}")
    # print("Training files:", train_files)
    print(f"Number of validation files: {len(val_files)}")
    # print("Validation files:", val_files)
    print(f"Number of test files: {len(test_files)}")
    
    
    # train_files, val_files = train_test_split(train_files, test_size=0.2, stratify=[label_dict[f] for f in train_files], random_state=42)
    torch.manual_seed(47)
    train_dataset = SpeechFeatureDataset(GeMAPs_features_dir, labels_file, train_files)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    val_dataset = SpeechFeatureDataset(GeMAPs_features_dir, labels_file, val_files)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    test_dataset = SpeechFeatureDataset(GeMAPs_features_dir, labels_file, test_files)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    # Run training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Weight = # negative / # positive
    # pos_weight = torch.tensor([neg/pos], dtype=torch.float32).to(device)

    model = LSTMClassifier(input_dim=6, hidden_dim=128).to(device)  # adjust input_dim as needed
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)  # Use weighted BCE loss
    criterion = FocalLoss(gamma=2, alpha=0.25).to(device)  # Use Focal Loss
    print(model)
    
    # # Hyperparameter tuning (optional)
    # best_params, best_score = hyperparameter_tuning(train_loader, val_loader, device, input_dim=6)
    # print(f"Best params: {best_params}, Best validation accuracy: {best_score:.4f}")
    
    
    # Train the model
    print("Starting training...")
    train(model, train_loader, val_loader, optimizer, criterion, device, epochs=40)
    
    model.load_state_dict(torch.load("lstm_classifier.pth", weights_only=True))
    model.eval()    
    # Evaluate on test set
    test_acc = evaluate(model, test_loader, device)
    print(f"Final Test Accuracy = {test_acc:.4f}")
    # Get predictions and ground truths for the test set
    # model.eval()
    all_preds, all_labels, all_conf = [], [], []

    with torch.no_grad():
        for x, y, lengths in test_loader:
            x, y, lengths = x.to(device), y.float().to(device), lengths.to(device)
            y_pred = model(x, lengths)
            probs = torch.sigmoid(y_pred).cpu().numpy()
            
            for prob, label in zip(probs, y.cpu().numpy()):
                pred = int(prob > 0.5)
                conf = round(prob * 100, 4) if pred == 1 else round((1 - prob) * 100, 4)
                
                all_preds.append(pred)
                all_labels.append(int(label))
                all_conf.append(conf)

    # Now `all_conf` matches sample count
    df_results = pd.DataFrame({
        "file": test_files,
        "prediction": all_preds,
        "groundtruth": all_labels,
        "confidence": all_conf
    })
    df_results.to_csv("test_predictions.csv", index=False)
    print("Saved test predictions and labels to test_predictions.csv")
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    cm = confusion_matrix(all_labels, all_preds)
    print("Test Confusion Matrix:")
    print(cm)

    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.pdf")
    plt.close()
    
    # Print a few predictions and labels for comparison
    print("Sample labels:", all_labels[:10])
    print("Sample predictions:", all_preds[:10])
    # Get predicted probabilities for AUC
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y, lengths in test_loader:
            x, y, lengths = x.to(device), y.float().to(device), lengths.to(device)
            probs = model(x, lengths).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y.cpu().numpy())
    print(f"Test Precision = {precision:.4f}, Recall = {recall:.4f}, F1-score = {f1:.4f}")
    
    
    auc = roc_auc_score(all_labels, all_probs)
    print(f"Test AUC = {auc:.4f}")

    # Optionally, plot ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curve.pdf")
    plt.close()

    # Compute PR-AUC (Precision-Recall AUC)

    pr_auc = average_precision_score(all_labels, all_probs)
    print(f"Test PR-AUC = {pr_auc:.4f}")

    # Optionally, plot Precision-Recall curve
    precision_vals, recall_vals, thresholds = precision_recall_curve(all_labels, all_probs)
    plt.figure()
    plt.plot(recall_vals, precision_vals, label=f"PR-AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig("pr_curve.pdf")
    plt.close()
    
    # Inference multiple unseen samples in other bands and print the result
    GeMAPs_features_3sets_dir = "/mnt/data/ying/SMAAT_1st_iterative_learning/TD/Band_1/geMAPs_features_band1"
    full_label_dict_path = "/mnt/data/ying/SMAAT_1st_iterative_learning/TD/Band_1/labels_band1.json"
   
    # Select files with label 1
    with open(full_label_dict_path, "r") as f:
        all_labels = json.load(f)
    
    # Print the number of files with label 1
    positive_files = [fname for fname, label in all_labels.items()]
    # print(f"Files with label 1: {positive_files}")
    # Inference on all positive_files and print/save results
    results = []
    model.eval()
    for fname in positive_files:
        feature_path = os.path.join(GeMAPs_features_3sets_dir, fname)
        if not os.path.exists(feature_path):
            print(f"Feature file not found: {feature_path}")
            continue
        feature = np.load(feature_path)
        label = all_labels[fname]
        tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
        length = torch.tensor([feature.shape[0]]).to(device)
        with torch.no_grad():
            output = model(tensor, length)
            prob = torch.sigmoid(output).item()
            pred = int(prob > 0.5)
            conf = round(prob * 100, 4) if pred == 1 else round((1 - prob) * 100, 4)
            results.append({
                "file": fname,
                "prediction": pred,
                "groundtruth": label,
                "confidence": conf,
                "probability": prob
            })
            # print(f"File: {fname}, GT: {label}, Pred: {pred}, Prob: {prob:.4f}, Conf: {conf}")

    # Compute metrics for positive_files inference
    y_true = [r["groundtruth"] for r in results]
    y_pred = [r["prediction"] for r in results]
    y_prob = [r["probability"] for r in results]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"Generalisation Metrics")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    # Optionally save to CSV
    df_pos = pd.DataFrame(results)
    df_pos.to_csv("test_inference_results_band1.csv", index=False)
    print("Saved inference results for positive files to test_inference_results_band1.csv")
    
    
    # # Inference on one sample and print the result
    # with open(full_label_dict_path, "r") as f:
    #     full_label_dict = json.load(f)
    # # Check if the sample file exists in the label dictionary
    # sample_feature_path = os.path.join(GeMAPs_features_3sets_dir, sample_file)
    # print(f"Sample feature path: {sample_feature_path}")
    # sample_feature = np.load(sample_feature_path)
    # sample_label = full_label_dict[sample_file]

    # sample_tensor = torch.tensor(sample_feature, dtype=torch.float32).unsqueeze(0).to(device)
    # sample_length = torch.tensor([sample_feature.shape[0]]).to(device)

    # model.eval()
    # # Load the trained model weights before inference
    # # model.load_state_dict(torch.load("lstm_classifier_positive_1_VT_errors_train_3.pth", map_location=device, weights_only=True))
    # with torch.no_grad():
    #     output = model(sample_tensor, sample_length)
    #     prob = torch.sigmoid(output).item()
    #     pred = int(prob > 0.5)
    #     # prob = float(prob)
    #     # if pred == 1:  # Only save positive predictions
    #     #     conf = round(prob * 100, 4)  # Confidence score
    #     # else:
    #     #     conf = round((1 - prob) * 100, 4)  # Confidence score for negative predictions
    #     print(f"File: {sample_file}")
    #     print(f"Ground Truth: {sample_label}")
    #     print(f"Predicted: {pred}")
    #     print(f"Probability: {prob:.4f}")