from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.profile import count_parameters
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
import torch_geometric.transforms as T
from sklearn.metrics import average_precision_score, accuracy_score, f1_score, roc_auc_score, recall_score, precision_score 
import torch.nn.functional as F
from torch import Tensor
import torch
import pandas as pd
import numpy as np
import sys
import tqdm
from torch_geometric.data import DataLoader

import torch.nn as nn



#summary text file with evaluation results
orig_stdout = sys.stdout
f = open('eval-soa-aa-homogeneous/gat/results/10_eval-gat-homo-combined-nn.txt', 'w')
sys.stdout = f

seed_everything(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)

#load data
author_csv_nld = "soa-sw-homogeneous-author/idx/nodes-nld/author_nld.csv"
author_df_nld = pd.read_csv(author_csv_nld, header=None)
author_df_nld = author_df_nld.astype(float)

author_csv_literals = "soa-sw-homogeneous-author/idx/nodes-literals-without-nld/author_literals_without_nld.csv"
author_df_literals = pd.read_csv(author_csv_literals, header=None)
author_df_literals = author_df_literals.astype(float)


# Concatenate the two DataFrames horizontally
author_df_nld_literals = pd.concat([ author_df_literals, author_df_nld], axis=1)



author_csv_transe = "soa-sw-homogeneous-author/idx/nodes-transe/author_transe.csv"
author_df_transe = pd.read_csv(author_csv_transe, header=None)
author_df_transe = author_df_transe.astype(float)


### vektor addition ####

# Linear Layers Literals
author_lin_nld_literal = torch.nn.Linear(132, 64).eval()

# lin layer literals
author_tensor_nld_literal = author_lin_nld_literal(torch.tensor(author_df_nld_literals.values, dtype=torch.float))


# Linear Layers Embeddings
author_lin_embedding = torch.nn.Linear(128, 64).eval()

# lin layer transe embeddings
author_embeddings_tensor = author_lin_embedding(torch.tensor(author_df_transe.values, dtype=torch.float))


# neural combinator


dim = 64 

class NeuralCombinator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralCombinator, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)  # Multiplizieren mit 2, da wir zwei Vektoren eingeben
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, v1, v2):
        # Kombiniere die beiden Vektoren
        combined = torch.cat((v1, v2), dim=-1)  
        x = F.relu(self.fc1(combined))
        x = self.fc2(x)
        return x

combinator = NeuralCombinator(input_dim=dim, hidden_dim=1, output_dim=dim)

author_tensor = combinator(author_tensor_nld_literal, author_embeddings_tensor)

### done neural combinator ####


has_author_csv = "eval-soa-aa-homogeneous/homogeneous-graph/author_author.csv"
has_author_df = pd.read_csv(has_author_csv, header=None)
has_author_src = torch.tensor(has_author_df.iloc[:, 0].values, dtype=torch.long)
has_author_dst = torch.tensor(has_author_df.iloc[:, 1].values, dtype=torch.long)
edge_index = torch.stack([has_author_dst, has_author_src], dim=0)

has_coauthor_lin = torch.nn.Linear(1, 1) 
has_coauthor_features_csv = "eval-soa-aa-homogeneous/homogeneous-graph/feature_author_author.csv"
has_coauthor_features_df = pd.read_csv(has_coauthor_features_csv, header=None)
has_coauthor_features_tensor = torch.tensor(has_coauthor_features_df.values, dtype=torch.float)
has_coauthor_features_tensor= has_coauthor_lin(has_coauthor_features_tensor)


data = Data(x=author_tensor, edge_index=edge_index, edge_attr=has_coauthor_features_tensor)


data = T.ToUndirected()(data)

transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    is_undirected = True
)

train_data, val_data, test_data = transform(data)


# Calculate validation loss
def validate(model):
    model.eval()
    total_loss = total_examples = 0
    with torch.no_grad():
        val_data.to(device)
        pred = model(val_data)
        ground_truth = val_data.edge_label
        loss = F.binary_cross_entropy_with_logits(pred, ground_truth.float())
        total_loss += loss.item() * pred.numel()
        total_examples += pred.numel()
    model.train()
    return total_loss / total_examples

def evaluate(model):
    model.eval()
    preds, ground_truths = [], []
    with torch.no_grad():
        test_data.to(device)
        preds.append(model(test_data))
        ground_truths.append(test_data.edge_label)


    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()


    auc = roc_auc_score(ground_truth, pred)
    ap = average_precision_score(ground_truth, pred)
    re = recall_score(ground_truth, pred > 0)
    pre = precision_score(ground_truth, pred > 0)
    acc = accuracy_score(ground_truth, pred > 0)
    f1 = f1_score(ground_truth, pred > 0)
    
    return auc, ap, re, pre, acc, f1


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_edge_features=1):
        super().__init__()
        self.conv1 = GATConv(hidden_channels, hidden_channels, add_self_loops=False, edge_dim=num_edge_features)
        self.conv2 = GATConv(hidden_channels, hidden_channels, add_self_loops=False, edge_dim=num_edge_features)
    
    def forward(self, x, edge_index, edge_attr) -> Tensor:
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = self.conv2(x, edge_index, edge_attr)
        return x



class Classifier(torch.nn.Module):
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        edge_feat_author_1= x[edge_index[0]]
        edge_feat_author_2 = x[edge_index[1]]
        return (edge_feat_author_1 * edge_feat_author_2).sum(dim=-1)
    


# Modify the forward pass for the model
class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(Model, self).__init__()
        self.author_lin = torch.nn.Linear(64, hidden_channels)

        self.gnn = GNN(hidden_channels)
        self.classifier = Classifier()



    def forward(self, data):

        x_out = self.author_lin(data.x)

        x = self.gnn(x_out, data.edge_index, data.edge_attr)

        pred = self.classifier(
         x, data.edge_label_index
        )


        return pred


model = Model(hidden_channels=64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



#Early Stopping
patience = 1
best_val_loss = float('inf')
epochs_without_improvement = 0

# For example, during training:
for epoch in range(1, 101):
    total_loss = total_examples = 0
    train_data.to(device)
    optimizer.zero_grad()
    pred = model(train_data)
    loss = F.binary_cross_entropy_with_logits(pred, train_data.edge_label)
    loss.backward(retain_graph=True)
    optimizer.step()
    total_loss += loss.item() * pred.size(0)
    total_examples += pred.size(0)

    print(f"Epoch: {epoch:03d}, Training Loss: {total_loss / total_examples:.4f}")
        
    #calcuate val loss
    val_loss = validate(model)
    print(f"Epoch: {epoch:03d}, Validation Loss: {val_loss:.4f}")

    #check early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement == patience:
        print(f"Early stopping after {epoch} Epochs.")
        print("Evaluation on Test Data:")
        auc, ap, re, pre, acc, f1_score = evaluate(model)
        print(f"Test AUC: {auc:.4f}")
        print(f"Test AP: {ap:.4f}")
        print(f"Test Recall: {re:.4f}")
        print(f"Test Precision: {pre:.4f}")
        print(f"Test Accuracy: {acc:.4f}")
        print(f"Test F1: {f1_score:.4f}")
        break

    if epoch == 100:
        print(f"Early stopping after {epoch} Epochs.")
        print("Evaluation on Test Data:")
        auc, ap, re, pre, acc, f1_score = evaluate(model)
        print(f"Test AUC: {auc:.4f}")
        print(f"Test AP: {ap:.4f}")
        print(f"Test Recall: {re:.4f}")
        print(f"Test Precision: {pre:.4f}")
        print(f"Test Accuracy: {acc:.4f}")
        print(f"Test F1: {f1_score:.4f}")
        break

model_parameters = count_parameters(model)
print(f'The model has {model_parameters:,} trainable parameters.')

sys.stdout = orig_stdout
f.close()
