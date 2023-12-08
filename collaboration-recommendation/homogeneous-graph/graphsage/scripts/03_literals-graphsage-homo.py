from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
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



#summary text file with evaluation results
orig_stdout = sys.stdout
f = open('eval-soa-aa-homogeneous/graphsage/results/03_eval-graphsage-homo-literals.txt', 'w')
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

combined_array = np.concatenate([author_df_literals.values, author_df_nld.values], axis=1)

author_tensor = torch.tensor(combined_array, dtype=torch.float)


has_author_csv = "eval-soa-aa-homogeneous/homogeneous-graph/author_author.csv"
has_author_df = pd.read_csv(has_author_csv, header=None)
has_author_src = torch.tensor(has_author_df.iloc[:, 0].values, dtype=torch.long)
has_author_dst = torch.tensor(has_author_df.iloc[:, 1].values, dtype=torch.long)
edge_index = torch.stack([has_author_dst, has_author_src], dim=0)

data = Data(x=author_tensor, edge_index=edge_index)

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

# Simplify the GNN model
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GNN, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class Classifier(torch.nn.Module):
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        edge_feat_author_1= x[edge_index[0]]
        edge_feat_author_2 = x[edge_index[1]]
        return (edge_feat_author_1 * edge_feat_author_2).sum(dim=-1)
    


# Modify the forward pass for the model
class Model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Model, self).__init__()
        self.author_lin = torch.nn.Linear(132, hidden_channels)

        self.gnn = GNN(in_channels, hidden_channels)
        self.classifier = Classifier()



    def forward(self, data):

        x_out = self.author_lin(data.x)

        x = self.gnn(x_out, data.edge_index)

        pred = self.classifier(
         x, data.edge_label_index
        )


        return pred

# Assuming you've set the dimensions properly
model = Model(64, 64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



#Early Stopping
patience = 1
best_val_loss = float('inf')
epochs_without_improvement = 0

# For example, during training:
for epoch in range(1, 100):
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
        print(f"Stopping after {epoch} Epochs.")
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

