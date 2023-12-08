## Code and results for Chapter 7 *Semantic in GNN-based Recommendation*

* [Paper Recommendation](./paper-recommendation) using SOA-SW.

* [Collaboration Recommendation](./collaboration-recommendation) using SOA-SW.

* [Task Recommendation](./task-recommendation) using LPWC.


The GNN-based recommendation scripts for the homogeneous, bipartite, and heterogeneous graph settings have the following structure (depending on whether GraphSAGE, GAT or HGT is used as GNN architecture):
* 01_one-hot-encoding-{graphsage/gat/hgt}.py
* 02_nld-{graphsage/gat/hgt}.py
* 03_literals-{graphsage/gat/hgt}.py
* 04_transe-{graphsage/gat/hgt}.py
* 05_nld-transe-{graphsage/gat/hgt}.py
* 06_combined-concatenated-{graphsage/gat/hgt}.py
* 07_combined-addition-{graphsage/gat/hgt}.py
* 08_combined-addition-weighted-{graphsage/gat/hgt}.py
* 09_combined-average-{graphsage/gat/hgt}.py
* 10_combined-nn-{graphsage/gat/hgt}.py

The result files contain the number of trained epochs, the validation and training loss for each epoch, the values of the test metrics and the number of trainable parameters of the GNN models. 
