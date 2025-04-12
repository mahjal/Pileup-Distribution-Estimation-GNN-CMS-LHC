hc=64

import os
from PUGNN import SW, HomoDataset, HomoDataReader, BoostedDataLoader, MultibatchTrainer
import numpy as np

input_directory  = '/eos/cms/store/user/hbakhshi/PUGNN/'
output_directory = '/eos/user/m/mjalalva/SWAN_projects/inConc/Folder/Edge_Threshold_02/AdvancedGCN/-VK/jun29/'


software1 = SW(input_directory, output_directory, name=f'model-{hc}')
sample_metadata = dict(zip(map(str, list(range(10, 91))), np.ones(81) * 700 ))
software1.set_dataset(HomoDataset, sample_metadata, HomoDataReader())
software1.set_loader(BoostedDataLoader, loading_workers=4, batch_size=64, num_workers=16)


from torch_geometric.nn import GraphConv, MLP, global_add_pool, GATv2Conv, LayerNorm, global_mean_pool
import torch
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.utils import add_self_loops

class AdvancedGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate):
        super(AdvancedGCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()

        # Input layer                                                                                                                                                                                                                                                                                                                                                                                                                                                       
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        self.dropouts.append(torch.nn.Dropout(dropout_rate))

        # Hidden layers                                                                                                                                                                                                                                                                                                                                                                                                                                                     
        for _ in range(1, num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(torch.nn.Dropout(dropout_rate))

        # Output layer                                                                                                                                                                                                                                                                                                                                                                                                                                                      
        self.linear = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr, y = data['x'], data['edge_index'], data.get('edge_attr'), data['y'].float()
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        for conv, bn, dropout in zip(self.convs, self.bns, self.dropouts):
            x = F.relu(bn(conv(x, edge_index)))

        x = global_mean_pool(x, data['batch'])

        x = self.linear(x)

        return x


input_dim = 15
hidden_dim = 64
output_dim = 1
num_layers = 5
dropout_rate = 0.5

model = AdvancedGCN(input_dim, hidden_dim, output_dim, num_layers, dropout_rate)
with software1.trainer_scope(MultibatchTrainer, num_batch=5) as pu_trainer:
    pu_trainer.set(model)
    res = pu_trainer.train(
        max_epochs=200, optimizer=torch.optim.RAdam,
        optimizer_args=dict(lr=5e-3),
        loss_fn=torch.nn.PoissonNLLLoss,
        loss_fn_args=dict(log_input=False, eps=1),
        metrics=[torch.nn.L1Loss()], select_topk=1,
        lr_scheduler=torch.optim.lr_scheduler.MultiStepLR,
        lr_scheduler_args=dict(milestones=[6, 100], gamma=0.05),
    )

eval_model = AdvancedGCN(input_dim, hidden_dim, output_dim, num_layers, dropout_rate)
with software1.analyzer_scope() as pu_analyzer:
    pu_analyzer(eval_model, res.models, torch.nn.L1Loss())
    model = pu_analyzer.model
    fig = pu_analyzer.residual_plot()
    res5 = pu_analyzer.distribution_plots()
    final_res = pu_analyzer.rangeLLE(30,60)
    res2 = pu_analyzer.LLEVisual(50)
#    res2.plot.write_html(os.path.join(output_directory, 'res2_plot.html'))                                                                                                                                                                                                                                                                                                                                                                                                 
    nv = pu_analyzer.extract_feature(0, 7)
    comp = pu_analyzer.compare(NV=(pu_analyzer.y, nv))
    centered_ll_plot = pu_analyzer.centered_log_likelihood_plot(starting_pu=40, ending_pu=60, fraction=0.3)






