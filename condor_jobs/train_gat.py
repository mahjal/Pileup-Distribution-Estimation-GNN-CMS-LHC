import torch.nn.functional as F
import torch
from torch_geometric.nn import GraphConv, MLP, global_add_pool, GATv2Conv, LayerNorm, global_mean_pool
import numpy as np
from src import SW, HomoDataset, HomoDataReader, BoostedDataLoader, MultibatchTrainer
import os
hc = 64


import sys
sys.path.append('/home/mahtab/Pileup-Distribution-Estimation-GNN-CMS-LHC/src')  # Absolute path

#print("Current sys.path:", sys.path)  # Print sys.path to check

from src import SW, HomoDataset, HomoDataReader, BoostedDataLoader, MultibatchTrainer




hc = 64


#input_directory = '/eos/cms/store/user/hbakhshi/PUGNN/'
#input_directory = '/eos/home-i03/m/mjalalva/Run1/Dec25/'
input_directory ='/home/mahtab/Pileup-Distribution-Estimation-GNN-CMS-LHC/data/'
output_directory = '.'


software1 = SW(input_directory, output_directory, name=f'model-{hc}')
sample_metadata = dict(zip(map(str, list(range(20, 51))), np.ones(81) * 70))
software1.set_dataset(HomoDataset, sample_metadata, HomoDataReader())

software1.set_loader(
    BoostedDataLoader,
    loading_workers=4,
    batch_size=64,
    num_workers=16)

software1.set_loader(BoostedDataLoader, loading_workers=4,
                     batch_size=64, num_workers=16)





class PUModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,
                 out_channels, num_features, GNN=GraphConv):
        super(PUModel, self).__init__()
        torch.manual_seed(12345)
        self.num_features = num_features


        self.conv1 = GNN(
            in_channels,
            hidden_channels,
            edge_dim=1,
            add_self_loops=False)
        self.norm1 = LayerNorm(hidden_channels)
        self.mlp1 = MLP([hidden_channels +
                         num_features, 2 *
                         hidden_channels, 2 *
                         hidden_channels], norm='layer_norm')

        self.conv2 = GNN(
            hidden_channels,
            hidden_channels,
            edge_dim=1,
            add_self_loops=False)
        self.norm2 = LayerNorm(hidden_channels)
        self.mlp2 = MLP([3 *
                         hidden_channels +
                         num_features, 2 *
                         hidden_channels, hidden_channels, hidden_channels //
                         2, out_channels], norm='layer_norm')

        self.conv1 = GNN(in_channels, hidden_channels,
                         edge_dim=1, add_self_loops=False)
        self.norm1 = LayerNorm(hidden_channels)
        self.mlp1 = MLP([hidden_channels + num_features, 2 *
                        hidden_channels, 2*hidden_channels], norm='layer_norm')

        self.conv2 = GNN(hidden_channels, hidden_channels,
                         edge_dim=1, add_self_loops=False)
        self.norm2 = LayerNorm(hidden_channels)
        self.mlp2 = MLP([3*hidden_channels + num_features, 2*hidden_channels,
                        hidden_channels, hidden_channels//2, out_channels], norm='layer_norm')


    def forward(self, data):
        x, adj, features, batch = data.x, data.adj_t, torch.reshape(
            data.features, (-1, self.num_features)), data.batch

        # 1. Obtain node embeddings

        x = self.conv1(x, adj)
        x = self.norm1(x)
        x = x.relu()

        g = self.mlp1(torch.cat([global_mean_pool(x, batch), features], dim=1))

        x = self.conv2(x, adj)
        x = self.norm2(x)
        x = x.relu()

        g = self.mlp2(
            torch.cat([global_mean_pool(x, batch), g, features], dim=1))

        return g  


model = PUModel(
    in_channels=15,
    hidden_channels=hc,
    num_features=7,
    out_channels=1,
    GNN=GATv2Conv)

model = PUModel(in_channels=16, hidden_channels=hc,
                num_features=7, out_channels=1, GNN=GATv2Conv)

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


eval_model = PUModel(
    in_channels=15,
    hidden_channels=hc,
    num_features=7,
    out_channels=1,
    GNN=GATv2Conv)

eval_model = PUModel(in_channels=15, hidden_channels=hc,
                     num_features=7, out_channels=1, GNN=GATv2Conv)

with software1.analyzer_scope() as pu_analyzer:
    pu_analyzer(eval_model, res.models, torch.nn.L1Loss())
    model = pu_analyzer.model
    fig = pu_analyzer.residual_plot()
    res5 = pu_analyzer.distribution_plots()
    final_res = pu_analyzer.rangeLLE(30, 60)
    res2 = pu_analyzer.LLEVisual(50)
#    res2.plot.write_html(os.path.join(output_directory, 'res2_plot.html'))
    nv = pu_analyzer.extract_feature(0, 7)
    comp = pu_analyzer.compare(NV=(pu_analyzer.y, nv))

    centered_ll_plot = pu_analyzer.centered_log_likelihood_plot(
        starting_pu=40, ending_pu=60, fraction=0.3)
