from .base import BaseDataReader, BaseDataset, BaseDataloader, BaseTrainer, BaseAnalyzer
from .utils.processing_tools import get_data, to_data, get_batch, edges_threshold
from .utils.preprocessing_tools import check_and_summarize
from .utils.postprocessing_tools import llikelihood_pois, max_log_likelihood
from collections import namedtuple
from tqdm.notebook import trange, tqdm

from torch_geometric.utils import to_undirected, add_self_loops
import plotly.io as pio

import plotly
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import torch
import datetime 
import sys
import os.path as osp
import os
import pytz
import os
import shutil
from scipy.stats import poisson

from torch_geometric.data import Dataset
from torch.utils.data import DataLoader as torch_dataloader


class HomoDataReader(BaseDataReader):
    def get(self, path_to_file, group, index, log=False):
        failed, res = get_data(path_to_file, group, index)
        if failed and log:
            print(res, file=sys.stderr)
            raise FileNotFoundError
        
        node_features = np.array(res["node_features"], dtype=np.float32)
        edge_attrs = np.array(res["edge_attributes"], dtype=np.float32)
        edge_index = np.array(res["edge_indecies"], dtype=np.int64)
        features   = np.array(res["graph_features"], dtype=np.float32)
        labels     = np.array(res["graph_labels"], dtype=np.float32)
                
        has_nan, res1 = self.remove_nan_nodes(node_features, edge_index, edge_attrs)
        has_inf, res2 = self.remove_inf_nodes(node_features, edge_index, edge_attrs)
        
        if has_nan:
            node_features, edge_index, edge_attrs = res1
            if log:
                print("'{}' : 'PU={}' : 'E{}' : nan".format(path_to_file, group, index), file=sys.stderr)
        if has_inf:
            node_features, edge_index, edge_attrs = res2
            if log:
                print("'{}' : 'PU={}' : 'E{}' : inf".format(path_to_file, group, index), file=sys.stderr)
        
        edge_index, edge_attrs = edges_threshold(edge_index, edge_attrs, threshold=0.2)
        
        try:
            # Convert numpy arrays to PyTorch tensors
            node_features = torch.tensor(node_features)
            edge_attrs = torch.tensor(edge_attrs)
            edge_index = torch.tensor(edge_index)
            features = torch.tensor(features)
            labels = torch.tensor(labels)
            
            return to_data(node_features, edge_attrs, edge_index, features, labels)
        except Exception as e:
            print(has_inf, has_nan, np.array(res["node_features"], dtype=np.float32).shape)
            print(e)
            raise RuntimeError




class HeteroDataReader(BaseDataReader):
    def __call__(self, path_to_file, group, index, log=False):
        ...


class HomoDataset(BaseDataset):
    def get(self, idx):
        filename, PU, infile_index = self._indexing_system._get_item(idx)
        path_to_file = osp.join(self._in_dir, filename + ".h5")
        data = self._data_reader(path_to_file, f"PU{PU}", f"E{infile_index}")
        return data


class DataLoader(BaseDataloader):
    def process(self, dataset_length, test_percentage, validation_percentage, **dataloader_args):
        super().process(dataset_length, test_percentage, validation_percentage, **dataloader_args)
        self._loader_metadata = dict(
            zip(['test', 'train', 'validation'], [self._test_gen, self._train_gen, self._validation_gen])
            )

    def __iter__(self):
        if not self._open:
            raise RuntimeError("Iteration only availabele when you open the dataloader")
        self._iterator = iter(self._loader_metadata[self._context_loader])
        return self
    
    def __next__(self):
        return next(self._iterator).to(self._context_device)


class _BatchDataset(Dataset):
    def __init__(self, root, num_batch, transform=None, pre_transform=None, pre_filter=None):
        self._num_batches = num_batch
        self._zeros = len(str(num_batch))
        self._files = [f'batch_{ind:0{self._zeros}d}.pt' for ind in range(num_batch)]
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self):
        return self._files

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        path_to_file = osp.join(self.root, f'batch_{idx:0{self._zeros}d}.pt')
        failed, res = get_batch(path_to_file)
        if failed:
            print(res, file=sys.stderr)
            raise FileNotFoundError(f'Cache directory is not reachable at this moment for {path_to_file}...')
        return res


class BoostedDataLoader(BaseDataloader):
    def __init__(self, *args, **kwargs):
        self._loading_workers = kwargs['loading_workers']
        kwargs.pop('loading_workers', None)
        super().__init__(*args, **kwargs)

    def process(self, dataset_length, test_percentage, validation_percentage, **dataloader_args):
        super().process(dataset_length, test_percentage, validation_percentage, **dataloader_args)
        print('Initializing...', file=sys.stderr)
        for name in ['test', 'train', 'validation']:
            os.mkdir(osp.join(self._root, name))

        with tqdm(total=sum(list(self._num.values()))//self._batch_size, desc=f"loading ...", unit='Batch', disable=self._prog) as pbar1:
            for subset, name in zip([self._test_gen, self._train_gen, self._validation_gen], ['test', 'train', 'validation']):
                zeros  = len(str(len(subset)))
                branch = osp.join(self._root, name)
                with tqdm(total=len(subset), desc=f"preparing directory '{name}' ...", unit='Batch', disable=self._prog) as pbar2:
                    for ind, data in enumerate(subset):
                        path_to_data = osp.join(branch, f'batch_{ind:0{zeros}d}.pt')
                        torch.save(data, path_to_data)
                        pbar2.update(1)
                        pbar1.update(1)
                self._loader_metadata[name] = torch_dataloader(
                    _BatchDataset(branch, self.lengths[name]), 
                    num_workers=self._loading_workers, collate_fn=lambda x: x,
                    )
        print('done.', file=sys.stderr)

    def __iter__(self):
        if not self._open:
            raise RuntimeError("Iteration only availabele when you open the dataloader")
        self._iterator = iter(self._loader_metadata[self._context_loader])
        # print(self._context_loader, self._last_iter)
        return self
    
    def __next__(self):
        data = next(self._iterator)
        if len(data) > 1:
            print(len(data))
        return data[0].to(self._context_device)

class Trainer(BaseTrainer):
    def train_one_epoch(self, epoch=-1):
        self._model.train()
        # self._loader.describe()
        with self._loader('train', self._device) as train_loader:
            total  = train_loader.num()
            length = train_loader.len()
            with tqdm(total=length, desc=f"Epoch {epoch+1:03d}", unit="Batch", disable=self._prog) as pbar:
                for data in train_loader:
                    out  = self._model(data)   # Perform a single forward pass.
                    loss = self._loss_func(out, data.y.unsqueeze(1))
                    if np.isnan(loss.item()):
                        w = "nan loss detected. Perhaps there is a divergence. Stopping the training..."
                        return 1, (w, data)
                    loss.backward()              # Derive gradients.
                    self._optimizer.step()       # Update parameters based on gradients.
                    self._optimizer.zero_grad()  # Clear gradients.
                    pbar.update(1)
        return 0, None

    def evaluate(self, subset, metrics=[]):
        self._model.eval()
        # self._loader.describe()
        # Iterate in batches over the training/test/validation dataset.
        with self._loader(subset, self._device) as loader, torch.no_grad():
            total  = loader.num()
            length = loader.len()
            loss_arr = np.zeros((length, 1+len(metrics)))
            for b_ind, data in enumerate(loader):
                # loader.describe()
                data = data.to(self._device)
#                 out = self._model(data.x, data.edge_index, data.batch)  # Pass necessary arguments to model                
                out = self._model(data)  
                if out.cpu().detach().isnan().sum() > 0:
                    w = f"nan loss detected during evaluation of '{subset}' set. Perhaps there is a problem..."
                    return 1, (w, data)
                for ind, metric in enumerate([self._loss_func, *metrics]):
                    loss_arr[b_ind, ind] += metric(out, data.y.unsqueeze(1)).cpu().item() * len(data) / total
                b_ind += 1
        return 0, loss_arr.sum(0)

class MultibatchTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        try:
            self._n = kwargs['num_batch']
            kwargs.pop('num_batch', None)
        except KeyError:
            self._n = 5
        super().__init__(*args, **kwargs)
        
        
    def train_one_epoch(self, epoch=-1, clip=None):
        self._model.train()
        # self._loader.describe()
        data_list = []
        with self._loader('train', self._device) as train_loader:
            total  = train_loader.num()
            length = train_loader.len()
            with tqdm(total=length, desc=f"Epoch {epoch+1:03d}", unit="Batch", disable=self._prog) as pbar:
                for b_ind, data in enumerate(train_loader):
                    is_last = (b_ind == length - 1)
                    data_list.append(data)
                    if (b_ind % self._n == self._n - 1) or is_last:
                        target = torch.cat([d.y for d in data_list])
                        out    = torch.cat([self._model(d) for d in data_list])   # Perform a multiple forward pass.
#                         out = torch.cat([self._model(d.x, d.edge_index, d.batch) for d in data_list])  # Pass necessary arguments to model                        
                        loss = self._loss_func(out, target.unsqueeze(1))
                        if np.isnan(loss.item()):
                            w = "nan loss detected. Perhaps there is a divergance. Stopping the training..."
                            return 1, (w, data)
                        loss.backward()              # Derive gradients.
                        self._optimizer.step()       # Update parameters based on gradients.
                        self._optimizer.zero_grad()  # Clear gradients.
                        del data_list
                        data_list = []
                    pbar.update(1)
        return 0, None
    
    
class Analyzer(BaseAnalyzer):
    def _poisson_dist_extraction(self, av, f=0.3):
        prv = poisson(av)
        n  = len(self._y) / len(self._range)
        N  = int(f*n / prv.pmf(av))
        freq       = np.histogram(prv.rvs(N), self._range)[0]
        pu_freq    = dict(zip(self._range, freq))
        pu_counter = dict(zip(self._range, freq))
        pu_inds    = []

        for i, data in enumerate(tqdm(self._y,  desc=f'PU = {av} Indexing...', unit='Graph')):
            y = int(data)
            if pu_counter[y] > 0:
                pu_inds.extend([i])
                pu_counter[y] -= 1
        
        if sum(list(pu_counter.values())) > 1:
            w = f"Change factor variable `fraction` (current value is `{f}`) based on the following:"
            print(w, file=sys.stderr)
            for pu in pu_counter:
                if pu_counter[pu] > 0:
                    w = f'\tPoisson distribution({av}) for PU = {pu} is violated by {pu_counter[pu]} count less than expected'
                    print(w, file=sys.stderr)
        return pu_inds
    
    

    def LLEVisual(self, av_pu, f=0.3):
        _, x, xhat = self.LLEstimation(av_pu, f)
        # print(f'xhat = ({xhat}) in LLEVisual')
        # print(f'x = ({x}) in LLEVisual')
        n = len(self._y) / len(self._range)

        print(f"n: {n}")
        print(f"x mean: {x.mean()}")
        print(f"xhat mean: {xhat.mean()}")

        try:
            prv1 = poisson(av_pu)
            pmf_x_mean = prv1.pmf(int(x.mean()))
            if pmf_x_mean == 0:
                raise ValueError(f"PMF of x.mean() is zero for av_pu={av_pu}")
            N1 = int(f * n / pmf_x_mean)
            print(f"N1: {N1}")
            gen_dist = [prv1.pmf(x_i) * N1 for x_i in self._range]

            prv2 = poisson(xhat.mean())
            pmf_xhat_mean = prv2.pmf(int(xhat.mean()))
            if pmf_xhat_mean == 0:
                raise ValueError(f"PMF of xhat.mean() is zero for xhat.mean={xhat.mean()}")
            N2 = int(f * n / pmf_xhat_mean)
            print(f"N2: {N2}")
            gnn_dist = [prv2.pmf(x_i) * N2 for x_i in self._range]
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

        # Create the figure
        fig = go.Figure()

        # Add traces
        fig.add_trace(go.Histogram(x=x, name='PU Distribution', opacity=0.75))
        fig.add_trace(go.Histogram(x=xhat, name='GNN Output Distribution', opacity=0.75))
        fig.add_trace(go.Scatter(x=self._range, y=gen_dist, name='PU Poisson Distribution Generator'))
        fig.add_trace(go.Scatter(x=self._range, y=gnn_dist, name='GNN Fitted Poisson Distribution'))

        # Update layout with a larger title font size and legend font size
        fig.update_layout(
            title=f"$\lambda=<PU>=L\sigma={av_pu}$",
            title_font_size=30,  # Increase the title font size
            legend=dict(
                font=dict(
                    size=20  # Increase the font size for the trace names (legend labels)
                )
            )
        )

        # Save the figure as a high-quality PNG image
        output_png_path = osp.join(self._output_dir, 'LLEVisual.png')
        pio.write_image(fig, output_png_path, width=1200, height=800, scale=2)

        # Save the plot as HTML and JSON
        pio.write_html(fig, osp.join(self._output_dir, 'LLEVisual.html'))
        pio.write_json(fig, osp.join(self._output_dir, 'LLEVisual.json'))

        # Define the named tuple and return the values
        s = namedtuple('LLEVisual', ['y', 'yhat', 'generator_dist', 'predicted_dist', 'plot'])
        return s(x, xhat, gen_dist, gnn_dist, fig)


    def LLEstimation(self, av_pu, f=0.3):
        inds = self._poisson_dist_extraction(av_pu, f)
        xhat = self._yhat[inds]
        x    = self._y[inds]
#         print (inds)
#         print(f' inds = ({inds}) in LLEstimation')
#         print(f' x = ({x}) LLEstimation')
#         print(f' xhat = ({xhat}) in LLEstimation')
#         print(f' xhat.mean = ({xhat.mean}) LLEstimation')
        return max_log_likelihood(xhat, llikelihood_pois, av_pu), x, xhat
    
    def rangeLLE(self, starting_pu, ending_pu, fraction=0.3):
        av_PUs = np.array([*range(starting_pu, ending_pu + 1)])
        mle_arr       = np.zeros(len(av_PUs))
        act_arr       = np.zeros_like(mle_arr)
        left_err_arr  = np.zeros_like(mle_arr)
        right_err_arr = np.zeros_like(mle_arr)

        for ind, pu in enumerate(tqdm(av_PUs, desc='Range-Based LLE',  unit='PU')):
            temp, x, xhat = self.LLEstimation(pu, fraction)
            act = x.mean()
            le, mle, re = temp
            
            mle_arr[ind]       = mle
            act_arr[ind]       = act
            left_err_arr[ind]  = le
            right_err_arr[ind] = re
        

        fig = go.Figure(data=go.Scatter(
            x=av_PUs,
            y=mle_arr,
            error_y=dict(
                type='data',
                symmetric=False,
                array=right_err_arr,
                arrayminus=left_err_arr,
                visible=True),
            name='Estimated <PU>',
            mode='markers'
        ))

        fig.add_trace(go.Scatter(x=av_PUs, y=av_PUs, name='Expected'))
        fig.update_xaxes(title='$L\sigma_{pp}$')
        fig.update_layout(
            title='Model Reliability for Poisson Distribution',
            title_font_size=30,  # Increase title font size
            xaxis=dict(
                title_font=dict(size=20)  # Increase x-axis label font size
            ),
            yaxis=dict(
                title_font=dict(size=20)  # Increase y-axis label font size
            ),
            legend=dict(
                font=dict(
                    size=20  # Increase legend font size
                )
            )
        )

        s = namedtuple('LogLikelihood_PU_Estimation',
                       ['plot', 'true_pu', 'estimated_pu', 'lower_bond_error', 'upper_bond_error'])

        # Save the figure as a high-quality PNG image
        output_png_path = osp.join(self._output_dir, 'model-reliabilty-poisson-dist.png')
        pio.write_image(fig, output_png_path, width=1200, height=800, scale=2)
    

        
        plotly.io.write_html(fig,  osp.join(self._output_dir, 'model-reliabilty-poisson-dist.html'))
        plotly.io.write_json(fig,  osp.join(self._output_dir, 'model-reliabilty-poisson-dist.json'))

        return s(fig, act_arr, mle_arr, left_err_arr, right_err_arr)
    
    
    def centered_log_likelihood_plot(self, starting_pu, ending_pu, fraction=0.3):
        av_PUs = np.array([*range(starting_pu, ending_pu + 1)])
        pu_mean = np.array([*range(10, 91)])

        fig = go.Figure()
        fig.update_layout(title="Centered Log-Likelihood Function")

        for pu in tqdm(av_PUs, desc='Generating Centered Log-Likelihood Plot...', unit='PU'):
            inds = self._poisson_dist_extraction(pu, fraction)
            arr = self._y[inds]
            llikelihood = [llikelihood_pois(arr, lam) for lam in pu_mean]

            mle = arr.mean()

            fig.add_trace(go.Scatter(x=pu_mean - mle, y=llikelihood, name=f'PU = {pu}'))

        fig.show()
        plotly.io.write_html(fig, osp.join(self._output_dir, 'centered-log-likelihood.html'))
        plotly.io.write_json(fig, osp.join(self._output_dir, 'centered-log-likelihood.json'))
        
        output_png_path = osp.join(self._output_dir, 'centered-log-likelihood.png')
        pio.write_image(fig, output_png_path, width=1200, height=800, scale=2)        
        return fig

    
class SW(object):
    def __init__(self, input_dir, root, name, seed=42) -> None:
        self._seed = seed
        self._time = str(datetime.datetime.now(pytz.timezone('Asia/Tehran'))).split('.')[0].replace(' ', '@')
        self._root = root
        self._name = name
        self._in_dir       = input_dir
        self._main_dir     = osp.join(root, f'{name}-pugnnsw')
        self._cache_dir    = osp.join(self._main_dir, f'cache-{self._time}')    
        self._output_dir   = osp.join(self._main_dir, f'out-{self._time}')
        self._metadata_dir = osp.join(self._main_dir, 'metadata')

        if not osp.isdir(self._main_dir):
            os.mkdir(self._main_dir)

        if not osp.isdir(self._metadata_dir):
            os.mkdir(self._metadata_dir)

        if not osp.isdir(self._output_dir):
            os.mkdir(self._output_dir)

        os.mkdir(self._cache_dir)

        print('Preprocessing...', file=sys.stderr)
        nfiles = check_and_summarize(self._in_dir, self._metadata_dir)
        print(nfiles, f"found at '{self._in_dir}'", file=sys.stderr)
    
    @property
    def root(self):
        return self._root

    @property
    def name(self):
        return self._name

    @property
    def main_directory(self):
        return self._main_dir

    @property 
    def dataset(self):
        return self._dataset

    @property
    def loader(self):
        return self._loader
    
    def set_dataset(self, Dataset, sample_metadata, data_reader, **kwargs) -> None:
        self._dataset = Dataset(self._metadata_dir, self._in_dir, sample_metadata, data_reader, self._seed, **kwargs)

    def set_loader(self, DataLoaderClass, **kwargs):
        self._loader = DataLoaderClass(self.dataset, self._seed, self._cache_dir, **kwargs)
    
    def trainer_scope(self, Trainer_type=Trainer, **kwargs):
        return Trainer_type(self._output_dir, self.loader, self._seed, **kwargs)

    def analyzer_scope(self, Analyzer_type=Analyzer, **kwargs):
        return Analyzer_type(self._output_dir, self.loader, self._seed, **kwargs)

    def __del__(self):
        shutil.rmtree(self._cache_dir)
