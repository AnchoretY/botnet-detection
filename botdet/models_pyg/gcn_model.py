import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from .gcn_base_models import NodeModelAdditive, NodeModelMLP
from .graph_attention import NodeModelAttention
from .common import activation


class GCNModel(nn.Module):
    """
    图神经网络模型，包含GCN Layer，残差连接、最终输出层几部分。

    Args:
        in_channels (int): input channels
        enc_sizes (List[int]): 每层输出通道数, e.g. [32, 64, 64, 32]
        num_classes (int): 最终预测的类别数
        non_linear (str): 非线性激活函数
        non_linear_layer_wise (str): 非线性激活函数在每层的残差之前，默认为none，一般不更改 
        residual_hop (int): 每隔几层建立一个残差连接. 如果维度是相同的，输出将来自之前层层输出的直接加和，否则先使用一个无bias的线性转换层进行转换再加.
        dropout (float): 应用在隐藏层节点的dropout系数 (不包括初始的输入特征).
        final_layer_config (dict):最后一层的配置参数,当最后一层是直接的输出层并且你想更改一些最后一层的设置时该参数起作用，例如想在最后一层中也加入多头attention等
        final_type (str): 最后一层预测分数的类型. Default: 'none'.
        pred_on (str): 预测认为类型，进行node分类还是graph分类， Default: 'node'.
        **kwargs: could include other configuration arguments for each layer, such as for graph attention layers.

    Input:
        - x (torch.Tensor): 节点特征 (B * N, C_in)
        - edge_index (torch.LongTensor): COO格式的边索引 (2, E)
        - edge_attr (torch.Tensor, optional): 边特征(E, D_in)
        - deg (torch.Tensor, optional): 节点的度 (B * N,); 
        - edge_weight (torch.Tensor, optional): 边权重，几乎不使用

    Output:
        - x (torch.Tensor): 更新了的节点特征，用来进行节点分类或者图分类 

    where
        B: 批处理图的数量
        N: 节点数量
        E: 边数量
        C_in: 输入节点特征维度
        num_classes: 分类数
        D_in: 输入的边特征维度
    """

    def __init__(self, in_channels, enc_sizes, num_classes, non_linear='relu', non_linear_layer_wise='none',
                 residual_hop=None, dropout=0.0, final_layer_config=None, final_type='none', pred_on='node', **kwargs):
        assert final_type in ['none', 'proj']
        assert pred_on in ['node', 'graph']
        super().__init__()

        self.in_channels = in_channels
        self.enc_sizes = [in_channels, *enc_sizes]
        self.num_layers = len(self.enc_sizes) - 1
        self.num_classes = num_classes
        self.residual_hop = residual_hop
        self.non_linear_layer_wise = non_linear_layer_wise
        self.final_type = final_type
        self.pred_on = pred_on

        # 允许不同层使用不同头数的卷积神经网络，
        if 'nheads' in kwargs:
            if isinstance(kwargs['nheads'], int):
                self.nheads = [kwargs['nheads']] * self.num_layers
            elif isinstance(kwargs['nheads'], list):
                self.nheads = kwargs['nheads']
                assert len(self.nheads) == self.num_layers
            else:
                raise ValueError
            del kwargs['nheads']
        else:
            # 否则就使用placeholder for 'nheads'
            self.nheads = [1] * self.num_layers
        
        # 使用多个GCNLayer叠加组成模型的主题结构
        if final_layer_config is None:
            self.gcn_net = nn.ModuleList([GCNLayer(in_c, out_c, nheads=nh, non_linear=non_linear_layer_wise, **kwargs)
                                          for in_c, out_c, nh in zip(self.enc_sizes, self.enc_sizes[1:], self.nheads)])
        else:
            assert isinstance(final_layer_config, dict)
            self.gcn_net = nn.ModuleList([GCNLayer(in_c, out_c, nheads=nh, non_linear=non_linear_layer_wise, **kwargs)
                                          for in_c, out_c, nh in zip(self.enc_sizes[:-2],
                                                                     self.enc_sizes[1:-1],
                                                                     self.nheads[:-1])])
            kwargs.update(final_layer_config)    # this will update with the new values in final_layer_config
            self.gcn_net.append(GCNLayer(self.enc_sizes[-2], self.enc_sizes[-1], nheads=self.nheads[-1],
                                         non_linear=non_linear_layer_wise, **kwargs))
        # 添加dropout
        self.dropout = nn.Dropout(dropout)
        # 添加残差结构
        if residual_hop is not None and residual_hop > 0:
            self.residuals = nn.ModuleList([nn.Linear(self.enc_sizes[i], self.enc_sizes[j], bias=False)
                                            if self.enc_sizes[i] != self.enc_sizes[j]
                                            else
                                            nn.Identity()
                                            for i, j in zip(range(0, len(self.enc_sizes), residual_hop),
                                                            range(residual_hop, len(self.enc_sizes), residual_hop))])
            self.num_residuals = len(self.residuals)

        self.non_linear = activation(non_linear)
        # 输出层设置
        if self.final_type == 'none':
            self.final = nn.Identity()
        elif self.final_type == 'proj':
            self.final = nn.Linear(self.enc_sizes[-1], num_classes)
        else:
            raise ValueError

    def reset_parameters(self):
        for net in self.gcn_net:
            net.reset_parameters()
        if self.residual_hop is not None:
            for net in self.residuals:
                net.reset_parameters()
        if self.final_type != 'none':
            self.final.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, deg=None, edge_weight=None, **kwargs):
        xr = None
        add_xr_at = -1

        for n, net in enumerate(self.gcn_net):
            # 通过线性激活函数的GCN层
            xo = net(x, edge_index, edge_attr, deg, edge_weight, **kwargs)
            xo = self.dropout(xo)
            
            # 残差连接处理
            if self.residual_hop is not None and self.residual_hop > 0:
                if n % self.residual_hop == 0 and (n // self.residual_hop) < self.num_residuals:
                    xr = self.residuals[n // self.residual_hop](x)  # 残差
                    add_xr_at = n + self.residual_hop - 1
                if n == add_xr_at:
                    if n < self.num_layers - 1:  # before the last layerds
                        # 在最后一层前面的每一层，都在原来的层之后与残差加和后使用非线性激活函数
                        xo = self.non_linear(xo + xr)
                    else:  # 最后一层（潜在的输出层，后面还可能有softmax）
                        # 在二级制分类中，不能使用非线性层，因为它将被传递到softmax计算损失。Relu将直降杀死所有为负值的部分
                        if self.final_type == 'none':
                            xo = xo + xr
                        else:
                            xo = self.non_linear(xo + xr)
            else:
                if n < self.num_layers - 1:  
                    xo = self.non_linear(xo)
                else:
                    if self.final_type == 'none':
                        pass
                    else:
                        xo = self.non_linear(xo)
            x = xo
            
        # x: (B * N, self.enc_sizes[-1]) -> (B * N, num_classes)
        x = self.final(x)

        # 使用图级别的平均池化，用来图分类
        if self.pred_on == 'graph':
            assert 'batch_slices_x' in kwargs
            batch_slices_x = kwargs['batch_slices_x']
            if len(batch_slices_x) == 2:
                # 一个batch中只有一个graph时
                x = x.mean(dim=0, keepdim=True)  # size (1, num_classes)
            else:
                # 一个batch中超过一张graph时
                x_batch, lengths = zip(*[(x[i:j], j - i) for (i, j) in zip(batch_slices_x, batch_slices_x[1:])])
                x_batch = pad_sequence(x_batch, batch_first=True,
                                       padding_value=0)  # size (batch_size, max_num_nodes, num_classes)
                x = x_batch.sum(dim=1) / x_batch.new_tensor(lengths)  # size (batch_size, num_classes)

        return x


class GCNLayer(nn.Module):
    """
    图卷积层. 各种节点更新模型的封装，例如基本加法模型、MLP、attention模型。也可以拓展为边更新模型和extra read out operations.
    Args:
        in_channels (int): input channels
        out_channels (int): output channels
        in_edgedim (int, optional): 输入边维度
        deg_norm (str, optional): 出度正则化方法.['none', 'sm', 'rw']. 默认为'sm'.
            'sm': symmetric, 更适合无向图. 'rw': random walk, 更适合有向图.
            注意：当sm用于有向图时，如果有的节点没有出度，将会报错
        edge_gate (str, optional): method of apply edge gating mechanism.  ['none', 'proj', 'free'].
            Note that when set to 'free', should also provide `num_edges` as an argument (but then it can only work with
            fixed edge graph).
        aggr (str, optional): 整合邻居特征的方法. ['add', 'mean', 'max'].默认为'add'.
        bias (bool, optional): 是否使用bias. 默认为True.
        nodemodel (str, optional): 要进行封装的节点模型名称.['additive','mlp','attention']
        non_linear (str, optional): 非线性激活函数名称.
        **kwargs: could include `num_edges`, etc.
    """
    # 输入的字符串和选择的模型映射字典
    nodemodel_dict = {'additive': NodeModelAdditive,
                      'mlp': NodeModelMLP,
                      'attention': NodeModelAttention}

    def __init__(self, in_channels, out_channels, in_edgedim=None, deg_norm='sm', edge_gate='none', aggr='add',
                 bias=True, nodemodel='additive', non_linear='relu', **kwargs):
        assert nodemodel in ['additive', 'mlp', 'attention']
        super().__init__()
        self.gcn = self.nodemodel_dict[nodemodel](in_channels,
                                                  out_channels,
                                                  in_edgedim,
                                                  deg_norm=deg_norm,
                                                  edge_gate=edge_gate,
                                                  aggr=aggr,
                                                  bias=bias,
                                                  **kwargs)

        self.non_linear = activation(non_linear)

    def forward(self, x, edge_index, edge_attr=None, deg=None, edge_weight=None, **kwargs):
        xo = self.gcn(x, edge_index, edge_attr, deg, edge_weight, **kwargs)
        xo = self.non_linear(xo)
        return xo
