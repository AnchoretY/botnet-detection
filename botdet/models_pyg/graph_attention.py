import torch
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros

from .gcn_base_models import NodeModelBase
from .common import activation, softmax


class NodeModelAttention(NodeModelBase):
    """
    Multi-head soft attention over a node's neighborhood.
    Note:
        - Inheritance to :class:`NodeModelBase` is only for organization purpose, which is actually not necessary
          So deg_norm=None, edge_gate=None, aggr='add' (defaults), and they are not currently used.
        - 当att_combine为cat是，每个头的out_channel为out_channnels/nheads，
          否则out_channel每个头都要有
    """

    def __init__(self, in_channels, out_channels, in_edgedim=None,
                 nheads=1, att_act='none', att_dropout=0, att_combine='cat', att_dir='in', bias=False, **kwargs):
        assert att_act in ['none', 'lrelu', 'relu']
        assert att_combine in ['cat', 'add', 'mean']
        assert att_dir in ['in', 'out']

        super(NodeModelAttention, self).__init__(in_channels, out_channels, in_edgedim)

        self.nheads = nheads
        if att_combine == 'cat':
            self.out_channels_1head = out_channels // nheads
            assert self.out_channels_1head * nheads == out_channels, 'out_channels should be divisible by nheads'
        else:
            self.out_channels_1head = out_channels

        self.att_combine = att_combine
        self.att_dir = att_dir

        if att_combine == 'cat':
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        else:    # 'add' or 'mean':
            self.weight = Parameter(torch.Tensor(in_channels, out_channels * nheads))
        self.att_weight = Parameter(torch.Tensor(1, nheads, 2 * self.out_channels_1head))
        self.att_act = activation(att_act)
        self.att_dropout = nn.Dropout(p=att_dropout)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att_weight)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr=None, deg=None, edge_weight=None, attn_store=None, **kwargs):
        """
        'deg' and 'edge_weight' are not used. Just to be consistent for API.
        """
        x = torch.mm(x, self.weight).view(-1, self.nheads, self.out_channels_1head)  # size (N, n_heads, C_out_1head)

        # 对源节点和邻域节点的信息进行合并后线性变换
        x_j = torch.index_select(x, 0, edge_index[0])
        x_i = torch.index_select(x, 0, edge_index[1])
        x_data = torch.cat([x_j, x_i], dim=-1) * self.att_weight


        # 计算注意力的因子, size (E, nheads)
        alpha = self.att_act(x_data.sum(dim=-1))

        # softmax over each node's neighborhood, size (E, nheads)
        if self.att_dir == 'out':
            # random walk
            alpha = softmax(alpha, edge_index[0], num_nodes=x.size(0))
        else:
            # attend over nodes that all points to the current one
            alpha = softmax(alpha, edge_index[1], num_nodes=x.size(0))

        # 在attention系数上应用dropout（即在训练期间，邻居具有一定的随机采样性）
        alpha = self.att_dropout(alpha)

        # 使用attention系数来调节各个节点的信息比重
        x_j = x_j * alpha.view(-1, self.nheads, 1)

        # 整合节点特征，（N,nheads,C_out_1head）
        x = scatter_(self.aggr, x_j, edge_index[1], dim_size=x.size(0))

        # 合并多个头产生的结果,  (N, C_out)
        if self.att_combine == 'cat':
            x = x.view(-1, self.out_channels)
        elif self.att_combine == 'add':
            x = x.sum(dim=1)
        else:
            x = x.mean(dim=1)

        # 添加 bias
        if self.bias is not None:
            x = x + self.bias
  
        if attn_store is not None:    # attn_store is a callback list in case we want to get the attention scores out
            attn_store.append(alpha)

        return x

    def __repr__(self):
        return ('{} (in_channels: {}, out_channels: {}, in_edgedim: {}, nheads: {}, att_activation: {},'
                'att_dropout: {}, att_combine: {}, att_dir: {} | number of parameters: {}').format(
                self.__class__.__name__, self.in_channels, self.out_channels, self.in_edgedim,
                self.nheads, self.att_act, self.att_dropout.p, self.att_combine, self.att_dir, self.num_parameters())
