import torch
import torch.nn as nn
from torch.nn import Parameter
# from torch_geometric.utils import scatter_
from torch_geometric.nn.inits import glorot, zeros
from torch_scatter import scatter_add

from .common import scatter_
from .common import activation


class NodeModelBase(nn.Module):
    """
    基于节点和边权重更新节点权重的模型的基础模型。
    注意:非线性聚合方式采用add的方式

    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        in_edgedim (int, optional): 输入的边特征维度
        deg_norm (str, optional): 节点正则化常亮计算方法
            Choose from [None, 'sm', 'rw'].
        edge_gate (str, optional): method of applying edge gating mechanism. Choose from [None, 'proj', 'free'].
            Note: 当设置free时，应该提分that when set to 'free', should also provide `num_edges` as an argument (but then it can only work
            with fixed edge graph).
        aggr (str, optional): 信息传递方法. ['add', 'mean', 'max']，默认为'add'.
        **kwargs: could include `num_edges`, etc.

    Input:
        - x (torch.Tensor): 节点特征矩阵 (N, C_in)
        - edge_index (torch.LongTensor): COO 格式的边索引，(2, E)
        - edge_attr (torch.Tensor, optional): 边特征矩阵 (E, D_in)

    Output:
        - xo (torch.Tensor):更新的节点特征 (N, C_out)

    where
        N: 输入节点数量
        E: 边数量
        C_in/C_out: 输入/输出节点特征的维度
        D_in: 输入的边特征维度
    """

    def __init__(self, in_channels, out_channels, in_edgedim=None, deg_norm='none', edge_gate='none', aggr='add',
                 *args, **kwargs):
        assert deg_norm in ['none', 'sm', 'rw']
        assert edge_gate in ['none', 'proj', 'free']
        assert aggr in ['add', 'mean', 'max']

        super(NodeModelBase, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_edgedim = in_edgedim
        self.deg_norm = deg_norm
        self.aggr = aggr

        if edge_gate == 'proj':
            self.edge_gate = EdgeGateProj(out_channels, in_edgedim=in_edgedim, bias=True)
        elif edge_gate == 'free':
            assert 'num_edges' in kwargs  # note: this will restrict the model to only a fixed number of edges
            self.edge_gate = EdgeGateFree(kwargs['num_edges'])  # so don't use this unless necessary
        else:
            self.register_parameter('edge_gate', None)

    @staticmethod
    def degnorm_const(edge_index=None, num_nodes=None, deg=None, edge_weight=None, method='sm', device=None):
        """
        计算归一化常数
        Calculating the normalization constants based on out-degrees for a graph.
        `_sm` 使用对称归一化，"symmetric". 更适合用于无向图.
        `_rw` 使用随即游走归一化(均值),"random walk". 更适合用于有向图.

        Procedure:
            - 检查edge_weight，如果不为None，那么必须同时提供edge_index和num_nodes，计算全部节点的度
            - 如果edge_weighe，如果是None，检查是否已经存在deg(节点的度矩阵):
                - 如果度矩阵存在，那么忽略edge_index和num_nodes
                - 如果度矩阵不存在，则必须提供edge_index和num_nodes，并计算全部节点的度
                
        Input:
            - edge_index (torch.Tensor): COO格式的图关系, (2, E)，long
            - num_nodes (int): 节点数量
            - deg (torch.Tensor): 节点的度,(N,),float
            - edge_weight (torch.Tensor): 边权重,(E,),float
            - method (str): 度标准化方法, choose from ['sm', 'rw']
            - device (str or torch.device): 驱动器编号

        Output:
            - norm (torch.Tensor): 基于节点度和边权重的标准化常数.
                If `method` == 'sm', size (E,);
                if `method` == 'rw' and `edge_weight` != None, size (E,);
                if `method` == 'rw' and `edge_weight` == None, size (N,).

        where
            N: 节点数量
            E: 边数量
        """
        assert method in ['sm', 'rw']

        if device is None and edge_index is not None:
            device = edge_index.device

        if edge_weight is not None:
            assert edge_index is not None, 'edge_index must be provided when edge_weight is not None'
            assert num_nodes is not None, 'num_nodes must be provided when edge_weight is not None'

            edge_weight = edge_weight.view(-1)
            assert edge_weight.size(0) == edge_index.size(1)
            calculate_deg = True    # 时候需要计算节点度
            edge_weight_equal = False
        else:
            if deg is None:
                assert edge_index is not None, 'edge_index must be provided when edge_weight is None ' \
                                               'but deg not provided'
                assert num_nodes is not None, 'num_nodes must be provided when edge_weight is None ' \
                                              'but deg not provided'
                edge_weight = torch.ones((edge_index.size(1),), device=device)
                calculate_deg = True
            else:
                # node degrees are provided
                calculate_deg = False
            edge_weight_equal = True

        row, col = edge_index
        # 计算节点度
        if calculate_deg:
            deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        # 节点度标准化
        if method == 'sm':
            deg_inv_sqrt = deg.pow(-0.5)
        elif method == 'rw':
            deg_inv_sqrt = deg.pow(-1)
        else:
            raise ValueError

        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        if method == 'sm':
        # 采用对称标准化的方式，得到的结果向量为(E,)
            norm = (deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col] if not edge_weight_equal # 注意，这里没有直接使用deg_inv_sqrt是因为要乘以权重
                    else deg_inv_sqrt[row] * deg_inv_sqrt[col])  # size (E,)
        elif method == 'rw':
        # 采用随即游走标准化，如果没有边权重矩阵，那么直接输出签名开方的结果，为（N,），否则与上面类似输出为(E,)
            norm = (deg_inv_sqrt[row] * edge_weight if not edge_weight_equal  # size (E,)
                    else deg_inv_sqrt)  # size (N,)
        else:
            raise ValueError

        return norm

    def forward(self, x, edge_index, edge_attr=None, deg=None, edge_weight=None, *args, **kwargs):
        return x

    def num_parameters(self):
        if not hasattr(self, 'num_para'):
            self.num_para = sum([p.nelement() for p in self.parameters()])
        return self.num_para

    def __repr__(self):
        return '{} (in_channels: {}, out_channels: {}, in_edgedim: {}, deg_norm: {}, edge_gate: {},' \
               'aggr: {} | number of parameters: {})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.in_edgedim,
            self.deg_norm, self.edge_gate.__class__.__name__, self.aggr, self.num_parameters())




class NodeModelAdditive(NodeModelBase):
    """
        通过邻域节点的节点和边特征更新节点特征，节点特征表示选用节点的出度
    """

    def __init__(self, in_channels, out_channels, in_edgedim=None, deg_norm='sm', edge_gate='none', aggr='sum',
                 bias=True,
                 **kwargs):
        super(NodeModelAdditive, self).__init__(in_channels, out_channels, in_edgedim, deg_norm, edge_gate, aggr,
                                                **kwargs)
        # 节点权重矩阵
        self.weight_node = Parameter(torch.Tensor(in_channels, out_channels))
        # 边权重矩阵
        if in_edgedim is not None:
            self.weight_edge = Parameter(torch.Tensor(in_edgedim, out_channels))
            
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight_node)
        if self.in_edgedim is not None:
            glorot(self.weight_edge)
        if self.bias is not None:
            zeros(self.bias)

    def forward(self, x, edge_index, edge_attr=None, deg=None, edge_weight=None, **kwargs):
        
        # 将节点特征转化为向量表达， (Node_nums, C_out)        
        x = torch.matmul(x, self.weight_node)
      
        # 构建边特征向量(如果存在的话)
        if edge_attr is not None:
            assert self.in_edgedim is not None
            x_je = torch.matmul(edge_attr, self.weight_edge)  # size (E, C_out)

        # 为信息传递准备节点特征, 包括信息normalization和合并边缘两部分
        if self.deg_norm == 'none':
            # 直接使用起始节点特征形成(E, C_out)的起始节点向量矩阵          
            x_j = torch.index_select(x, 0, edge_index[0])
        else:
            # 使用节点的度和边权重计算节点的正则化量，（E,）或（N,）
            norm = self.degnorm_const(edge_index, num_nodes=x.size(0), deg=deg,
                                      edge_weight=edge_weight, method=self.deg_norm, device=x.device)
            if self.deg_norm == 'rw' and edge_weight is None:
                x_j = x * norm.view(-1, 1)  # this saves much memory when N << E
                # lift the features to source nodes, resulting size (E, C_out)
                x_j = torch.index_select(x_j, 0, edge_index[0])
            else:
                # lift the features to source nodes, resulting size (E, C_out)
                x_j = torch.index_select(x, 0, edge_index[0])
                x_j = x_j * norm.view(-1, 1)  # norm.view(-1, 1) second dim set to 1 for broadcasting
                
        #----------------- 聚合节点+边特征，得到最终新的节点特征--------------
        # 获得最终要进行聚合的特征向量，是否包含边特征两种
        x_j = x_j + x_je if edge_attr is not None else x_j

        # 使用edge gates
        if self.edge_gate is not None:
            eg = self.edge_gate(x, edge_index, edge_attr=edge_attr, edge_weight=edge_weight)
            x_j = eg * x_j

        # 整合特征信息到节点中,这里需要重点理解 得到(N, C_out)
        x = scatter_(self.aggr, x_j, edge_index[1], dim_size=x.size(0))

        # 添加bias
        if self.bias is not None:
            x = x + self.bias

        return x


class NodeModelMLP(NodeModelBase):
    """
        将MLP应用在[node_features, edge_features]上来更新节点特征，节点特征使用标准化的出度
    Note:
        This is currently the same as the :class:`NodeModelAdditive` method,
        for a single layer MLP without non-linearity.
        There is a slight different when `bias` == True: here the bias is applied to messages on each edge
        before doing edge gates, whereas in the above model the bias is applied after aggregation on the nodes.
    """

    def __init__(self, in_channels, out_channels, in_edgedim=None, deg_norm='sm', edge_gate='none', aggr='add',
                 bias=True, mlp_nlay=1, mlp_nhid=32, mlp_act='relu',
                 **kwargs):
        super(NodeModelMLP, self).__init__(in_channels, out_channels, in_edgedim, deg_norm, edge_gate, aggr, **kwargs)

        if in_edgedim is None:
            in_features = in_channels
            # self.mlp = nn.Linear(in_channels, out_channels,
            #                      bias=bias)  # can also have multiple layers with non-linearity
        else:
            in_features = in_channels + in_edgedim
            # self.mlp = nn.Linear(in_channels + in_edgedim, out_channels, bias=bias)

        if mlp_nlay == 1:
            self.mlp = nn.Linear(in_features, out_channels, bias=bias)
        elif mlp_nlay >= 2:
            self.mlp = [nn.Linear(in_features, mlp_nhid, bias=bias)]
            for i in range(mlp_nlay - 1):
                self.mlp.append(activation(mlp_act))
                if i < mlp_nlay - 2:
                    self.mlp.append(nn.Linear(mlp_nhid, mlp_nhid, bias=bias))
                else:
                    # last layer, and we do not apply non-linear activation after
                    self.mlp.append(nn.Linear(mlp_nhid, out_channels, bias=bias))
            self.mlp = nn.Sequential(*self.mlp)

        # self.reset_parameters()

    def reset_parameters(self, initrange=0.1):
        # TODO: this only works for 1-layer mlp
        nn.init.uniform_(self.mlp.weight, -initrange, initrange)
        if self.mlp.bias is not None:
            nn.init.constant_(self.mlp.bias, 0)

        # self.mlp.reset_parameters()    # this was done automatically when nn.Linear class was initialized

    def forward(self, x, edge_index, edge_attr=None, deg=None, edge_weight=None, **kwargs):
        if self.deg_norm == 'none':
            row, col = edge_index
            x_j = x[row]  # size (E, C_in)
            # alternatively
            # x_j = torch.index_select(x, 0, edge_index[0])
        else:
            # calculate the degree normalization factors, of size (E,)
            # or of size (N,) when `self.deg_norm` == 'rw' and `edge_weight` == None
            norm = self.degnorm_const(edge_index, num_nodes=x.size(0), deg=deg,
                                      edge_weight=edge_weight, method=self.deg_norm, device=x.device)
            if self.deg_norm == 'rw' and edge_weight is None:
                x_j = x * norm.view(-1, 1)  # this saves much memory when N << E
                # lift the features to source nodes, resulting size (E, C_out)
                x_j = torch.index_select(x_j, 0, edge_index[0])
            else:
                # lift the features to source nodes, resulting size (E, C_out)
                x_j = torch.index_select(x, 0, edge_index[0])
                x_j = x_j * norm.view(-1, 1)  # norm.view(-1, 1) second dim set to 1 for broadcasting

        if edge_attr is not None:
            assert self.in_edgedim is not None
            x_j = self.mlp(torch.cat([x_j, edge_attr], dim=1))  # size (E, C_out)
        else:
            assert self.in_edgedim is None
            x_j = self.mlp(x_j)  # size (E, C_out)

        # use edge gates
        if self.edge_gate is not None:
            eg = self.edge_gate(x, edge_index, edge_attr=edge_attr, edge_weight=edge_weight)
            x_j = eg * x_j

        # aggregate the features into nodes, resulting size (N, C_out)
        # x_o = scatter_(self.aggr, x_j, edge_index[1], dim_size=x.size(0))
        x = scatter_(self.aggr, x_j, edge_index[1], dim_size=x.size(0))

        return x


class EdgeGateProj(nn.Module):
    """
        GRU标准门结构中输出门，表达式为:
            r = σ(Wa + Uh)
        其中，
            a为边特征，这里使用的信息双向传递方式，因此Wa = W_in a_in + W_out a_out
            h为邻域节点特征
    """

    def __init__(self, in_channels, in_edgedim=None, bias=False):
        super(EdgeGateProj, self).__init__()

        self.in_channels = in_channels
        self.in_edgedim = in_edgedim

        self.linsrc = nn.Linear(in_channels, 1, bias=False)
        self.lintgt = nn.Linear(in_channels, 1, bias=False)
        if in_edgedim is not None:
            self.linedge = nn.Linear(in_edgedim, 1, bias=False)

        if bias:
            self.bias = Parameter(torch.Tensor(1))  # a scalar bias applied to all edges.
            # self.bias = Parameter(torch.Tensor(num_edges))    # could also have a different bias for each edge.
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self, initrange=0.1):
        nn.init.uniform_(self.linsrc.weight, -initrange, initrange)
        nn.init.uniform_(self.lintgt.weight, -initrange, initrange)
        if self.in_edgedim is not None:
            nn.init.uniform_(self.linedge.weight, -initrange, initrange)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x, edge_index, edge_attr=None, edge_weight=None):
        x_j = torch.index_select(x, 0, edge_index[0])  # source node features, size (E, C_in)
        x_i = torch.index_select(x, 0, edge_index[1])  # target node features, size (E, C_in)
        edge_gate = self.linsrc(x_j) + self.lintgt(x_i)  # size (E, 1)
        if edge_attr is not None:
            assert self.linedge is not None
            edge_gate += self.linedge(edge_attr)
        if self.bias is not None:
            edge_gate += self.bias.view(-1, 1)
        edge_gate = torch.sigmoid(edge_gate)
        return edge_gate


class EdgeGateFree(nn.Module):
    """
    Calculate gates for each edge in message passing.
    The gates are free parameters.
    Note:
        This will make the parameters depend on the number of edges, which will limit the model
        to work only on graphs with fixed number of edges.
    """

    def __init__(self, num_edges):
        super(EdgeGateFree, self).__init__()

        self.num_edges = num_edges

        self.edge_gates = Parameter(torch.Tensor(num_edges, 1))

        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.uniform_(self.edge_gates, 0, 1)
        nn.init.constant_(self.edge_gates, 1)

    def forward(self, *args, **kwargs):  # *args and **kwargs to have the same argument API as the other class
        return torch.sigmoid(self.edge_gates)  # size (E, 1)
