import torch
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

import onmt
from onmt.modules.UtilClass import glorot, zeros


class GatedGCN(MessagePassing):

    def __init__(
        self, in_channels, out_channels, edge_aware='add', edge_aggr='mean', **kwargs):
        super(GatedGCN, self).__init__(aggr=edge_aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.edge_node_fuse = nn.Linear(2 * in_channels, out_channels)

        print(' ** [GatedGCN] edge_aware = {}'.format(edge_aware))
        self.edge_aware = edge_aware
        if self.edge_aware == 'linear':
            self.edge_neighbour_fuse = nn.Linear(2 * in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, edge_index_i, x_i, x_j, edge_attr):

        x_j = x_j.view(-1, self.out_channels)
        x_i = x_i.view(-1, self.out_channels)

        #! combine node itself with the edge types
        node_edge_aware = self.edge_node_fuse(torch.cat([x_i, edge_attr], -1))
        #! compute a node-edge-aware elementwise gate to filter neighbour information
        gated_neighbours = torch.sigmoid(node_edge_aware).mul(x_j)

        #! add edge information back to message passed to target
        if self.edge_aware == 'add':
            out = gated_neighbours + edge_attr
        elif self.edge_aware == 'linear':
            concat_c = torch.cat([gated_neighbours, edge_attr], -1)
            out = self.edge_neighbour_fuse(concat_c)

        return out

    def update(self, aggr_out):
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(
        self, in_channels, out_channels, heads=1, concat=True, dropout=0,
        edge_aware='linear', attn_hidden=0, edge_attn_bias='scalar', **kwargs):
        super(GATConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        print(' ** [GAT] edge_aware = {}'.format(edge_aware))
        print(' ** [GAT] edge_attn_bias = {}'.format(edge_attn_bias))

        self.edge_aware = edge_aware
        if self.edge_aware == 'linear':
            self.edge_node_fuse = nn.Linear(2 * out_channels, out_channels)

        self.edge_attn_bias = edge_attn_bias
        if self.edge_attn_bias == 'weighted':
            self.reduce_edge = nn.Linear(out_channels, 1, bias=False)

        self.attn_hidden = attn_hidden
        #! Default: bias=True
        self.transform_in = nn.Sequential(nn.Linear(in_channels, attn_hidden), nn.ELU(0.1))
        self.linear_in = nn.Linear(attn_hidden, attn_hidden, bias=False)

    def forward(self, x, edge_index, edge_attr, edge_scalar, size=None):
        # if size is None and torch.is_tensor(x):
        #     edge_index, _ = remove_self_loops(edge_index)
        #     edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # if torch.is_tensor(x):
        #     x = torch.matmul(x, self.weight)
        # else:
        #     x = (None if x[0] is None else torch.matmul(x[0], self.weight),
        #             None if x[1] is None else torch.matmul(x[1], self.weight))

        return self.propagate(edge_index, size=size, x=x, edge_attr=edge_attr, edge_scalar=edge_scalar)


    def score(self, h_t, h_s, edge_attr, edge_scalar=None):
        """
        Args:
            h_t (`FloatTensor`): query tensor `[#edges x #heads x dim]`
            h_s (`FloatTensor`): source tensor (memory bank) `[#edges x #heads x dim]`
        Returns:
            raw attention scores (unnormalized) for each src index (memory bank)
            `[#edges x #heads x dim]

        """
        h_t = self.transform_in(h_t)
        h_s = self.transform_in(h_s)

        # attn_type == "general"
        h_t = self.linear_in(h_t)

        scores = torch.mul(h_t, h_s).sum(dim=-1)

        if self.edge_attn_bias == 'weighted':
            assert edge_scalar is None
            edge_bias = self.reduce_edge(edge_attr)
        else:
            edge_bias = edge_scalar

        scores = torch.add(scores, edge_bias.squeeze(-1))

        return scores


    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr, edge_scalar):
        # Compute attention coefficients.
        edge_attr = edge_attr.unsqueeze(1)
        if edge_scalar is not None:
            edge_scalar = edge_scalar.unsqueeze(1)

        # reshape the multi-headed Q and K (torch.matmul(x, self.weight) into #heads*dim
        # TODO: change here if multi-headed attn is needed
        x_j = x_j.view(-1, self.heads, self.out_channels)
        x_i = x_i.view(-1, self.heads, self.out_channels)

        #! x_j is index selected from the source nodes
        #! correspond to the memorybank in score(input, memorybabnk)
        alpha = self.score(x_i, x_j, edge_attr, edge_scalar=edge_scalar)
        alpha = softmax(alpha, edge_index_i, size_i)
        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # TODO: here could be multiple options for merging two sources of information
        if self.edge_aware == 'add':
            out = x_j + edge_attr
        elif self.edge_aware == 'linear':
            concat_c = torch.cat([x_j, edge_attr], 2)
            out = self.edge_node_fuse(concat_c)

        return out * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                                self.in_channels,
                                                self.out_channels, self.heads)
