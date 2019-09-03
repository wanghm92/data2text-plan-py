from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import onmt
from onmt.Utils import aeq
from torch_geometric.data import Data, Batch
# from torch_geometric.nn import GCNConv, GATConv


def rnn_factory(rnn_type, **kwargs):
    # Use pytorch version when available.
    no_pack_padded_seq = False
    if rnn_type == "SRU":
        # SRU doesn't support PackedSequence.
        no_pack_padded_seq = True
        rnn = onmt.modules.SRU(**kwargs)
    else:
        rnn = getattr(nn, rnn_type)(**kwargs)
    return rnn, no_pack_padded_seq


class EncoderBase(nn.Module):
    """
    Base encoder class. Specifies the interface used by different encoder types
    and required by :obj:`onmt.Models.NMTModel`.
    """
    def _check_args(self, input, edges=None, lengths=None):
        if isinstance(self, GraphEncoder) and edges is None:
            raise ValueError('edges cannot be None for GraphEncoder')
        s_len, n_batch, n_feats = input.size()
        if lengths is not None:
            n_batch_, = lengths.size()
            aeq(n_batch, n_batch_)

    def forward(self, src, lengths=None, encoder_state=None):
        """
        Args:
            src (:obj:`LongTensor`):
                padded sequences of sparse indices `[src_len x batch x nfeat]`
            lengths (:obj:`LongTensor`): length of each sequence `[batch]`
            encoder_state (rnn-class specific):
                initial encoder_state state.

        Returns:
            (tuple of :obj:`FloatTensor`, :obj:`FloatTensor`):
                * final encoder state, used to initialize decoder
                * memory bank for attention, `[src_len x batch x hidden]`
        """
        raise NotImplementedError


class MeanEncoder(EncoderBase):
    """A trivial non-recurrent encoder. Simply applies mean pooling.

    Args:
        num_layers (int): number of replicated layers
        embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """
    def __init__(
            self, num_layers, src_bundle, emb_size,
            dropout=0.0, no_self_attn=False, attn_hidden=0, attn_type="general", coverage_attn=False,
            output_layer='gating', cs_loss=False
        ):
        super(MeanEncoder, self).__init__()
        self.num_layers = num_layers
        self.table_embeddings = None
        assert isinstance(src_bundle, tuple)
        embeddings, table_embeddings, _, _ = src_bundle
        self.table_embeddings = table_embeddings
        self.embeddings = embeddings
        self.dropout = nn.Dropout(p=dropout)
        self.no_self_attn = no_self_attn
        self.relu = nn.ReLU()
        self.cs_loss = cs_loss
        if self.cs_loss:
            self.hidden2logits = nn.Linear(emb_size, 1)

        self.output_layer = output_layer
        assert self.output_layer == 'gating'
        self.linear_out = nn.Linear(emb_size * 2, emb_size)  #, bias=False)
        if not self.no_self_attn:
            self.attn = onmt.modules.GlobalSelfAttention(
                emb_size, coverage=coverage_attn, attn_type=attn_type, attn_hidden=attn_hidden)

    def _get_outputs(self, emb, attn_vectors):
        concat_c = torch.cat([attn_vectors, emb], 2)
        out = self.linear_out(concat_c)
        out = torch.sigmoid(out).mul(emb)
        return out

    def forward(self, src, lengths=None, encoder_state=None, memory_lengths=None):
        assert isinstance(src, tuple)
        src, _ = src
        self._check_args(src, lengths=lengths)

        emb = self.dropout(self.embeddings(src))  # src: word/feature ids
        tbl_emb = None if self.table_embeddings is None else self.table_embeddings(src)
        s_len, batch, emb_dim = emb.size()

        if self.no_self_attn:
            encoder_output = emb
        else:
            attn_vectors, p_attn = self.attn(emb.transpose(0, 1).contiguous(), emb.transpose(0, 1), memory_lengths=lengths)
            encoder_output = self._get_outputs(emb, attn_vectors)

        node_logits = None
        if self.cs_loss:
            node_logits = self.hidden2logits(encoder_output)

        mean = encoder_output.mean(0).expand(self.num_layers, batch, emb_dim)
        memory_bank = encoder_output
        encoder_final = (mean, mean)
        return encoder_final, (memory_bank, tbl_emb, node_logits)


class GraphEncoder(EncoderBase):

    def __init__(
        self, num_layers, src_bundle, emb_size,
        dropout=0.0, no_self_attn=False, attn_hidden=0, attn_type="general", coverage_attn=False,
        output_layer='add', encoder_graph_fuse = 'highway',
        edge_aware='linear', edge_aggr='mean', edge_nei_fuse='uni', num_edge_types=-1,
        cs_loss=False
        ):
        super(GraphEncoder, self).__init__()
        self.num_layers = num_layers
        assert isinstance(src_bundle, tuple)
        embeddings, table_embeddings, edge_embeddings, graph_embeddings = src_bundle
        self.table_embeddings = table_embeddings
        self.edge_embeddings = edge_embeddings
        self.graph_embeddings = graph_embeddings
        self.edge_aggr = edge_aggr
        self.embeddings = embeddings
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.cs_loss = cs_loss
        if self.cs_loss:
            self.hidden2logits = nn.Linear(emb_size, 1)

        self.conv = onmt.modules.GatedGCN(emb_size, emb_size,
                                            edge_aware=edge_aware,
                                            edge_aggr=edge_aggr,
                                            num_edge_types=num_edge_types,
                                            edge_nei_fuse=edge_nei_fuse)

        print(' ** [GraphEncoder] encoder_graph_fuse = {}'.format(encoder_graph_fuse))
        self.encoder_graph_fuse = encoder_graph_fuse
        if self.encoder_graph_fuse == 'highway':
            self.graph_highway = onmt.modules.HighwayMLP(emb_size)
        elif self.encoder_graph_fuse == 'dense':
            self.graph_linear = nn.Linear(emb_size * 2, emb_size)
        elif not self.encoder_graph_fuse in ['nothing', 'add']:
            raise ValueError('{} is not supported'.format(self.encoder_graph_fuse))

        print(' ** [GraphEncoder] no_self_attn = {}'.format(no_self_attn))
        self.no_self_attn = no_self_attn
        if not self.no_self_attn:
            self.attn = onmt.modules.GlobalSelfAttention(
                emb_size, coverage=coverage_attn, attn_type=attn_type, attn_hidden=attn_hidden)
            self.linear_global = nn.Linear(emb_size * 2, emb_size, bias=False)
        else:
            if self.encoder_graph_fuse == 'nothing':
                raise ValueError("no_self_attn is {} and encoder_graph_fuse cannot be {}".format(self.no_self_attn, self.encoder_graph_fuse))

        print(' ** [GraphEncoder] output_layer = {}'.format(output_layer))
        self.output_layer = output_layer
        if not self.no_self_attn:
            if 'highway' in self.output_layer:
                self.out_highway = onmt.modules.HighwayMLP(emb_size)
            elif self.output_layer != 'add':
                raise ValueError('{} is not supported'.format(self.output_layer))


    def _node_encoding(self, graph_batch, shape, non_linear=False):
        out = self.conv(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr, graph_batch.edge_label, graph_batch.edge_norm)
        if non_linear:
            out = self.dropout(F.elu(out))
        out = out.reshape(shape)
        return out

    def fuse_them_all(self, emb, graph_emb, self_attn_vectors, graph_vectors):

        if self.encoder_graph_fuse == 'highway':
            neighbour_node_fuse = self.graph_highway(graph_emb, graph_vectors)
        elif self.encoder_graph_fuse == 'dense':
            neighbour_node_fuse = self.graph_linear(torch.cat([graph_emb, graph_vectors], 2))
        elif self.encoder_graph_fuse == 'nothing':
            neighbour_node_fuse = graph_vectors
        elif self.encoder_graph_fuse == 'add':
            neighbour_node_fuse = graph_vectors + graph_emb

        if self_attn_vectors is not None:
            global_context = self.linear_global(torch.cat([emb, self_attn_vectors], 2))
            global_gated_out = torch.sigmoid(global_context).mul(emb)
            if self.output_layer == 'highway-graph':
                out = self.out_highway(global_gated_out, graph_vectors)
            elif self.output_layer == 'highway-fuse':
                out = self.out_highway(global_gated_out, neighbour_node_fuse)
            elif self.output_layer == 'add':
                out = global_gated_out + neighbour_node_fuse
        else:
            out = neighbour_node_fuse

        return out

    def _construct_data_list(self, edges, emb):
        edge_left, edge_right, edge_norms, edge_label, num_edge = edges
        edge_embed = self.dropout(self.edge_embeddings(edge_label.unsqueeze(-1)))
        data_list = []
        for left, right, norm, edge_attr, label, length, x in \
                zip(torch.split(edge_left, 1, dim=1),
                    torch.split(edge_right, 1, dim=1),
                    torch.split(edge_norms.detach(), 1, dim=1),
                    torch.split(edge_embed, 1, dim=1),
                    torch.split(edge_label, 1, dim=1),
                    torch.split(num_edge, 1),
                    torch.split(emb, 1, dim=1)):
            edge_index = torch.cat([left, right], dim=1).t().contiguous()[:, :length.item()]
            edge_attr = edge_attr.squeeze(1)[:length.item(), :]  #! NOTE cut off by actual number of edges
            norm = norm[:length.item(), :]
            label = label[:length.item()]-2  # <unk>, <blank>
            x = x.squeeze(1)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_norm=norm, edge_label=label)
            data_list.append(data)
        return data_list

    def forward(self, src, lengths=None, encoder_state=None, memory_lengths=None):
        assert isinstance(src, tuple)
        src, edges = src
        self._check_args(src, edges=edges, lengths=lengths)

        # (1) original record embeddings
        emb = self.dropout(self.embeddings(src))
        s_len, batch_size, emb_dim = emb.size()

        # (2) gloabl self-attention record encodings
        self_attn_vectors = None
        if not self.no_self_attn:
            self_attn_vectors, _ = self.attn(emb.transpose(0, 1).contiguous(), emb.transpose(0, 1), memory_lengths=lengths)

        # (3) local graph constrained node encodings
        graph_emb = emb if self.graph_embeddings is None else self.graph_embeddings(src)
        data_list = self._construct_data_list(edges, graph_emb)
        graph_batch = Batch.from_data_list(data_list)
        graph_vectors = self._node_encoding(graph_batch, graph_emb.size())

        encoder_output = self.fuse_them_all(emb, graph_emb, self_attn_vectors, graph_vectors)

        node_logits = None
        if self.cs_loss:
            node_logits = self.hidden2logits(encoder_output)

        mean = encoder_output.mean(0).expand(self.num_layers, batch_size, emb_dim)
        memory_bank = encoder_output
        encoder_final = (mean, mean)

        tbl_emb = None if self.table_embeddings is None else self.table_embeddings(src)
        return encoder_final, (memory_bank, tbl_emb, node_logits)


class RNNEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
        rnn_type (:obj:`str`):
            style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
        bidirectional (bool) : use a bidirectional RNN
        num_layers (int) : number of stacked layers
        hidden_size (int) : hidden size of each layer
        dropout (float) : dropout value for :obj:`nn.Dropout`
        embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """
    def __init__(
        self, rnn_type, bidirectional, num_layers,
        hidden_size, dropout=0.0, embeddings=None,
        use_bridge=False):
        super(RNNEncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings

        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)

        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(rnn_type,
                                    hidden_size,
                                    num_layers)

    def forward(self, src, lengths=None, encoder_state=None):
        "See :obj:`EncoderBase.forward()`"
        assert isinstance(src, tuple)
        src, _ = src  #! _ is edges
        self._check_args(src, lengths=lengths)

        emb = src
        s_len, batch, emb_dim = emb.size()

        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            lengths = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths)

        memory_bank, encoder_final = self.rnn(packed_emb, encoder_state)

        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]

        if self.use_bridge:
            encoder_final = self._bridge(encoder_final)
        return encoder_final, memory_bank

    def _initialize_bridge(self, rnn_type, hidden_size, num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList(
            [nn.Linear(self.total_hidden_dim, self.total_hidden_dim, bias=True) for i in range(number_of_states)]
            )

    def _bridge(self, hidden):
        """
        Forward hidden state through bridge
        """
        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix]) for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs


class RNNDecoderBase(nn.Module):
    """
    Base recurrent attention-based decoder class.
    Specifies the interface used by different decoder types
    and required by :obj:`onmt.Models.NMTModel`.


    .. mermaid::

        graph BT
            A[Input]
            subgraph RNN
                C[Pos 1]
                D[Pos 2]
                E[Pos N]
            end
            G[Decoder State]
            H[Decoder State]
            I[Outputs]
            F[Memory_Bank]
            A--emb-->C
            A--emb-->D
            A--emb-->E
            H-->C
            C-- attn --- F
            D-- attn --- F
            E-- attn --- F
            C-->I
            D-->I
            E-->I
            E-->G
            F---I

    Args:
        rnn_type (:obj:`str`):
            style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
        bidirectional_encoder (bool) : use with a bidirectional encoder
        num_layers (int) : number of stacked layers
        hidden_size (int) : hidden size of each layer
        attn_type (str) : see :obj:`onmt.modules.GlobalAttention`
        coverage_attn (str): see :obj:`onmt.modules.GlobalAttention`
        context_gate (str): see :obj:`onmt.modules.ContextGate`
        copy_attn (bool): setup a separate copy attention mechanism
        dropout (float) : dropout value for :obj:`nn.Dropout`
        embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """
    def __init__(
        self, rnn_type, bidirectional_encoder, num_layers,
        hidden_size, attn_type="general",
        coverage_attn=False, context_gate=None,
        copy_attn=False, dropout=0.0, embeddings=None,
        reuse_copy_attn=False, pointer_decoder_type = None
        ):
        super(RNNDecoderBase, self).__init__()

        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)

        # Build the RNN.
        self.rnn = self._build_rnn(
            rnn_type,
            input_size=self._input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout)

        # Set up the context gate.
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = onmt.modules.context_gate_factory(
                context_gate, self._input_size,
                hidden_size, hidden_size, hidden_size
            )

        # Set up the standard attention.
        self._coverage = coverage_attn
        self.attn = onmt.modules.GlobalAttention(
            hidden_size, coverage=coverage_attn,
            attn_type=attn_type
        )
        self.table_attn = None
        if pointer_decoder_type == 'pointer':
            self.attn = onmt.modules.PointerAttention(hidden_size)
        else:
            # stage 2
            self.table_attn = onmt.modules.GlobalAttention(
                hidden_size, coverage=coverage_attn,
                attn_type=attn_type
            )

        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn and not reuse_copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                hidden_size, attn_type=attn_type
            )
        if copy_attn:
            self._copy = True
        self._reuse_copy_attn = reuse_copy_attn

    def forward(self, tgt, memory_bank, state, memory_lengths=None):
        """
        Args:
            tgt (`LongTensor`): sequences of padded tokens
                                `[tgt_len x batch x nfeats]`.
            memory_bank (`FloatTensor`): vectors from the encoder
                    `[src_len x batch x hidden]`.
            state (:obj:`onmt.Models.DecoderState`):
                    decoder state object to initialize the decoder
            memory_lengths (`LongTensor`): the padded source lengths
                `[batch]`.
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * decoder_outputs: output from the decoder (after attn)
                            `[tgt_len x batch x hidden]`.
                * decoder_state: final hidden state from the decoder
                * attns: distribution over src at each tgt
                        `[tgt_len x batch x src_len]`.
        """
        assert isinstance(memory_bank, tuple)
        memory_bank, trimmed_tbl_embs = memory_bank

        assert isinstance(state, RNNDecoderState)
        tgt_len, tgt_batch, _ = tgt.size()
        _, memory_batch, _ = memory_bank.size()
        aeq(tgt_batch, memory_batch)
        # END

        # Run the forward pass of the RNN.
        decoder_final, decoder_outputs, attns = self._run_forward_pass(
            tgt, (memory_bank, trimmed_tbl_embs), state, memory_lengths=memory_lengths)

        if decoder_outputs is None:
            final_output = None
        else:
            # Update the state with the result.
            final_output = decoder_outputs[-1].unsqueeze(0)

        coverage = None
        state.update_state(decoder_final, final_output, coverage)
        if decoder_outputs is not None:
            # Concatenates sequence of tensors along a new dimension.
            decoder_outputs = torch.stack(decoder_outputs)

        for k in attns:
            if type(attns[k]) == list:
                attns[k] = torch.stack(attns[k])

        return decoder_outputs, state, attns

    def init_decoder_state(self, src, memory_bank, encoder_final):
        def _fix_enc_hidden(h):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder:
                h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
            return h

        if isinstance(encoder_final, tuple):  # LSTM
            return RNNDecoderState(self.hidden_size,
                                    tuple([_fix_enc_hidden(enc_hid)
                                            for enc_hid in encoder_final]))
        else:  # GRU
            return RNNDecoderState(self.hidden_size,
                                    _fix_enc_hidden(encoder_final))

class PointerRNNDecoder(RNNDecoderBase):
    """
    Standard fully batched RNN decoder with attention.
    Faster implementation, uses CuDNN for implementation.
    See :obj:`RNNDecoderBase` for options.


    Based around the approach from
    "Neural Machine Translation By Jointly Learning To Align and Translate"
    :cite:`Bahdanau2015`


    Implemented without input_feeding and currently with no `coverage_attn`
    or `copy_attn` support.
    """
    def _run_forward_pass(self, tgt, memory_bank, state, memory_lengths=None):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.
        Args:
            tgt (LongTensor): a sequence of input tokens tensors
                                    [len x batch x nfeats].
            memory_bank (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                    initializing the decoder.
            memory_lengths (LongTensor): the source memory_bank lengths.
        Returns:
            decoder_final (Variable): final hidden state from the decoder.
            decoder_outputs ([FloatTensor]): an array of output of every time
                                        step from the decoder.
            attns (dict of (str, [FloatTensor]): a dictionary of different
                            type of attention Tensor array of every time
                            step from the decoder.
        """
        assert isinstance(memory_bank, tuple)
        memory_bank, _ = memory_bank  #! _ is table embeddings

        assert not self._copy  # TODO, no support yet.
        assert not self._coverage  # TODO, no support yet.
        # Initialize local and return variables.
        attns = {}
        emb = torch.transpose(torch.cat(
            [torch.index_select(a, 0, i).unsqueeze(0) for a, i in
                zip(torch.transpose(memory_bank, 0, 1), torch.t(torch.squeeze(tgt,2)))]), 0, 1)
        # Run the forward pass of the RNN.
        if isinstance(self.rnn, nn.GRU):
            rnn_output, decoder_final = self.rnn(emb, state.hidden[0])
        else:
            rnn_output, decoder_final = self.rnn(emb, state.hidden)

        # Check
        tgt_len, tgt_batch, _ = tgt.size()
        output_len, output_batch, _ = rnn_output.size()
        aeq(tgt_len, output_len)
        aeq(tgt_batch, output_batch)
        # END

        # Calculate the attention.
        p_attn = self.attn(
            rnn_output.transpose(0, 1).contiguous(),
            memory_bank.transpose(0, 1),
            memory_lengths=memory_lengths
        )
        attns["std"] = p_attn


        #decoder_outputs = self.dropout(decoder_outputs)
        return decoder_final, None, attns

    def _build_rnn(self, rnn_type, **kwargs):
        rnn, _ = rnn_factory(rnn_type, **kwargs)
        return rnn

    @property
    def _input_size(self):
        """
        Private helper returning the number of expected features.
        """
        return self.embeddings.embedding_size


class StdRNNDecoder(RNNDecoderBase):
    """
    Standard fully batched RNN decoder with attention.
    Faster implementation, uses CuDNN for implementation.
    See :obj:`RNNDecoderBase` for options.


    Based around the approach from
    "Neural Machine Translation By Jointly Learning To Align and Translate"
    :cite:`Bahdanau2015`


    Implemented without input_feeding and currently with no `coverage_attn`
    or `copy_attn` support.
    """
    def _run_forward_pass(self, tgt, memory_bank, state, memory_lengths=None):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.
        Args:
            tgt (LongTensor): a sequence of input tokens tensors
                                    [len x batch x nfeats].
            memory_bank (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                    initializing the decoder.
            memory_lengths (LongTensor): the source memory_bank lengths.
        Returns:
            decoder_final (Variable): final hidden state from the decoder.
            decoder_outputs ([FloatTensor]): an array of output of every time
                                        step from the decoder.
            attns (dict of (str, [FloatTensor]): a dictionary of different
                            type of attention Tensor array of every time
                            step from the decoder.
        """
        if isinstance(memory_bank, tuple):
            memory_bank, _ = memory_bank

        assert not self._copy  # TODO, no support yet.
        assert not self._coverage  # TODO, no support yet.

        # Initialize local and return variables.
        attns = {}
        emb = self.embeddings(tgt)

        # Run the forward pass of the RNN.
        if isinstance(self.rnn, nn.GRU):
            rnn_output, decoder_final = self.rnn(emb, state.hidden[0])
        else:
            rnn_output, decoder_final = self.rnn(emb, state.hidden)

        # Check
        tgt_len, tgt_batch, _ = tgt.size()
        output_len, output_batch, _ = rnn_output.size()
        aeq(tgt_len, output_len)
        aeq(tgt_batch, output_batch)
        # END

        # Calculate the attention.
        decoder_outputs, p_attn = self.attn(
            rnn_output.transpose(0, 1).contiguous(),
            memory_bank.transpose(0, 1),
            memory_lengths=memory_lengths
        )
        attns["std"] = p_attn

        # Calculate the context gate.
        if self.context_gate is not None:
            decoder_outputs = self.context_gate(
                emb.view(-1, emb.size(2)),
                rnn_output.view(-1, rnn_output.size(2)),
                decoder_outputs.view(-1, decoder_outputs.size(2))
            )
            decoder_outputs = \
                decoder_outputs.view(tgt_len, tgt_batch, self.hidden_size)

        decoder_outputs = self.dropout(decoder_outputs)
        return decoder_final, decoder_outputs, attns

    def _build_rnn(self, rnn_type, **kwargs):
        rnn, _ = rnn_factory(rnn_type, **kwargs)
        return rnn

    @property
    def _input_size(self):
        """
        Private helper returning the number of expected features.
        """
        return self.embeddings.embedding_size


class InputFeedRNNDecoder(RNNDecoderBase):
    """
    Input feeding based decoder. See :obj:`RNNDecoderBase` for options.

    Based around the input feeding approach from
    "Effective Approaches to Attention-based Neural Machine Translation"
    :cite:`Luong2015`


    .. mermaid::

    graph BT
        A[Input n-1]
        AB[Input n]
        subgraph RNN
            E[Pos n-1]
            F[Pos n]
            E --> F
        end
        G[Encoder]
        H[Memory_Bank n-1]
        A --> E
        AB --> F
        E --> H
        G --> H
    """

    def _run_forward_pass(self, tgt, memory_bank, state, memory_lengths=None):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        assert isinstance(memory_bank, tuple)
        memory_bank, trimmed_tbl_embs = memory_bank

        # Additional args check.
        input_feed = state.input_feed.squeeze(0)
        input_feed_batch, _ = input_feed.size()
        tgt_len, tgt_batch, _ = tgt.size()
        aeq(tgt_batch, input_feed_batch)
        # END Additional args check.

        # Initialize local and return variables.
        decoder_outputs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []
        if trimmed_tbl_embs is not None:
            attns["table"] = []

        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        hidden = state.hidden
        coverage = state.coverage.squeeze(0) \
            if state.coverage is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.
        for i, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            decoder_input = torch.cat([emb_t, input_feed], 1)

            rnn_output, hidden = self.rnn(decoder_input, hidden)
            decoder_output, p_attn = self.attn(
                rnn_output,
                memory_bank.transpose(0, 1),
                memory_lengths=memory_lengths)

            if trimmed_tbl_embs is not None:
                _, table_attn = self.table_attn(
                    rnn_output,
                    trimmed_tbl_embs.transpose(0, 1),
                    memory_lengths=memory_lengths)
                attns["table"] += [table_attn]

            # print("[InputFeedRNNDecoder] self.context_gate = {}".format(self.context_gate))
            if self.context_gate is not None:
                # TODO: context gate should be employed
                # instead of second RNN transform.
                decoder_output = self.context_gate(decoder_input, rnn_output, decoder_output)

            decoder_output = self.dropout(decoder_output)
            input_feed = decoder_output

            decoder_outputs += [decoder_output]
            attns["std"] += [p_attn]

            # Update the coverage attention.
            if self._coverage:
                coverage = coverage + p_attn \
                    if coverage is not None else p_attn
                attns["coverage"] += [coverage]

            # Run the forward pass of the copy attention layer.
            if self._copy and not self._reuse_copy_attn:
                _, copy_attn = self.copy_attn(decoder_output, memory_bank.transpose(0, 1))
                attns["copy"] += [copy_attn]
            elif self._copy:
                attns["copy"] = attns["std"]
        # Return result.
        return hidden, decoder_outputs, attns

    def _build_rnn(self, rnn_type, input_size, hidden_size, num_layers, dropout):
        assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
                "Please set -input_feed 0!"
        if rnn_type == "LSTM":
            stacked_cell = onmt.modules.StackedLSTM
        else:
            stacked_cell = onmt.modules.StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embeddings.embedding_size + self.hidden_size


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
        encoder (:obj:`EncoderBase`): an encoder object
        decoder (:obj:`RNNDecoderBase`): a decoder object
        multigpu (bool): setup for multigpu support
    """
    def __init__(self, encoder, decoder, multigpu=False):
        self.multigpu = multigpu
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, dec_state=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                    a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        #! NOTE: src is indices for stage1 and vectors for stage2

        tgt = tgt[:-1]  #! NOTE: exclude </s>

        assert isinstance(src, tuple)
        src, trimmed_tbl_embs, edges = src

        enc_final, memory_bank = self.encoder((src, edges), lengths)

        enc_embs = None
        node_logits = None
        if isinstance(memory_bank, tuple):
            #! stage1: Mean or GraphEncoder
            memory_bank, enc_embs, node_logits = memory_bank

        enc_state = self.decoder.init_decoder_state(src, memory_bank, enc_final)

        decoder_outputs, dec_state, attns = \
            self.decoder(
                tgt,
                (memory_bank, trimmed_tbl_embs),
                enc_state if dec_state is None else dec_state,
                memory_lengths=lengths
            )
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return decoder_outputs, attns, dec_state, (memory_bank, enc_embs, node_logits)


class DecoderState(object):
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.

    Modules need to implement this to utilize beam search decoding.
    """
    def detach(self):
        for h in self._all:
            if h is not None:
                h.detach()

    def beam_update(self, idx, positions, beam_size):
        for e in self._all:
            sizes = e.size()
            br = sizes[1]
            if len(sizes) == 3:
                sent_states = e.view(sizes[0], beam_size, br // beam_size, sizes[2])[:, :, idx]
            else:
                sent_states = e.view(sizes[0], beam_size,
                                        br // beam_size,
                                        sizes[2],
                                        sizes[3])[:, :, idx]

            sent_states.data.copy_(
                sent_states.data.index_select(1, positions))


class RNNDecoderState(DecoderState):
    def __init__(self, hidden_size, rnnstate):
        """
        Args:
            hidden_size (int): the size of hidden layer of the decoder.
            rnnstate: final hidden state from the encoder.
                transformed to shape: layers x batch x (directions*dim).
        """
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.coverage = None

        # Init the input feed.
        batch_size = self.hidden[0].size(1)
        h_size = (batch_size, hidden_size)
        self.input_feed = self.hidden[0].data.new(*h_size).zero_().unsqueeze(0)

    @property
    def _all(self):
        return self.hidden + (self.input_feed,)

    def update_state(self, rnnstate, input_feed, coverage):
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        if input_feed is not None:
            self.input_feed = input_feed
        self.coverage = coverage

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        vars = [Variable(e.data.repeat(1, beam_size, 1), requires_grad=False)
                for e in self._all]
        self.hidden = tuple(vars[:-1])
        self.input_feed = vars[-1]
