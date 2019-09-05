"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
            sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable

import onmt
import onmt.io

# TGT_VOCAB_SIZE = 638  # TODO this should not be hard coded
TGT_VOCAB_SIZE = 117  # TODO this should not be hard coded
class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating mutiple
    loss computations


    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
            module that maps the output of the decoder to a
            distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
            torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """
    def __init__(self, generator, tgt_vocab):
        super(LossComputeBase, self).__init__()
        self.generator = generator
        self.tgt_vocab = tgt_vocab
        self.padding_idx = tgt_vocab.stoi[onmt.io.PAD_WORD]

    def _make_shard_state(self, batch, output, range_, attns=None, node_logits=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def monolithic_compute_loss(
            self, batch, output, attns,
            stage1=True,
            node_logits=None,
        ):
        """
        Compute the forward loss for the batch.

        Args:
            batch (batch): batch of labeled examples
            output (:obj:`FloatTensor`):
                output of decoder model `[tgt_len x batch x hidden]`
            attns (dict of :obj:`FloatTensor`) :
                dictionary of attention distributions
                `[tgt_len x batch x src_len]`
            stage1: is it stage1
        Returns:
            :obj:`onmt.Statistics`: loss statistics
        """
        if stage1:
            range_ = (0, batch.tgt1.size(0))
        else:
            range_ = (0, batch.tgt2.size(0))
        shard_state = self._make_shard_state(batch, output, range_, attns=attns, node_logits=node_logits)
        _, batch_stats = self._compute_loss(batch, **shard_state)

        return batch_stats

    def nonsharded_compute_loss(
            self, batch, output, attns,
            cur_trunc, trunc_size,
            normalization,
            stage1=True,
            retain_graph=False,
            node_logits=None,
        ):
        """
        Compute the forward loss for the batch.

        Args:
            batch (batch): batch of labeled examples
            output (:obj:`FloatTensor`):
                output of decoder model `[tgt_len x batch x hidden]`
            attns (dict of :obj:`FloatTensor`) :
                dictionary of attention distributions
                `[tgt_len x batch x src_len]`
            stage1: is it stage1
        Returns:
            :obj:`onmt.Statistics`: loss statistics
        """
        assert stage1
        range_ = (cur_trunc, cur_trunc + trunc_size)
        shard_state = self._make_shard_state(batch, output, range_, attns=attns, node_logits=node_logits)

        loss, batch_stats = self._compute_loss(batch, **shard_state)
        loss.div(normalization).backward(retain_graph=retain_graph)

        return batch_stats


    def sharded_compute_loss(
            self, batch, output, attns,
            cur_trunc, trunc_size, shard_size,
            normalization,
            retain_graph=False,
            node_logits=None
        ):
        """Compute the forward loss and backpropagate.  Computation is done
        with shards and optionally truncation for memory efficiency.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(cur_trunc, cur_trunc + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
            batch (batch) : batch of labeled examples
            output (:obj:`FloatTensor`) :
                output of decoder model `[tgt_len x batch x hidden]`
            attns (dict) : dictionary of attention distributions
                `[tgt_len x batch x src_len]`
            cur_trunc (int) : starting position of truncation window
            trunc_size (int) : length of truncation window
            shard_size (int) : maximum number of examples in a shard
            normalization (int) : Loss is divided by this number

        Returns:
            :obj:`onmt.Statistics`: validation loss statistics

        """
        batch_stats = onmt.Statistics()
        if node_logits is not None:
            batch_stats = (onmt.Statistics(), onmt.Statistics())

        range_ = (cur_trunc, cur_trunc + trunc_size)
        shard_state = self._make_shard_state(batch, output, range_, attns=attns, node_logits=node_logits)

        for shard in shards(shard_state, shard_size, retain_graph=retain_graph):
            loss, stats = self._compute_loss(batch, **shard)

            loss.div(normalization).backward()
            if isinstance(stats, tuple):
                batch_stats[0].update(stats[0])
                batch_stats[1].update(stats[1])
            else:
                batch_stats.update(stats)

        return batch_stats

    def _stats(self, loss, scores, target, binary=False):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`Statistics` : statistics for this batch.
        """
        if binary:
            pred = scores.lt(0.5).type(target.dtype)
        else:
            pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target).masked_select(non_padding).sum().item()
        if binary:
            positives = target.eq(1)
            n_tp = pred.eq(target).masked_select(positives).sum().item()
            n_fn = pred.ne(target).masked_select(positives).sum().item()
            return onmt.Statistics(loss.item(), non_padding.sum().item(), num_correct, n_tp=n_tp, n_fn=n_fn)
        else:
            return onmt.Statistics(loss.item(), non_padding.sum().item(), num_correct)

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def _unbottle(self, v, batch_size):
        return v.view(-1, batch_size, v.size(1))


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """
    def __init__(
        self, generator, tgt_vocab, normalization="sents",
        label_smoothing=0.0, decoder_type='rnn', cs_loss=False
        ):
        super(NMTLossCompute, self).__init__(generator, tgt_vocab)
        assert (label_smoothing >= 0.0 and label_smoothing <= 1.0)
        self.decoder_type = decoder_type
        self.cs_loss = cs_loss
        self.tgt_vocab = tgt_vocab
        if label_smoothing > 0:
            # When label smoothing is turned on,
            # KL-divergence between q_{smoothed ground truth prob.}(w)
            # and p_{prob. computed by model}(w) is minimized.
            # If label smoothing value is set to zero, the loss
            # is equivalent to NLLLoss or CrossEntropyLoss.
            # All non-true labels are uniformly set to low-confidence.
            self.criterion = nn.KLDivLoss(size_average=False)
            one_hot = torch.randn(1, len(tgt_vocab))
            one_hot.fill_(label_smoothing / (len(tgt_vocab) - 2))
            one_hot[0][self.padding_idx] = 0
            self.register_buffer('one_hot', one_hot)
        else:
            if self.decoder_type == 'pointer':
                weight = torch.ones(TGT_VOCAB_SIZE)
                if self.cs_loss:
                    self.cs_loss_criterion = nn.BCEWithLogitsLoss(
                        size_average=False)  # the losses are summed for each minibatch
            else:
                weight = torch.ones(len(tgt_vocab))
            weight[self.padding_idx] = 0
            self.criterion = nn.NLLLoss(weight, size_average=False)
        self.confidence = 1.0 - label_smoothing

    def _make_shard_state(self, batch, output, range_, attns=None, node_logits=None):
        assert attns is not None
        assert self.decoder_type == 'pointer'
        if self.cs_loss:
            assert node_logits is not None
        return {
            "output": attns['std'],
            #! NOTE: range_ is 0 to target_size for tgt1_planning
            "target": batch.tgt1_planning[range_[0] + 1: range_[1]],
            "node_logits": node_logits.unsqueeze(0) if self.cs_loss else None
        }

    def _compute_loss(self, batch, output, target, node_logits=None):

        if self.decoder_type == 'pointer':
            scores = self._bottle(output)  # [tgt_len, batch, src_len] --> [tgt_len*batch, src_len]
            if self.cs_loss:
                node_logits = node_logits.squeeze(0).squeeze(-1)
                cs_gtruth = torch.zeros_like(node_logits, requires_grad=False)
                # reference: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
                cs_gtruth.scatter_(dim=0, index=target, value=1)
                # node_logits = self._bottle(node_logits)
                # cs_gtruth = self._bottle(cs_gtruth)
        else:
            scores = self.generator(self._bottle(output))

        gtruth = target.view(-1)  # [tgt_len, batch] --> [tgt_len*batch]
        if self.confidence < 1:
            assert False
            tdata = gtruth.data
            mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze()
            log_likelihood = torch.gather(scores.data, 1, tdata.unsqueeze(1))
            tmp_ = self.one_hot.repeat(gtruth.size(0), 1)
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
            if mask.dim() > 0:
                log_likelihood.index_fill_(0, mask, 0)
                tmp_.index_fill_(0, mask, 0)
            gtruth = Variable(tmp_, requires_grad=False)

        loss = self.criterion(scores, gtruth)
        # print("loss {} = {}".format(loss.shape, loss))
        '''
        if self.confidence < 1:
            # Default: report smoothed ppl.
            # loss_data = -log_likelihood.sum(0)
            loss_data = loss.data.clone()
        else:
        '''
        loss_data = loss.data.clone()
        stats = self._stats(loss_data, scores.data.clone(), gtruth.data.clone())
        if self.cs_loss:
            cs_gtruth[self.tgt_vocab.stoi[onmt.io.PAD_WORD], :] = 0  # masking <blank>
            cs_loss = self.cs_loss_criterion(node_logits, cs_gtruth)
            # print("cs_loss {} = {}".format(cs_loss.shape, cs_loss))
            loss += cs_loss
            loss_data = cs_loss.data.clone()
            cs_scores = torch.sigmoid(node_logits).data.clone()
            cs_stats = self._stats(loss_data, cs_scores, cs_gtruth.data.clone(), True)
            stats = (stats, cs_stats)
        return loss, stats


def filter_shard_state(state, shard_size=None):
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)

def shards(state, shard_size, eval=False, retain_graph=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
                those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval: If True, only yield the state, nothing else.
                Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))
        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split]) for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                    [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads, retain_graph=retain_graph)