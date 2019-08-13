# -*- coding: utf-8 -*-

from collections import Counter
from itertools import chain
import io
import codecs
import sys

import torch
import torchtext

from onmt.Utils import aeq
from onmt.io.BoxField import BoxField
from onmt.io.DatasetBase import (ONMTDatasetBase, UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD)
import jsonlines, os
from tqdm import tqdm
PAD_INDEX = 1
BOS_INDEX = 2
EOS_INDEX = 3

class TextDataset(ONMTDatasetBase):
    """ Dataset for data_type=='text'

        Build `Example` objects, `Field` objects, and filter_pred function
        from text corpus.

        Args:
            fields (dict): a dictionary of `torchtext.data.Field`.
                Keys are like 'src', 'tgt', 'src_map', and 'alignment'.
            src_examples_iter (dict iter): preprocessed source example
                dictionary iterator.
            tgt_examples_iter (dict iter): preprocessed target example
                dictionary iterator.
            num_src_feats (int): number of source side features.
            num_tgt_feats (int): number of target side features.
            src_seq_length (int): maximum source sequence length.
            tgt_seq_length (int): maximum target sequence length.
            dynamic_dict (bool): create dynamic dictionaries?
            use_filter_pred (bool): use a custom filter predicate to filter
                out examples?
    """
    def __init__(
        self, fields, src_examples_iter, tgt_examples_iter, src2_examples_iter, tgt2_examples_iter,
        num_src_feats=0, num_tgt_feats=0, num_src_feats2=0, num_tgt_feats2=0,
        src_seq_length=0, tgt_seq_length=0,
        dynamic_dict=True, use_filter_pred=True, pointers_file=None, edge_file=None
        ):
        self.data_type = 'text'

        # self.src_vocabs: mutated in dynamic_dict, used in
        # collapse_copy_scores and in Translator.py
        self.src_vocabs = []

        self.n_src_feats = num_src_feats
        self.n_tgt_feats = num_tgt_feats

        # Each element of an example is a dictionary whose keys represents
        # at minimum the src tokens and their indices and potentially also
        # the src and tgt features and alignment information.
        pointers = None
        if pointers_file is not None:
            with open(pointers_file) as f:
                content = f.readlines()
            pointers = [x.strip() for x in content]

        if edge_file is not None:
            with jsonlines.open(edge_file, 'r') as f:
                edges = [i for i in f.iter(type=dict, skip_invalid=True)]

        # dictionaries are joint together
        if tgt2_examples_iter is not None:
            examples_iter = (self._join_dicts(src, tgt, src2, tgt2) for src, tgt, src2, tgt2 in
                                zip(src_examples_iter, tgt_examples_iter, src2_examples_iter, tgt2_examples_iter))
        elif src2_examples_iter is not None:
            examples_iter = (self._join_dicts(src, tgt, src2) for src, tgt, src2 in
                                zip(src_examples_iter, tgt_examples_iter, src2_examples_iter))
        else:
            examples_iter = src_examples_iter

        if dynamic_dict and src2_examples_iter is not None:
            # additional field keys are added
            examples_iter = self._dynamic_dict(examples_iter, pointers=pointers, edges=edges)

        # Peek at the first to see which fields are used.
        ex, examples_iter = self._peek(examples_iter)
        keys = ex.keys()

        # pair with corresponding field objects
        out_fields = [(k, fields[k]) if k in fields else (k, None) for k in keys]

        # convert dict to lists
        example_values = ([ex[k] for k in keys] for ex in examples_iter)

        # If out_examples is a generator, we need to save the filter_pred
        # function in serialization too, which would cause a problem when
        # `torch.save()`. Thus we materialize it as a list.
        src_size = 0

        out_examples = []
        print("construct_example_fromlist ...")
        for ex_values in tqdm(example_values):
            example = self._construct_example_fromlist(
                ex_values, out_fields)
            src_size += len(example.src1)
            out_examples.append(example)

        print("average src size", src_size / len(out_examples), len(out_examples))

        def filter_pred(example):

            return 0 < len(example.src1) <= src_seq_length \
                and 0 < len(example.tgt1) <= tgt_seq_length \
                    and (pointers_file is None or 1 < example.ptrs.size(0))

        filter_pred = filter_pred if use_filter_pred else lambda x: True

        super(TextDataset, self).__init__(
            out_examples, out_fields, filter_pred
        )

    @staticmethod
    def collapse_copy_scores(scores, batch, tgt_vocab, src_vocabs):
        """
        Given scores from an expanded dictionary
        corresponeding to a batch, sums together copies,
        with a dictionary word when it is ambigious.
        """
        offset = len(tgt_vocab)
        for b in range(batch.batch_size):
            blank = []
            fill = []
            index = batch.indices.data[b]
            src_vocab = src_vocabs[index]
            for i in range(1, len(src_vocab)):
                sw = src_vocab.itos[i]
                ti = tgt_vocab.stoi[sw]
                if ti != 0:
                    blank.append(offset + i)
                    fill.append(ti)
            if blank:
                blank = torch.Tensor(blank).type_as(batch.indices.data)
                fill = torch.Tensor(fill).type_as(batch.indices.data)
                scores[:, b].index_add_(1, fill,
                                        scores[:, b].index_select(1, blank))
                scores[:, b].index_fill_(1, blank, 1e-10)
        return scores

    @staticmethod
    def make_text_examples_nfeats_tpl(path, truncate, side):
        """
        Args:
            path (str): location of a src or tgt file.
            truncate (int): maximum sequence length (0 for unlimited).
            side (str): "src" or "tgt".

        Returns:
            (example_dict iterator, num_feats) tuple.
        """
        assert side in ['src1', 'src2', 'tgt1', 'tgt2']

        if path is None:
            return (None, 0)

        # All examples have same number of features, so we peek first one
        # to get the num_feats.
        examples_nfeats_iter = \
            TextDataset.read_text_file(path, truncate, side)

        first_ex = next(examples_nfeats_iter)
        num_feats = first_ex[1]

        # Chain back the first element - we only want to peek it.
        examples_nfeats_iter = chain([first_ex], examples_nfeats_iter)
        examples_iter = (ex for ex, nfeats in examples_nfeats_iter)

        return (examples_iter, num_feats)

    @staticmethod
    def read_text_file(path, truncate, side):
        """
        Args:
            path (str): location of a src or tgt file.
            truncate (int): maximum sequence length (0 for unlimited).
            side (str): "src" or "tgt".

        Yields:
            (word, features, nfeat) triples for each line.
        """
        with codecs.open(path, "r", "utf-8") as corpus_file:
            for i, line in enumerate(corpus_file):
                line = line.strip().split()
                if truncate:
                    line = line[:truncate]

                words, feats, n_feats = \
                    TextDataset.extract_text_features(line)

                example_dict = {side: words, "indices": i}
                if side == 'tgt1':
                    example_dict = {side: words, 'tgt1_planning': [int(word) for word in words], "indices": i}
                if feats:
                    prefix = side + "_feat_"
                    example_dict.update((prefix + str(j), f)
                                        for j, f in enumerate(feats))
                yield example_dict, n_feats

    @staticmethod
    def get_fields(n_src_features, n_tgt_features):
        """
        Args:
            n_src_features (int): the number of source features to
                create `torchtext.data.Field` for.
            n_tgt_features (int): the number of target features to
                create `torchtext.data.Field` for.

        Returns:
            A dictionary whose keys are strings and whose values
            are the corresponding Field objects.
        """
        fields = {}

        #! NOTE: PAD_WORD is <blank>, different from the default <pad> used in torchtext/BoxField
        #! NOTE: sequential=False --> assumed to be the same size and no padding, here passed for building vocab
        fields["src1"] = BoxField(
            sequential=False,
            init_token=BOS_WORD,
            eos_token=EOS_WORD,
            pad_token=PAD_WORD)

        for j in range(n_src_features):
            fields["src1_feat_" + str(j)] = \
                BoxField(sequential=False, pad_token=PAD_WORD)

        #! NOTE: INIT/EOS are None --> no pre/appending and not included in vocab
        fields["edge_labels"] = BoxField(
            pad_token=PAD_WORD,
            include_lengths=True
        )
        # ! NOTE: PAD_INDEX for <blank>
        fields["edge_left"] = BoxField(
            use_vocab=False,
            pad_token=PAD_INDEX
        )
        # ! NOTE: PAD_INDEX for <blank>
        fields["edge_right"] = BoxField(
            use_vocab=False,
            pad_token=PAD_INDEX
        )

        #! NOTE: PAD_INDEX for <blank> BOS_INDEX for <s>, EOS_INDEX for </s> are the positions of the pre/appended to the planning sequence
        fields["tgt1_planning"] = BoxField(
            use_vocab=False,
            init_token=BOS_INDEX,
            eos_token=EOS_INDEX,
            pad_token=PAD_INDEX)

        fields["tgt1"] = torchtext.data.Field(
            init_token=BOS_WORD, eos_token=EOS_WORD,
            pad_token=PAD_WORD)

        for j in range(n_tgt_features):
            fields["tgt1_feat_"+str(j)] = \
                torchtext.data.Field(init_token=BOS_WORD, eos_token=EOS_WORD, pad_token=PAD_WORD)

        fields["src2"] = torchtext.data.Field(
            pad_token = PAD_WORD,
            include_lengths = True)

        for j in range(n_src_features):
            fields["src2_feat_" + str(j)] = \
                torchtext.data.Field(pad_token=PAD_WORD)

        fields["tgt2"] = torchtext.data.Field(
            init_token=BOS_WORD, eos_token=EOS_WORD,
            pad_token=PAD_WORD)

        def make_src(data, vocab, is_train):
            #! NOTE vocab is None since use_vocab=False
            src_size = max([t.size(0) for t in data])
            src_vocab_size = max([t.max() for t in data]) + 1
            alignment = torch.zeros(src_size, len(data), src_vocab_size)
            for i, sent in enumerate(data):
                for j, t in enumerate(sent):
                    alignment[j, i, t] = 1
            return alignment

        fields["src_map"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.FloatTensor,
            postprocessing=make_src, sequential=False)

        def make_tgt(data, vocab, is_train):
            #! NOTE vocab is None since use_vocab=False
            tgt_size = max([t.size(0) for t in data])
            alignment = torch.zeros(tgt_size, len(data)).long()
            for i, sent in enumerate(data):
                alignment[:sent.size(0), i] = sent
            return alignment

        fields["alignment"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.LongTensor,
            postprocessing=make_tgt, sequential=False)

        def make_pointer(data, vocab, is_train):
            #! NOTE vocab is None since use_vocab=False
            if is_train:
                src_size = max([t[-2][0] for t in data])
                tgt_size = max([t[-1][0] for t in data])
                #! NOTE: format of data is tgt_len, batch, src_len
                alignment = torch.zeros(tgt_size+2, len(data), src_size).long()  #! NOTE: +2 for bos and eos
                for i, sent in enumerate(data):
                    for j, t in enumerate(sent[:-2]):
                        #! NOTE: only iterate till the 3rd last row as the last 2 rows contains lengths of src and tgt
                        for k in range(1,t[t.size(0)-1]):   #! NOTE: iterate from index 1 as index 0 is tgt position
                            alignment[t[0]+1][i][t[k]] = 1  #! NOTE: +1 to accommodate bos
                return alignment
            else:
                return torch.zeros(50, 5, 602).long()

        fields["ptrs"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.LongTensor,
            postprocessing=make_pointer,sequential=False)

        fields["indices"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.LongTensor,
            sequential=False)

        return fields

    @staticmethod
    def get_num_features(corpus_file, side):
        """
        Peek one line and get number of features of it.
        (All lines must have same number of features).
        For text corpus, both sides are in text form, thus
        it works the same.

        Args:
            corpus_file (str): file path to get the features.
            side (str): 'src' or 'tgt'.

        Returns:
            number of features on `side`.
        """
        with codecs.open(corpus_file, "r", "utf-8") as cf:
            f_line = cf.readline().strip().split()
            _, _, num_feats = TextDataset.extract_text_features(f_line)

        return num_feats

    # Below are helper functions for intra-class use only.
    def _dynamic_dict(self, examples_iter, pointers=None, edges=None):
        loop_index = -1
        for example in examples_iter:
            src = example["src2"]  #! NOTE: ratish's content plan contains unique records
            loop_index += 1
            src_vocab = torchtext.vocab.Vocab(Counter(src), specials=[UNK_WORD, PAD_WORD])
            self.src_vocabs.append(src_vocab)
            # Mapping source tokens to indices in the dynamic dict.
            src_map = torch.LongTensor([src_vocab.stoi[w] for w in src])
            example["src_map"] = src_map
            if "tgt2" in example:
                tgt = example["tgt2"]
                mask = torch.LongTensor(
                    [0] + [src_vocab.stoi[w] for w in tgt] + [0])  # 0 for unk
                example["alignment"] = mask

                if pointers is not None and loop_index < len(pointers):
                    pointer_entries = pointers[loop_index].split()
                    pointer_entries = [int(entry.split(",")[0]) for entry in pointer_entries]
                    #! NOTE: overriding the mask above
                    mask = torch.LongTensor([0] + [src_vocab.stoi[w] if i in pointer_entries
                                                    else src_vocab.stoi[UNK_WORD] for i, w in enumerate(tgt)] + [0])
                    example["alignment"] = mask
                    max_len = 0
                    line_tuples = []
                    for pointer in pointers[loop_index].split():
                        val = [int(entry) for entry in pointer.split(",")]
                        if len(val)>max_len:
                            max_len = len(val)
                        line_tuples.append(val)
                    num_rows = len(line_tuples)+2   #+2 for storing the length of the source and target sentence
                    ptrs = torch.zeros(num_rows, max_len+1).long()  #last col is for storing the size of the row
                    for j in range(ptrs.size(0)-2): #iterating until row-1 as row contains the length of the sentence
                        for k in range(len(line_tuples[j])):
                            ptrs[j][k]=line_tuples[j][k]
                        ptrs[j][max_len] = len(line_tuples[j])
                    ptrs[ptrs.size(0)-2][0] = len(src)
                    ptrs[ptrs.size(0)-1][0] = len(tgt)
                    example["ptrs"] = ptrs
                else:
                    example["ptrs"] = None

                #! NOTE: edges are added here
                if edges is not None:
                    edge_left = []
                    edge_right = []
                    edge_labels = []
                    for k, v in edges[loop_index].items():
                        left, right = k.split(',')
                        edge_left.append(int(left))
                        edge_right.append(int(right))
                        edge_labels.append(v)
                    example["edge_left"] = torch.LongTensor(edge_left)
                    example["edge_right"] = torch.LongTensor(edge_right)
                    example["edge_labels"] = edge_labels
                else:
                    example["edge_left"] = None
                    example["edge_right"] = None
                    example["edge_labels"] = None

            yield example


class ShardedTextCorpusIterator(object):
    """
    This is the iterator for text corpus, used for sharding large text
    corpus into small shards, to avoid hogging memory.

    Inside this iterator, it automatically divides the corpus file into
    shards of size `shard_size`. Then, for each shard, it processes
    into (example_dict, n_features) tuples when iterates.
    """
    def __init__(self, corpus_path, line_truncate, side, shard_size, assoc_iter=None):
        """
        Args:
            corpus_path: the corpus file path.
            line_truncate: the maximum length of a line to read.
                            0 for unlimited.
            side: "src" or "tgt".
            shard_size: the shard size, 0 means not sharding the file.
            assoc_iter: if not None, it is the associate iterator that
                        this iterator should align its step with.
        """
        try:
            # The codecs module seems to have bugs with seek()/tell(),
            # so we use io.open().
            self.corpus = io.open(corpus_path, "r", encoding="utf-8")
        except IOError:
            sys.stderr.write("Failed to open corpus file: %s" % corpus_path)
            sys.exit(1)
        self.corpus_path = corpus_path
        self.line_truncate = line_truncate
        self.side = side
        self.shard_size = shard_size
        self.assoc_iter = assoc_iter
        self.last_pos = 0
        self.line_index = -1
        self.eof = False

    def __iter__(self):
        """
        Iterator of (example_dict, nfeats).
        On each call, it iterates over as many (example_dict, nfeats) tuples
        until this shard's size equals to or approximates `self.shard_size`.
        """
        iteration_index = -1
        if self.assoc_iter is not None:
            # We have associate iterator, just yields tuples
            # util we run parallel with it.
            while self.line_index < self.assoc_iter.line_index:
                line = self.corpus.readline()
                self.line_index += 1
                iteration_index += 1
                if line == '':
                    self.eof = True
                    self.corpus.close()
                    print("[up] Reached end of file {} iteration_index = {}"
                          .format(os.path.basename(self.corpus_path), iteration_index))
                    break
                    # raise AssertionError("Two corpora must have same number of lines!")

                yield self._example_dict_iter(line, iteration_index)

            print("self.assoc_iter.eof = {}".format(self.assoc_iter.eof))
            if self.assoc_iter.eof:
                self.eof = True
                self.corpus.close()
        else:
            # Yield tuples util this shard's size reaches the threshold.
            self.corpus.seek(self.last_pos)
            while True:
                self.line_index += 1
                iteration_index += 1
                if self.shard_size != 0 and self.line_index % 64 == 0:
                    # This part of check is time consuming on Py2 (but
                    # it is quite fast on Py3, weird!). So we don't bother
                    # to check for very line. Instead we chekc every 64
                    # lines. Thus we are not dividing exactly per
                    # `shard_size`, but it is not too much difference.
                    cur_pos = self.corpus.tell()
                    if cur_pos >= self.last_pos + self.shard_size:
                        self.last_pos = cur_pos
                        raise StopIteration

                line = self.corpus.readline()
                if line == '':
                    self.eof = True
                    self.corpus.close()
                    print("[bt] Reached end of file {} iteration_index = {}"
                          .format(os.path.basename(self.corpus_path), iteration_index))
                    break
                    # raise StopIteration('{} is misformated at {}'.format(self.corpus_path, self.line_index))

                yield self._example_dict_iter(line, iteration_index)

    def hit_end(self):
        return self.eof

    @property
    def num_feats(self):
        # We peek the first line and seek back to
        # the beginning of the file.
        saved_pos = self.corpus.tell()

        line = self.corpus.readline().split()
        if self.line_truncate:
            line = line[:self.line_truncate]
        _, _, self.n_feats = TextDataset.extract_text_features(line)

        self.corpus.seek(saved_pos)

        return self.n_feats

    def _example_dict_iter(self, line, index):
        line = line.split()
        if self.line_truncate:
            line = line[:self.line_truncate]
        words, feats, n_feats = TextDataset.extract_text_features(line)
        example_dict = {self.side: words, "indices": index}
        if self.side == 'tgt1':
            example_dict = {self.side: words, 'tgt1_planning': [int(word) for word in words], "indices": index}
        if feats:
            # All examples must have same number of features.
            aeq(self.n_feats, n_feats)

            prefix = self.side + "_feat_"
            example_dict.update((prefix + str(j), f)
                                for j, f in enumerate(feats))

        return example_dict
