# -*- coding: utf-8 -*-

from collections import Counter, defaultdict, OrderedDict
from itertools import count

import torch
import torchtext.data
import torchtext.vocab

from onmt.io.DatasetBase import UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD
from onmt.io.TextDataset import TextDataset
from onmt.io.ImageDataset import ImageDataset
from onmt.io.AudioDataset import AudioDataset
from tqdm import tqdm

def _getstate(self):
    return dict(self.__dict__, stoi=dict(self.stoi))


def _setstate(self, state):
    self.__dict__.update(state)
    self.stoi = defaultdict(lambda: 0, self.stoi)


torchtext.vocab.Vocab.__getstate__ = _getstate
torchtext.vocab.Vocab.__setstate__ = _setstate


def get_fields(data_type, n_src_features, n_tgt_features):
    """
    Args:
        data_type: type of the source input. Options are [text|img|audio].
        n_src_features: the number of source features to
            create `torchtext.data.Field` for.
        n_tgt_features: the number of target features to
            create `torchtext.data.Field` for.

    Returns:
        A dictionary whose keys are strings and whose values are the
        corresponding Field objects.
    """
    if data_type == 'text':
        return TextDataset.get_fields(n_src_features, n_tgt_features)
    elif data_type == 'img':
        return ImageDataset.get_fields(n_src_features, n_tgt_features)
    elif data_type == 'audio':
        return AudioDataset.get_fields(n_src_features, n_tgt_features)


def load_fields_from_vocab(vocab, data_type="text"):
    """
    Load Field objects from `vocab.pt` file.
    """
    vocab = dict(vocab)
    n_src_features = len(collect_features(vocab, 'src1'))
    n_tgt_features = len(collect_features(vocab, 'tgt1'))
    fields = get_fields(data_type, n_src_features, n_tgt_features)
    for k, v in vocab.items():
        # Hack. Can't pickle defaultdict :(
        v.stoi = defaultdict(lambda: 0, v.stoi)
        fields[k].vocab = v
    return fields


def save_fields_to_vocab(fields):
    """
    Save Vocab objects in Field objects to `vocab.pt` file.
    """
    vocab = []
    for k, f in fields.items():
        if f is not None and 'vocab' in f.__dict__:
            f.vocab.stoi = dict(f.vocab.stoi)
            vocab.append((k, f.vocab))
    return vocab


def merge_vocabs(vocabs, vocab_size=None):
    """
    Merge individual vocabularies (assumed to be generated from disjoint
    documents) into a larger vocabulary.

    Args:
        vocabs: `torchtext.vocab.Vocab` vocabularies to be merged
        vocab_size: `int` the final vocabulary size. `None` for no limit.
    Return:
        `torchtext.vocab.Vocab`
    """
    merged = sum([vocab.freqs for vocab in vocabs], Counter())
    return torchtext.vocab.Vocab(merged,
                                specials=[UNK_WORD, PAD_WORD,
                                            BOS_WORD, EOS_WORD],
                                max_size=vocab_size)


def get_num_features(data_type, corpus_file, side):
    """
    Args:
        data_type (str): type of the source input.
            Options are [text|img|audio].
        corpus_file (str): file path to get the features.
        side (str): for source or for target.

    Returns:
        number of features on `side`.
    """
    assert side in ['src1', 'src2', 'tgt1', 'tgt2']

    if data_type == 'text':
        return TextDataset.get_num_features(corpus_file, side)
    elif data_type == 'img':
        return ImageDataset.get_num_features(corpus_file, side)
    elif data_type == 'audio':
        return AudioDataset.get_num_features(corpus_file, side)


def make_features(batch, side, data_type='text'):
    """
    Args:
        batch (Variable): a batch of source or target data.
        side (str): for source or for target.
        data_type (str): type of the source input.
            Options are [text|img|audio].
    Returns:
        A sequence of src/tgt tensors with optional feature tensors
        of size (len x batch).
    """
    assert side in ['src1','src2', 'tgt1', 'tgt2']
    if isinstance(batch.__dict__[side], tuple):
        data = batch.__dict__[side][0]
    else:
        data = batch.__dict__[side]

    feat_start = side + "_feat_"
    keys = sorted([k for k in batch.__dict__ if feat_start in k])
    features = [batch.__dict__[k] for k in keys]
    levels = [data] + features

    if data_type == 'text':
        return torch.cat([level.unsqueeze(2) for level in levels], 2)
    else:
        return levels[0]


def collect_features(fields, side="src1"):
    """
    Collect features from Field object.
    """
    assert side in ['src1', 'src2', 'tgt1', 'tgt2']
    feats = []
    for j in count():
        key = side + "_feat_" + str(j)
        if key not in fields:
            break
        feats.append(key)
    return feats


def collect_feature_vocabs(fields, side):
    """
    Collect feature Vocab objects from Field object.
    """
    assert side in ['src1', 'src2', 'tgt1', 'tgt2']
    feature_vocabs = []
    for j in count():
        key = side + "_feat_" + str(j)
        if key not in fields:
            break
        feature_vocabs.append(fields[key].vocab)
    return feature_vocabs

# for translate
def build_dataset(fields, data_type, src_path, tgt_path, src_path2, tgt_path2, src_dir=None,
                    src_seq_length=0, tgt_seq_length=0,
                    src_seq_length_trunc=0, tgt_seq_length_trunc=0,
                    dynamic_dict=True, sample_rate=0,
                    window_size=0, window_stride=0, window=None,
                    normalize_audio=True, use_filter_pred=True,
                    edge_file=None):

    # Build src/tgt examples iterator from corpus files, also extract
    # number of features.
    #! NOTE: building iterators using essentially the same APIs for text
    src_examples_iter, num_src_feats = \
        _make_examples_nfeats_tpl(data_type, src_path, src_dir,
                                    src_seq_length_trunc, sample_rate,
                                    window_size, window_stride,
                                    window, normalize_audio, "src1")

    tgt_examples_iter, num_tgt_feats = \
        TextDataset.make_text_examples_nfeats_tpl(
            tgt_path, tgt_seq_length_trunc, "tgt1")

    src_examples_iter2, num_src_feats2 = \
        _make_examples_nfeats_tpl(data_type, src_path2, src_dir,
                                    src_seq_length_trunc, sample_rate,
                                    window_size, window_stride,
                                    window, normalize_audio, "src2")

    tgt_examples_iter2, num_tgt_feats2 = \
        TextDataset.make_text_examples_nfeats_tpl(
            tgt_path2, tgt_seq_length_trunc, "tgt2")

    #! NOTE building dataset
    if data_type == 'text':
        dataset = TextDataset(fields, src_examples_iter, tgt_examples_iter, src_examples_iter2, tgt_examples_iter2,
                                num_src_feats, num_tgt_feats,
                                src_seq_length=src_seq_length,
                                tgt_seq_length=tgt_seq_length,
                                dynamic_dict=dynamic_dict,
                                use_filter_pred=use_filter_pred,
                                edge_file=edge_file)
    '''
    # temporarily not used
    elif data_type == 'img':
        dataset = ImageDataset(fields, src_examples_iter, tgt_examples_iter,
                                num_src_feats, num_tgt_feats,
                                tgt_seq_length=tgt_seq_length,
                                use_filter_pred=use_filter_pred)

    elif data_type == 'audio':
        dataset = AudioDataset(fields, src_examples_iter, tgt_examples_iter,
                                num_src_feats, num_tgt_feats,
                                tgt_seq_length=tgt_seq_length,
                                sample_rate=sample_rate,
                                window_size=window_size,
                                window_stride=window_stride,
                                window=window,
                                normalize_audio=normalize_audio,
                                use_filter_pred=use_filter_pred)
    '''
    return dataset


def _build_field_vocab(field, counter, **kwargs):
    specials = list(OrderedDict.fromkeys(
        tok for tok in [field.unk_token, field.pad_token, field.init_token,
                        field.eos_token]
        if tok is not None))
    field.vocab = field.vocab_cls(counter, specials=specials, **kwargs)


def build_vocab(train_dataset_files, fields, data_type, share_vocab,
                src_vocab_size, src_words_min_frequency,
                tgt_vocab_size, tgt_words_min_frequency):
    """
    Args:
        train_dataset_files: a list of train dataset pt file.
        fields (dict): fields to build vocab for.
        data_type: "text", "img" or "audio"?
        share_vocab(bool): share source and target vocabulary?
        src_vocab_size(int): size of the source vocabulary.
        src_words_min_frequency(int): the minimum frequency needed to
                include a source word in the vocabulary.
        tgt_vocab_size(int): size of the target vocabulary.
        tgt_words_min_frequency(int): the minimum frequency needed to
                include a target word in the vocabulary.

    Returns:
        Dict of Fields
    """
    counter = {}
    for k in fields:
        counter[k] = Counter()

    for path in train_dataset_files:
        dataset = torch.load(path)
        print(" * reloading %s and build counters" % path)
        for ex in tqdm(dataset.examples):
            for k in fields:
                #! don't update edge vocabs to save time
                if 'edge' in k:
                    continue
                val = getattr(ex, k, None)
                if val is not None and k in ('indices', 'src_map', 'alignment'):
                    val = [val]
                counter[k].update(val)

    # TODO: save edge label vocab during graph construction, and include here
    for _ in range(10):
        counter['edge_labels'].update(['greater', 'equal', 'has_player', 'has_record', 'top_1', 'top_2', 'top_3'])

    for tgt in ("tgt1", "tgt2"):
        _build_field_vocab(fields[tgt], counter[tgt],
                            max_size=tgt_vocab_size,
                            min_freq=tgt_words_min_frequency)
        print(" * %s vocab size: %d." % (tgt, len(fields[tgt].vocab)))

        # All datasets have same num of n_tgt_features,
        # getting the last one is OK.

        for j in range(dataset.n_tgt_feats):
            key = tgt+"_feat_" + str(j)
            _build_field_vocab(fields[key], counter[key])
            print(" * %s vocab size: %d." % (key, len(fields[key].vocab)))

    if "edge_labels" in fields:
        _build_field_vocab(fields["edge_labels"], counter["edge_labels"],
                            max_size=tgt_vocab_size,
                            min_freq=tgt_words_min_frequency)
        print(" * edge_labels vocab size: {}.".format(len(fields["edge_labels"].vocab)))
        print(" * edge_labels vocab: {}".format(fields["edge_labels"].vocab.stoi))

    if data_type == 'text':
        for src in ("src1", "src2"):
            _build_field_vocab(fields[src], counter[src],
                                max_size=src_vocab_size,
                                min_freq=src_words_min_frequency)
            print(" * %s vocab size: %d." % (src, len(fields[src].vocab)))

            # All datasets have same num of n_src_features,
            # getting the last one is OK.
            for j in range(dataset.n_src_feats):
                key = src+"_feat_" + str(j)
                _build_field_vocab(fields[key], counter[key])
                print(" * %s vocab size: %d." % (key, len(fields[key].vocab)))

        # Merge the input and output vocabularies.
        if share_vocab:
            # `tgt_vocab_size` is ignored when sharing vocabularies
            print(" * merging src and tgt vocab...")
            merged_vocab = merge_vocabs(
                [fields["src1"].vocab, fields["src2"].vocab, fields["tgt2"].vocab],
                vocab_size=src_vocab_size)
            fields["src1"].vocab = merged_vocab
            fields["src2"].vocab = merged_vocab
            fields["tgt2"].vocab = merged_vocab

    return fields


def _make_examples_nfeats_tpl(data_type, src_path, src_dir,
                                src_seq_length_trunc, sample_rate,
                                window_size, window_stride,
                                window, normalize_audio, src="src1"):
    """
    Process the corpus into (example_dict iterator, num_feats) tuple
    on source side for different 'data_type'.
    """

    if data_type == 'text':
        src_examples_iter, num_src_feats = \
            TextDataset.make_text_examples_nfeats_tpl(
                src_path, src_seq_length_trunc, src)

    elif data_type == 'img':
        src_examples_iter, num_src_feats = \
            ImageDataset.make_image_examples_nfeats_tpl(
                src_path, src_dir)

    elif data_type == 'audio':
        src_examples_iter, num_src_feats = \
            AudioDataset.make_audio_examples_nfeats_tpl(
                src_path, src_dir, sample_rate,
                window_size, window_stride, window,
                normalize_audio)

    return src_examples_iter, num_src_feats


class OrderedIterator(torchtext.data.Iterator):
    def sort_batch_key(self, ex):
        """ Sort using length of source sentences. """
        return len(ex.src1)
    def create_batches(self):
        if self.train:
            def pool(data, random_shuffler):
                for p in torchtext.data.batch(data, self.batch_size * 100):
                    p_batch = torchtext.data.batch(
                        sorted(p, key=self.sort_batch_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in torchtext.data.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_batch_key))
