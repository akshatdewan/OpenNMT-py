# -*- coding: utf-8 -*-

import codecs
import os
import io

import torch
import torchtext

import onmt.io.TextDataset
from onmt.io.DatasetBase import ONMTDatasetBase, PAD_WORD, BOS_WORD, EOS_WORD

class AudioDataset(ONMTDatasetBase):
    """ Dataset for data_type=='audio'

        Build `Example` objects, `Field` objects, and filter_pred function
        from audio corpus.

        Args:
            fields (dict): a dictionary of `torchtext.data.Field`.
            src_examples_iter (dict iter): preprocessed source example
                dictionary iterator.
            tgt_examples_iter (dict iter): preprocessed target example
                dictionary iterator.
            num_src_feats (int): number of source side features.
            num_tgt_feats (int): number of target side features.
            tgt_seq_length (int): maximum target sequence length.
            sample_rate (int): sample rate.
            window_size (float): window size for spectrogram in seconds.
            window_stride (float): window stride for spectrogram in seconds.
            window (str): window type for spectrogram generation.
            normalize_audio (bool): subtract spectrogram by mean and divide
                by std or not.
            use_filter_pred (bool): use a custom filter predicate to filter
                out examples?
    """
    def __init__(self, fields, src_examples_iter, tgt_examples_iter,
                 num_src_feats=0, num_tgt_feats=0,
                 tgt_seq_length=0, sample_rate=0,
                 window_size=0.0, window_stride=0.0, window=None,
                 normalize_audio=True, use_filter_pred=True):
        self.data_type = 'audio'

        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride
        self.window = window
        self.normalize_audio = normalize_audio

        self.n_src_feats = num_src_feats
        self.n_tgt_feats = num_tgt_feats

        if tgt_examples_iter is not None:
            examples_iter = (self._join_dicts(src, tgt) for src, tgt in
                             zip(src_examples_iter, tgt_examples_iter))
        else:
            examples_iter = src_examples_iter

        # Peek at the first to see which fields are used.
        ex, examples_iter = self._peek(examples_iter)
        keys = ex.keys()

        out_fields = [(k, fields[k]) if k in fields else (k, None)
                      for k in keys]
        example_values = ([ex[k] for k in keys] for ex in examples_iter)
        out_examples = (self._construct_example_fromlist(
                            ex_values, out_fields)
                        for ex_values in example_values)
        # If out_examples is a generator, we need to save the filter_pred
        # function in serialization too, which would cause a problem when
        # `torch.save()`. Thus we materialize it as a list.
        out_examples = list(out_examples)

        def filter_pred(example):
            if tgt_examples_iter is not None:
                #if len(example.tgt) > 500:
                #    print(len(example.tgt))
                return 0 < len(example.tgt) <= tgt_seq_length
            else:
                return True

        filter_pred = filter_pred if use_filter_pred else lambda x: True

        super(AudioDataset, self).__init__(
            out_examples, out_fields, filter_pred
        )

    def sort_key(self, ex):
        """ Sort using duration time of the sound spectrogram. """
        return ex.src.size(1)

    @staticmethod
    def make_audio_examples_nfeats_tpl(path, audio_dir,
                                       sample_rate, window_size,
                                       window_stride, window,
                                       normalize_audio, truncate=None):
        """
        Args:
            path (str): location of a src file containing audio paths.
            audio_dir (str): location of source audio files.
            sample_rate (int): sample_rate.
            window_size (float) : window size for spectrogram in seconds.
            window_stride (float): window stride for spectrogram in seconds.
            window (str): window type for spectrogram generation.
            normalize_audio (bool): subtract spectrogram by mean and divide
                by std or not.
            truncate (int): maximum audio length (0 or None for unlimited).

        Returns:
            (example_dict iterator, num_feats) tuple
        """
        examples_iter = AudioDataset.read_audio_file(
                path, audio_dir, "src", sample_rate,
                window_size, window_stride, window,
                normalize_audio, truncate)
        num_feats = 0  # Source side(audio) has no features.

        return (examples_iter, num_feats)

    @staticmethod
    def read_audio_file(path, src_dir, side, sample_rate, window_size,
                        window_stride, window, normalize_audio,
                        truncate=None):
        """
        Args:
            path (str): location of a src file containing audio paths.
            src_dir (str): location of source audio files.
            side (str): 'src' or 'tgt'.
            sample_rate (int): sample_rate.
            window_size (float) : window size for spectrogram in seconds.
            window_stride (float): window stride for spectrogram in seconds.
            window (str): window type for spectrogram generation.
            normalize_audio (bool): subtract spectrogram by mean and divide
                by std or not.
            truncate (int): maximum audio length (0 or None for unlimited).

        Yields:
            a dictionary containing audio data for each line.
        """
        assert (src_dir is not None) and os.path.exists(src_dir),\
            "src_dir must be a valid directory if data_type is audio"

        global torchaudio, librosa, np
        import torchaudio
        import librosa
        import numpy as np

        with codecs.open(path, "r", "utf-8") as corpus_file:
            index = 0
            for line in corpus_file:
                audio_path = os.path.join(src_dir, line.strip())
                if not os.path.exists(audio_path):
                    audio_path = line

                assert os.path.exists(audio_path), \
                    'audio path %s not found' % (line.strip())

                sound, sample_rate = torchaudio.load(audio_path)
                if truncate and truncate > 0:
                    if sound.size(0) > truncate:
                        continue

                assert sample_rate == sample_rate, \
                    'Sample rate of %s != -sample_rate (%d vs %d)' \
                    % (audio_path, sample_rate, sample_rate)

                sound = sound.numpy()
                if len(sound.shape) > 1:
                    if sound.shape[1] == 1:
                        sound = sound.squeeze()
                    else:
                        sound = sound.mean(axis=1)  # average multiple channels

                n_fft = int(sample_rate * window_size)
                win_length = n_fft
                hop_length = int(sample_rate * window_stride)
                # STFT
                d = librosa.stft(sound, n_fft=n_fft, hop_length=hop_length,
                                 win_length=win_length, window=window)
                spect, _ = librosa.magphase(d)
                spect = np.log1p(spect)
                spect = torch.FloatTensor(spect)
                if normalize_audio:
                    mean = spect.mean()
                    std = spect.std()
                    spect.add_(-mean)
                    spect.div_(std)

                example_dict = {side: spect,
                                side + '_path': line.strip(),
                                'indices': index}
                index += 1

                yield example_dict

    @staticmethod
    def get_fields(n_src_features, n_tgt_features):
        """
        Args:
            n_src_features: the number of source features to
                create `torchtext.data.Field` for.
            n_tgt_features: the number of target features to
                create `torchtext.data.Field` for.

        Returns:
            A dictionary whose keys are strings and whose values
            are the corresponding Field objects.
        """
        fields = {}

        def make_audio(data, vocab, is_train):
            nfft = data[0].size(0)
            t = max([t.size(1) for t in data])
            sounds = torch.zeros(len(data), 1, nfft, t)
            for i, spect in enumerate(data):
                sounds[i, :, :, 0:spect.size(1)] = spect
            return sounds

        fields["src"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.FloatTensor,
            postprocessing=make_audio, sequential=False)

        for j in range(n_src_features):
            fields["src_feat_"+str(j)] = \
                torchtext.data.Field(pad_token=PAD_WORD)

        fields["tgt"] = torchtext.data.Field(
            init_token=BOS_WORD, eos_token=EOS_WORD,
            pad_token=PAD_WORD)

        for j in range(n_tgt_features):
            fields["tgt_feat_"+str(j)] = \
                torchtext.data.Field(init_token=BOS_WORD, eos_token=EOS_WORD,
                                     pad_token=PAD_WORD)

        def make_src(data, vocab, is_train):
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
            tgt_size = max([t.size(0) for t in data])
            alignment = torch.zeros(tgt_size, len(data)).long()
            for i, sent in enumerate(data):
                alignment[:sent.size(0), i] = sent
            return alignment

        fields["alignment"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.LongTensor,
            postprocessing=make_tgt, sequential=False)

        fields["indices"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.LongTensor,
            sequential=False)

        return fields

    @staticmethod
    def get_num_features(corpus_file, side):
        """
        For audio corpus, source side is in form of audio, thus
        no feature; while target side is in form of text, thus
        we can extract its text features.

        Args:
            corpus_file (str): file path to get the features.
            side (str): 'src' or 'tgt'.

        Returns:
            number of features on `side`.
        """
        if side == 'src':
            num_feats = 0
        else:
            with codecs.open(corpus_file, "r", "utf-8") as cf:
                f_line = cf.readline().strip().split()
                _, _, num_feats = AudioDataset.extract_text_features(f_line)

        return num_feats

class ShardedAudioCorpusIterator(object):
    """
    This is the iterator for audio corpus, used for sharding large audio
    corpus into small shards, to avoid hogging memory.

    Inside this iterator, it automatically divides the corpus file into
    shards of size `shard_size`. Then, for each shard, it processes
    into (example_dict, n_features) tuples when iterates.
    """
    def __init__(self, corpus_path, line_truncate, side, shard_size, src_dir,
                 sample_rate, window_size, window_stride, window, normalize_audio, use_filter_pred, assoc_iter=None):
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

        self.line_truncate = line_truncate
        self.side = side
        self.shard_size = shard_size
        self.src_dir = src_dir
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride
        self.window = window
        self.normalize_audio = normalize_audio
        self.use_filter_pred = use_filter_pred
        self.assoc_iter = assoc_iter
        self.last_pos = 0
        self.line_index = -1
        self.eof = False
        self.length = 0

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
                if line == '':
                    raise AssertionError(
                        "Two corpuses must have same number of lines!")

                self.line_index += 1
                iteration_index += 1
                yield self._example_dict_iter(line, iteration_index)

            if self.assoc_iter.eof:
                self.eof = True
                self.corpus.close()
        else:
            # Yield tuples util this shard's size reaches the threshold.
            self.corpus.seek(self.last_pos)
            while True:
                if self.shard_size != 0:
                    # This part of check is time consuming on Py2 (but
                    # it is quite fast on Py3, weird!). So we don't bother
                    # to check for very line. Instead we chekc every 64
                    # lines. Thus we are not dividing exactly per
                    # `shard_size`, but it is not too much difference.
                    cur_pos = self.corpus.tell()
                    if self.length >= self.shard_size * 1000000 / 2:
                        self.length = 0
                        self.last_pos = cur_pos
                        raise StopIteration
                line = self.corpus.readline()
                if line == '':
                    self.eof = True
                    self.corpus.close()
                    raise StopIteration

                self.line_index += 1
                iteration_index += 1
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
        _, _, self.n_feats = AudioDataset.extract_text_features(line)

        self.corpus.seek(saved_pos)

        return self.n_feats

    def _example_dict_iter(self, line, index):
        if self.side == 'tgt':
            line = line.split()
            if self.line_truncate:
                line = line[:self.line_truncate]
            words, feats, n_feats = AudioDataset.extract_text_features(line)
            example_dict = {self.side: words, "indices": index}
            if feats:
                # All examples must have same number of features.
                aeq(self.n_feats, n_feats)

                prefix = self.side + "_feat_"
                example_dict.update((prefix + str(j), f)
                                    for j, f in enumerate(feats))

        else:
            global torchaudio, librosa, np
            import torchaudio
            import librosa
            import numpy as np
            audio_path = os.path.join(self.src_dir, line.strip())
            if not os.path.exists(audio_path):
                audio_path = line

            assert os.path.exists(audio_path), \
                'audio path %s not found' % (line.strip())

            sound, sample_rate = torchaudio.load(audio_path)
            if self.line_truncate and self.line_truncate > 0:
                if sound.size(0) > truncate:
                    #continue
                    return

            assert sample_rate == self.sample_rate, \
                'Sample rate of %s != -sample_rate (%d vs %d)' \
                % (audio_path, sample_rate, sample_rate)

            sound = sound.numpy()
            if len(sound.shape) > 1:
                if sound.shape[1] == 1:
                    sound = sound.squeeze()
                else:
                    sound = sound.mean(axis=1)  # average multiple channels
            n_fft = int(self.sample_rate * self.window_size)
            win_length = n_fft
            hop_length = int(self.sample_rate * self.window_stride)
            # STFT
            d = librosa.stft(sound, n_fft=n_fft, hop_length=hop_length,
                             win_length=win_length, window=self.window)

            spect, _ = librosa.magphase(d)
            spect = np.log1p(spect)
            spect = torch.FloatTensor(spect)
            if self.normalize_audio:
                mean = spect.mean()
                std = spect.std()
                spect.add_(-mean)
                spect.div_(std)
            example_dict = {self.side: spect,
                            self.side + '_path': line.strip(),
                            'indices': index}
            self.length += os.path.getsize(audio_path)
        return example_dict

