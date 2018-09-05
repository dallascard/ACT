import os
import errno

from spacy.lang.en import English

from utils import file_handling as fh
from datasets.text_dataset import TextDataset, Vocab, tokenize


class CustomDataset(TextDataset):
    """Custom text data."""

    raw_folder = 'raw'
    processed_folder = 'processed'
    train_file = 'train.jsonlist'
    test_file = 'test.jsonlist'
    vocab_file = 'vocab.json'
    label_file = 'label_list.json'

    def __init__(self, root, train=True, lower=True):
        super().__init__()
        self.root = os.path.join(os.path.expanduser(root))
        self.train = train
        self.raw_text_field_name = 'text'
        self.raw_label_field_name = 'label'
        self.text_field_name = 'tokens'
        self.label_field_name = 'label'
        self.classes = None
        self.class_to_idx = None

        if not self._check_raw_exists():
            raise RuntimeError('Dataset not found. You must place train.json and test.json in the raw directory')

        self.preprocess()

        if train:
            self.all_docs = fh.read_jsonlist(os.path.join(self.root, self.processed_folder, self.train_file))
        else:
            self.all_docs = fh.read_jsonlist(os.path.join(self.root, self.processed_folder, self.test_file))

        # Do lower-casing on demand, to avoid redoing slow tokenization
        if lower:
            for doc in self.all_docs:
                doc['tokens'] = [token.lower() for token in doc['tokens']]

        # load and build a vocabulary, also lower-casing if necessary
        vocab = fh.read_json(os.path.join(self.root, self.processed_folder, self.vocab_file))
        if lower:
            vocab = list(set([token.lower() for token in vocab]))
        self.vocab = Vocab(vocab, add_pad_unk=True)

        self.classes = fh.read_json(os.path.join(self.root, self.processed_folder, self.label_file))
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

        self.label_vocab = Vocab(self.classes)

    def _check_processed_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.train_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.vocab_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.label_file))

    def _check_raw_exists(self):
        return os.path.exists(os.path.join(self.root, self.raw_folder, self.train_file)) and \
               os.path.exists(os.path.join(self.root, self.raw_folder, self.test_file))

    def preprocess(self):
        """Preprocess the raw data files (train and test)"""
        if self._check_processed_exists():
            return

        try:
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        print("Preprocessing raw data")
        print("Loading spacy")
        # load a spacy parser
        tokenizer = English()

        train_lines = []
        test_lines = []
        vocab = set()

        train_labels = set()
        test_labels = set()

        train_docs = fh.read_jsonlist(os.path.join(self.root, self.raw_folder, self.train_file))
        test_docs = fh.read_jsonlist(os.path.join(self.root, self.raw_folder, self.test_file))

        print("Processing training documents")
        for d_i, doc in enumerate(train_docs):
            if d_i % 1000 == 0:
                print(d_i)

            label = doc[self.raw_label_field_name]
            text = doc[self.raw_text_field_name]

            # tokenize the text
            text = tokenize(tokenizer, text)

            # save the text, label, and doc id (if available)
            if 'id' in doc:
                doc_id = doc['id']
            else:
                doc_id = 'train_' + str(d_i)

            doc_out = {'id': doc_id, self.text_field_name: text.split(), self.label_field_name: label}

            train_lines.append(doc_out)
            vocab.update(doc_out[self.text_field_name])
            train_labels.update([label])

        print("Processing test documents")
        for d_i, doc in enumerate(test_docs):
            if d_i % 1000 == 0:
                print(d_i)

            label = doc[self.raw_label_field_name]
            text = doc[self.raw_text_field_name]

            text = tokenize(tokenizer, text)

            if 'id' in doc:
                doc_id = doc['id']
            else:
                doc_id = 'test_' + str(d_i)

            doc_out = {'id': doc_id, self.text_field_name: text.split(), self.label_field_name: label}

            test_lines.append(doc_out)
            # for test documents, don't add to the vocab or train label set
            test_labels.update([label])

        print("Train counts: {:d} documents, {:d} labels".format(len(train_lines), len(train_labels)))
        print("Test counts: {:d} documents, {:d} labels".format(len(train_lines), len(test_labels)))
        vocab = list(vocab)
        vocab.sort()
        print("Vocab size = {:d}".format(len(vocab)))
        all_labels = list(train_labels.union(test_labels))
        all_labels.sort()
        print("Total number of labels = {:d}".format(len(all_labels)))

        print("Saving processed data")
        fh.write_jsonlist(train_lines, os.path.join(self.root, self.processed_folder, self.train_file))
        fh.write_jsonlist(test_lines, os.path.join(self.root, self.processed_folder, self.test_file))
        fh.write_json(vocab, os.path.join(self.root, self.processed_folder, self.vocab_file), sort_keys=False)
        fh.write_json(all_labels, os.path.join(self.root, self.processed_folder, self.label_file), sort_keys=False)
