from optparse import OptionParser

from datasets.loader import load_dataset


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--dataset', dest='dataset', default='imdb',
                      help='Dataset [imdb|amazon|stackoverflow|subjectivity|framing]: default=%default')
    parser.add_option('--subset', dest='subset', default=None,
                      help='Subset (for amazon): default=%default')
    parser.add_option('--root-dir', default='./data',
                      help='Root dir: default=%default')
    parser.add_option('--partition', dest="partition", default='test',
                      help='Partition to load instance from [train|test|ood]: default=%default')
    parser.add_option('-i', type=int, dest='index', default=0,
                      help='Index to print: default=%default')

    (options, args) = parser.parse_args()
    partition = options.partition
    index = options.index

    train_dataset, test_dataset, ood_dataset = load_dataset(options.root_dir, options.dataset, options.subset, lower=False)

    if partition == 'train':
        dataset = train_dataset
    elif partition == 'test':
        dataset = test_dataset
    elif partition == 'ood':
        dataset = ood_dataset
    else:
        raise RuntimeError("Partition must be train, test, or ood")

    document = dataset.all_docs[index]
    print(' '.join(document[dataset.text_field_name]), 'Label: {:s}'.format(str(document[dataset.label_field_name])))


if __name__ == '__main__':
    main()
