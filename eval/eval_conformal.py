import os
from optparse import OptionParser

import numpy as np


def main():
    usage = "%prog exp_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('--eps', dest='epsilon', default=0.05,
                      help='Confidence threshold (1 - desired accuracy): default=%default')
    parser.add_option('--weights', action="store_true", dest="use_weights", default=False,
                      help='Use weights instead of probabilities [ACT model only]: default=%default')

    (options, args) = parser.parse_args()
    indir = args[0]

    test_file = os.path.join(indir, 'test.npz')
    dev_file = os.path.join(indir, 'dev.npz')

    epsilon = float(options.epsilon)
    use_weights = options.use_weights

    eval_conformal_baseline(test_file, dev_file, epsilon, use_weights)


def eval_conformal_baseline(test_file, dev_file, epsilon, use_weights=False):

    test_data = np.load(test_file)
    dev_data = np.load(dev_file)

    dev_labels = dev_data['labels']
    dev_pred_probs = dev_data['pred_probs']
    n_dev, _ = dev_pred_probs.shape

    test_labels = test_data['labels']
    test_pred_probs = test_data['pred_probs']
    n_test, n_classes = test_pred_probs.shape

    print("Accuracy = ", np.sum(test_labels == test_pred_probs.argmax(axis=1))/float(n_test))

    # scatter the labels
    test_labels = scatter(test_labels, n_classes)
    dev_labels = scatter(dev_labels, n_classes)

    if use_weights:
        dev_confs = dev_data['confs']
        test_confs = test_data['confs']
        dev_scores = -dev_confs[np.arange(n_dev), dev_labels.argmax(axis=1)]
        test_scores = -test_confs
    else:
        # get the nonconformity score for each dev item, based on the true label
        dev_scores = -dev_pred_probs[np.arange(n_dev), dev_labels.argmax(axis=1)]
        # get the nonconformity score for all test items for all labels
        test_scores = -test_pred_probs

    # compute conformal p-values for all test points/labels = proportion of dev points with greater nonconformity
    test_quantiles = np.sum(test_scores.reshape((n_test, n_classes, 1)) < dev_scores.reshape((1, 1, n_dev)), axis=2) / float(n_dev)

    credibility = test_quantiles.max(axis=1)
    temp = test_quantiles.copy()
    temp[np.arange(n_test), test_quantiles.argmax(axis=1)] = 0.0
    confidence = 1.0 - temp.max(axis=1)

    print("Mean credibility:", np.mean(credibility))
    print("Credibility histogram [0-1]:", np.histogram(credibility, bins=np.linspace(0, 1.0, 11))[0])
    print("Confidences histogram [0-1]:", np.histogram(confidence, bins=np.linspace(0, 1.0, 11))[0])

    test_preds = np.array(test_quantiles > epsilon, dtype=int)
    correct_prop = np.sum(test_preds[np.arange(n_test), test_labels.argmax(axis=1)]) / float(n_test)
    print("Correct @ {:.2f} = {:.2f}%".format(epsilon, correct_prop*100))

    label_set_counts = np.histogram(test_preds.sum(axis=1), bins=np.arange(n_classes+2))[0]
    print("Empty predictions: {:d} {:.2f}%".format(label_set_counts[0], label_set_counts[0]/float(n_test)*100))
    print("Singly-labeled   : {:d} {:.2f}%".format(label_set_counts[1], label_set_counts[1]/float(n_test)*100))
    print("Multiply-labeled : {:d} {:.2f}%".format(label_set_counts[2:].sum(), label_set_counts[2:].sum()/float(n_test)*100))


def scatter(labels, n_classes):
    if len(labels.shape) == 1 or labels.shape[1] == 1:
        n_items = len(labels)
        temp = np.zeros((n_items, n_classes), dtype=int)
        temp[np.arange(n_items), labels] = 1
        labels = temp
    return labels


if __name__ == '__main__':
    main()
