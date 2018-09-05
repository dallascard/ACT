#ACT: Averaging Classifiers for Text

This is a preliminary repo for classifiers restricted to make predictions in terms of a weighted average of training data, specifically for text datasets.

### Requirements:

- python3
- pytorch 0.4.0
- torchvision
- numpy
- spacy
- scikit-learn

### Setup:

In addition to setting up a python environment with the packages listed above, these models assume access to Glove embeddings, which can be downloaded from [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)

By default, the models will look for the embeddings in `data/glove/` but a different location can be specified at run time.

### Basic Usage:

To train a model using one of the pre-specified datasets, such as StackOverflow use:

`python run.py --dataset stackoverflow`

This will download the IMDB dataset to `data/stackoverflow`, preprocess it, train a baseline CNN model, predict on the test data, and save the output to `data/temp`.

The output directory will contain files for the train, dev, and test data, each of which is .npz file containing labels, predictions, and predicted probabilities.

To train a weighted averaging model, use `--model act`

### Custom datasets:

To train a model on a dataset that has not been prespecified, create a directory called `data/[name]/raw`, where `[name]` is the name of your dataset. In that directory, created files called `train.jsonlist` and `test.jsonlist`. Each of those files should contain one document per line. Each line should be a JSON object with at least two fields: "text" and "label".

For example, the first line of a file could be the following JSON object:
`{"text": "This is a positive document", "label": "positive"}`

To train a model on this data, use:

`python run.py --dataset [name]`

again replacing `[name]` with the name of your dataset as above.

This will load the data, tokenize the text, and then proceeed as above.

### Options:

To choose the size of the output layer for the averaging classifier, use `--z-dim [dz]`, where `[dz]` will be the dimensionality.

To train on a GPU, include the option `--cuda`.

To choose a different output directory, use `--output-dir [output-dir]` where `[output-dir]` is the desired target directory.

For additional options, such as model size and optimization choices, run:

`python run.py -h`

### Evaluation:

The `eval` directory contains a number of scripts to help with evaluation. For example, to evaluate the calibration (and accuracy) of the predictions on test data in the `data/temp` directory, use:

`python -m eval.eval_calibration data/temp/test.npz`

To inspect the calibration and confidence values, and correcteness at a given epsilon value, say 0.1, use:

`python -m eval.eval_conformal data/temp --eps 0.1`

To evaluate these using the sum of weights rather than the probabilities, add `--weights`.