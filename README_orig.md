# semtagger

Requirements:

* Keras (use fork by bjerva, see below)
* Tensorflow or Theano
* h5py (for model saving)

Tested with:
* Python 2.7.11 / 3.5.1
* Keras 1.0.4
* Tensorflow 0.8.0


### Keras:

Use the fork by Johannes (see issue #12: https://github.com/bjerva/semtagger/issues/12):
git clone https://github.com/bjerva/keras.git

And set you PYTHONPATH to this directory, e.g.
export PYTHONPATH=$PYTHONPATH:/home/username/code/keras/

### Peregrine

To run on Peregrine, load modules in this order (cuDNN optional).

module load tensorflow/0.8.0-foss-2016a-Python-3.5.1-CUDA-7.5.18

module load h5py/2.5.0-foss-2016a-Python-3.5.1-HDF5-1.8.16

module load cuDNN

Make sure your .keras/keras.json contains:
"backend": "tensorflow"

Running example:

python src/semtagger.py --train ~/my_data/semtag/ud1.2/en-ud-train.conllu ~/my_data/semtag/v0_6/semtag_train.conll --dev ~/my_data/semtag/ud1.2/en-ud-dev.conllu ~/my_data/semtag/v0_6/semtag_dev.conll --test ~/my_data/semtag/ud1.2/en-ud-test.conllu ~/my_data/semtag/v0_6/semtag_test.conll  --words --chars --bn --bypass --inception 3 --dropout 0.5 --rnn --shorten-sents --epochs 50 --bsize 500 --bookkeeping ~/my_data/semtag/bookkeeping/ --tag inception3_bypass 
