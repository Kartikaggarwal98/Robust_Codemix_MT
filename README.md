## Robust Machine Translation for Codemix Languages

This is a framework for translating text from codemix languages such as Hinglish, Bengalish to English. The repository includes datasets and pretrained-models for training and prediction. The models are capable of translating all-inclusive text i.e. an input with a mix of Devanagari / Romanized Hindi. The robustness capabilities of the model enables it to effectively handle spelling mistakes.

The following datasets are available:

1. hinmix_ctrans
2. hinmix_calign

The following models are available:

1. rcmt1
2. rcmt2
3. zcmt

The following languages are available:

1. Hindi (```hi```)
2. Bengali (```bn```)
3. English (```en```)

### Installation

1. Download the repository and unzip.
1. `pip install -r requirements.txt`
1. `cd jamt`
1. `pip install .`


### Training our own model from scratch

For training a model from scratch, the raw data needs to be preprocessed, tokenized, binarized and then used for training a multilingual model. We propose two robust and one zeroshot codemix translation model: RCMT1, RCMT2, ZCMT.  Change the dataset name according to the requirements: `calign/ctrans`.

**Preprocessing:** 

This involves cleaning of raw data and training a sentencepiece unigram model using train versions (noisy,romanized,codemix) of languages (hindi, bengali). The sentencepiece model is then used to tokenize all train, valid, test datasets.

1. `cd examples/translation/`
1. `bash prepare_calign.sh`

**Binarization:**

As the training data is very large (~4.5 million pairs), the raw tokenized needs to be binarized for faster loading. From the `jamt/examples/translation` directory run:

1. `cd ../../`
1. `bash calign_binarize.sh`

**Training:**

The binarized data can now be used to train a model. The default model is RCMT1. To change the model name, uncomment that specific part from the `calign_train.sh` file and run:

1. `bash calign_train.sh`

**Prediction:**

For prediction from the provided test set run:

1. `bash calign_test.sh`


### Examples


After preprocessing and training, the file structure would look like this:

```
├──    jamt
|   ├──    examples
|   |   ├──    translation
|   |   |    ├──    calign
|   |   |    ├──    ctrans
|   ├──    checkpoints
|   |   ├──    hinmix_calign_rcmt1
|   |   |    ├──    checkpoint.pt
|   ├──    data-bin
|   |   ├──    hinmix_calign_rcmt1
|   |   |   ├──    dict.lang.txt
|   |   |   ├──    fairseq.vocab
```



### Requirements

1. Python<=3.6 (for torch 1.4)
2. torch==1.4.0
3. tqdm
4. numpy
5. sentencepiece

