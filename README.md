## KenLM
KenLM performs interpolated modified Kneser Ney Smoothing for estimating the n-gram probabilities.

--------

### Step-by-step guide for training an n-gram based Language Model using [KenLM toolkit](https://kheafield.com/code/kenlm/estimation/)

## 1) Installing KenLM dependencies
Before installing KenLM toolkit, you should install all the dependencies which can be found in [kenlm-dependencies](https://kheafield.com/code/kenlm/dependencies/).

**For Debian/Ubuntu distro**:

To get a working compiler, install the `build-essential` package. [Boost](https://www.boost.org/) is known as `libboost-all-dev`. The three supported compression options each have a separate dev package.

    $ sudo apt-get install build-essential libboost-all-dev cmake zlib1g-dev libbz2-dev liblzma-dev
    
## 2) Installing KenLM toolkit
For this, it's suggested to use a *conda or virtualenv* virtual environment. For conda, you can create one using:

    $ conda create -n kenlm_deepspeech python=3.6 nltk
    
Then activate the environment using:

    $ source activate kenlm_deepspeech
    
Now we're ready to install kenlm. Let's first clone the kenlm repo:

    $ git clone --recursive https://github.com/vchahun/kenlm.git

And then compile the LM estimation code using:

    $ cd kenlm
    $ ./bjam 
   
As a final step, optionally, install the Python module using:

    $ python setup.py install
    

## 3) Training a Language Model

First let's get some training data. Here, I'll use the Bible:

    $ wget -c https://github.com/vchahun/notes/raw/data/bible/bible.en.txt.bz2
   
Next we will need a simple preprocessing script. The reason is because:

- the training text should be a single text/compressed file (e.g. `.bz2`) which has a single sentence per line.
- it need to be tokenized and lowercased before feeding it into kenlm

So, create a simple script `preprocess.py` with the following lines:

```python
import sys
import nltk

for line in sys.stdin:
    for sentence in nltk.sent_tokenize(line):
        print(' '.join(nltk.word_tokenize(sentence)).lower())
```

For sanity check, do:

    $ bzcat bible.en.txt.bz2 | python preprocess.py | wc
    
And see that it works fine.

Now we can train the model. For training a *trigram model* with Kneser-Ney smoothing, use:

    # -o means `order` which translates to the `n` in n-gram
    $ bzcat bible.en.txt.bz2 |\
      python preprocess.py |\
      ./kenlm/bin/lmplz -o 3 > bible.arpa

  The above command will first pipe the data thru the preprocessing script which performs tokenization and lowercasing. Next, this tokenized and lowercased text is piped to the `lmplz` program which performs the estimation work.
  
  It should finish in a couple of seconds and then generate an arpa file `bible.arpa`. You can inspect the arpa file using something like `less` or `more` (i.e. `$ less bible.arpa`). In the very beginning, it should have a *data section* with unigram, bigram, and trigram counts followed by the estimated values.
 
 
 #### Binarizing the model
 
 ARPA files can be read directly. But, the binary format loads much faster and provides more flexibility. Using the binary format significantly reduces loading time and also exposes more configuration options. For these reasons, we will binarize the model using:
 
     $ ./kenlm/bin/build_binary bible.arpa bible.binary
     
  Note that, unlike IRSTLM, the file extension does not matter; the binary format is recognized using magic bytes.
  
  One can also use `trie` when binarizing. For this, use:
  
      $ ./kenlm/bin/build_binary trie bible.arpa bible.binary
      
  ----------------------
  
  ### Using the model (i.e. scoring sentences)
  
  Now that we have a Language Model, we can *score* sentences. It's super easy to do this using the Python interface. Below is an example:
  
  ```python
  import kenlm
  model = kenlm.LanguageModel('bible.binary')
  model.score('in the beginning was the word')
  ```
  
  Then, you might get a score such as:
  
    -15.03003978729248
  
  
  ---------------
  
  #### References:
  1) http://www.statmt.org/moses/?n=FactoredTraining.BuildingLanguageModel
  2) http://victor.chahuneau.fr/notes/2012/07/03/kenlm.html
