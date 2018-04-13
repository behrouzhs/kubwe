# Kernelized Unit Ball Word Embedding (KUBWE)

This software learns a word embedding from the input co-occurrence matrix (preferably extracted from a large corpus such as Wikipedia). This work is submitted to ECML 2018 and is under review.

The following instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites for input preparation

The input to this algorithm is a word-word co-occurrence matrix. For calculating this co-occurrence matrix, we use existing software from GloVe which can be downloaded at:

* [vocab_count](https://github.com/stanfordnlp/GloVe/blob/master/src/vocab_count.c) - This file is used to scan the corpus and build a vocabulary.
* [cooccur](https://github.com/stanfordnlp/GloVe/blob/master/src/cooccur.c) - This file is used, given a vocabulary, to calculate the word-word co-occurrence matrix.

## Compiling the source code

The source code of the software is written in C and can be compiled using standard C compilers in any operating system (Linux, Windows, and macOS). To compile the prerequisites use:

```
$ gcc -Wall -m64 -O3 vocab_count.c -o vocab_count -lm -lpthread
$ gcc -Wall -m64 -O3 cooccur.c -o cooccur -lm -lpthread
```

You can ignore `-Wall` (show all warnings), `-m64` (compile for 64-bit system), `-O3` (optimization level 3). However, `-lm` (link math library) and `-lpthread` (multi-threading library) are required to compile and run the program.

To compile our program run:

```
$ gcc -Wall -m64 -O3 pmi.c -o pmi -lm
$ gcc -Wall -fopenmp -m64 -O3 kubwe.c -o kubwe -lm
```

Our program uses OpenMP shared memory multi-threading library which is standard and is implemented in almost every C compiler. If you ignore `-fopenmp` switch, it will run on a single thread, however, for better performance use this option.

## Running the software to train a word embedding

For this purpose, you need to have a large text corpus (e.g Wikipedia) in a single text file. For instance, the latest dump of Wikipedia (articles in XML format) can be downloaded at: [https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2). For a more complete list please visit: [https://dumps.wikimedia.org/enwiki/latest/](https://dumps.wikimedia.org/enwiki/latest/).

These types of corpuses require a lot of preprocessing such as removing HTML tags and structure to get clean text from it, handling or removing special characters, etc. We will not go through the details of preprocessing but it is a neccessary step in order to get a high quality embedding with meaningful and manageable sized vocabulary.

After downloading and extracting the zip file, and also preprocessing steps you will get the clean text file. Let's call the clean text file `corpus_clean.txt`. In order to train and obtain a word embedding, run the following 4 commands one after another:

```
$ ./vocab_count -min-count 5 < ./corpus_clean.txt > ./vocab.txt
$ ./cooccur -window-size 10 -vocab-file ./vocab.txt < ./corpus_clean.txt > ./cooccurrence_matrix.bin
$ ./pmi -pmicutoff 0 -contextsmooth 0.75 -input ./cooccurrence_matrix.bin -output ./pmi_matrix.bin
$ ./kubwe -thread 16 -dim 100 -kernel 13 -input ./pmi_matrix.bin -vocab ./vocab.txt -output ./kubwe_embedding_d100.txt
```

After running the above commands, `kubwe_embedding_d100.txt` will be generated which contains the word embeddings. Each row will contain a word and its corresponding vector representation.

## Options and switches for executing the code

For `vocab_count` it is good to limit the vocabulary to the words occuring at least `-min-count` times. This option will remove extremely rare words from the vocabulary.

For `cooccur` you need to use a proper `-window-size`. Reasonable range for `-window-size` is between 5 and 15.

For our algorithm, there are two parts `pmi` and `kubwe`:

* pmi
    * -pmicutoff \<float\>: Using this option will set all the PMI values less than cutoff threshold to zero and the matrix will become sparser. (default: -infinity, which does not filter anything.)
    * -contextsmooth \<float\>: Context distribution smoothing parameter. It will raise all the context probabilities to the power of `<float>` which alleviates PMI's bias towards infrequrnt words. 0.75 has been shown to be practically a good choice. (default: 0)
    * -input \<file\>: Specifies the input co-occurrence file. This co-occurrence file is the output of `cooccur`.
    * -output \<file\>: Specifies the output PMI file. The resulting PMI table will be stored in this file.

* kubwe
    * -dim \<int\>: The dimensionality of the word embedding. (default: 100)
    * -thread \<int\>: The number of threads to use in parallel processing. (default: 4)
    * -kernel \<int\>: The degree of polynomial kernel to use in the method. For higher dimensional embedding, use higher kernel degrees. As a rule of thumb, it should be proportional to the log of dimensionality. (default: 1 which equals to linear kernel)
    * -vocab \<file\>: Specifies the input vocabulary file. This vocabulary file is the output of `vocab_count`.
    * -input \<file\>: Specifies the input PMI matrix file. This PMI matrix file is the output of `pmi`.
    * -output \<file\>: Specifies the output embedding file. The resulting word vectors will be stored in this file.

## Pre-trained word vectors

We have run our algorithm on Wikipedia dump of 20160305 and the pre-trained word vectors file `kubwe_wikipedia20160305_d100` which contains the first 70,000 frequent words can be downloaded at the following link. Our trained word vectors contain 163,188 words but because of file size limitation on GitHub we removed the bottom infrequent ones.

[https://github.com/behrouzhs/kubwe/raw/master/kubwe_wikipedia20160305_d100.zip](https://github.com/behrouzhs/kubwe/raw/master/kubwe_wikipedia20160305_d100.zip)

## License

MIT License

