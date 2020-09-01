<h1 align=center>AbTextSumm</h1>
Abstractive Summarization: Code of the ILP-based algorithm similar to the IJCAI paper: Multi-document Abstractive Summarization using ilp based multi-sentence compression. Some differences exist as pointed below:

**Please note that this code only tackles the summarization component and not the clustering part. Also, the code for the original paper (mentioned below) was written in JAVA and this is Python. I tried to reconstruct most of the technique, but there might be subtle differences in the evaluation results if you use this version.**

The code takes a list of sentences, or a paragraph and produces an extractive or abstractive summary driven by the parameter "mode".
This code was also used for a part of the work of this paper:

K. Rudra, S. Banerjee, N. Ganguly, P. Goyal, M. Imran, and
P. Mitra, “Summarizing situational tweets in crisis scenario,” in Proceedings of the 27th ACM Conference on Hypertext and
Social Media. ACM, 2016, pp. 137–147.


For language model (only required for abstractive summarization):
Needs kenlm: https://kheafield.com/code/kenlm/ [See how to install]
Use any available ARPA format language model and convert to kenlm format as binary. KENLM is really fast. 

Other several packages required: PuLP for optimization, sklearn, nltk, cpattern, igraph
Best option is to use Anaconda package. All the above mentioned packages can be installed using pip.

##Install

On a Debian system, first install sqlclient library using:
```
apt install libmysqlclient-dev
```

Then to install python dependencies, use:
```
pip install - r requirements.txt
```
in the root folder of the project. 

A major part of the word graph generation code has been taken from https://github.com/boudinfl/takahe.

The main program is Example.py.

Before running the demo download a language model.
The 3-gram models from [here](https://www.keithv.com/software/giga/) perform
well.

Given a passage, it can generate a summary using the following code:
```
  list_Sentences=segmentize(passage)
  generateSummaries(list_Sentences, mode="Extractive")
```
Changing the mode = "Extractive" to:
```
mode="Abstractive"
```
will run Abstractive summarization with TextRank as the default ranking parameter. However, it requires a language model described earlier. By default, this code runs extractive summarization. You can also use the length parameter (in words) to control length of the output summary. For example:

```
generateSummaries(list_Sentences, mode="Extractive", length=50)
```

Note: The code here does not contain the clustering step (mentioned in the paper), which should be pretty straightforward to implement. 
This is research quality code, but if you find major bugs, please let me know.

**If you use the code here, please cite the paper:**

Siddhartha Banerjee, Prasenjit Mitra, and Kazunari Sugiyama. _"Multi-Document Abstractive Summarization Using ILP based Multi-Sentence Compression."_ Proceedings of the 24th International Joint Conference on Artificial Intelligence (IJCAI 2015), Buenos Aires, Argentina. 2015.
