# Select Additive Learning
Implementation for the paper [Select-Additive Learning: Improving Generalization in Multimodal Sentiment Analysis](https://arxiv.org/abs/1609.05244)

Extracted features used for the paper is here: [Multimodal Sentiment Analysis data set](http://www.cs.cmu.edu/~haohanw/SAL.html). 

Examples are showed with Verbal Modality

THEANO VERSION: 0.8.2

The main method is at `model/run.py`

Make sure you run `pretrainModel/model.py` before you run `model/run.py`. Then paremeters of the pre-trained model will be written into the file `params/pretrain/videoModelParams.npy` to be used by `model/run.py`.

    Wang, Haohan, Aaksha Meghawat, Louis-Philippe Morency, and Eric P. Xing. "Select-Additive Learning: Improving Generalization in Multimodal Sentiment Analysis." arXiv preprint arXiv:1609.05244 (2016).
