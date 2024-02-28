# AI-dam-mickiewicz
AI-dam-mickiewicz generate text in style of "Pan Tadeusz" based on deep neutral networks

This repository has been created only for educational purposes.

Repository provide training, testing and sampling from few architectures for generating text. Models have the possibility to save them and load from checkpoints. Code is simple and easy to read for everyone. Results are not quite well, because of small length of text and restricted time usage in gpu's. 
Available architectures:

- Decoder only Transformer (GPT)
- RNN (simple Recurrent Neural Network)
- N-Gram NN-style
- BiGram classic-style

TO DO:
- GRU
- LSTM
- Mamba

Repository contains only character level language, but in future i will try to implement Byte-Pair encoding. It can influence in bad results too.

References : 

https://github.com/karpathy/nanoGPT

https://github.com/lucidrains/

https://nlp.seas.harvard.edu/2018/04/03/attention.html
