<p align="center">
<h3 align="center">DeepFix-Lite</h3>
<div align="center">
<p>Deep learning for automatically fixing single-line syntax errors in C programs</p>


</div>

------------------------------------------
Designed, implemented and evaluated a deep learning based solution to fix syntactic errors in C programs. The programs were written by students in
introductory programming assignments and contain single-line errors.

</div>

------------------------------------------
### Goal :
The goal of this assignment is to design, implement and evaluate a deep learning based
solution to fix syntactic errors in C programs. The programs were written by students in
introductory programming assignments and contain single-line errors. Completing this
assignment will require preparing the data for deep learning, implementing a deep learning
model for sequence-to-sequence translation (that is, from a buggy program to its fix), and
training and evaluating the deep learning model.

### Problem Statement :

We were provided two CSV files `train.csv` and `valid.csv` which respectively contain training and validation data. Each row in these CSV files contains the following fields/columns:

a) `Unnamed: 0` = the row number

b) `sourceText` = the buggy C program

c) `targetText` = the fixed C program

d) `sourceLineText` = code on the buggy line in sourceText

e) `targetLineText` = code on the fixed line in targetText

f) `lineNums_Text` = line number (starting from 1) of the buggy line in sourceText

g) `sourceTokens` = a list of token-lists where each token-list gives the tokens
corresponding to the respective line of code in sourceText

h) `targetTokens` = a list of token-lists where each token-list gives the tokens
corresponding to the respective line of code in targetText

i) `sourceLineTokens` = a list of tokens corresponding to sourceLineText

j) `targetLineTokens` = a list of tokens corresponding to targetLineText



I had to train a `sequence-to-sequence` neural network to
map `sourceLineTokens` to `targetLineTokens`, that is, from the token sequence of only the buggy line in the input program (sourceLineText) to the corresponding
fixed line (targetLineText).

------------------------------------------

### Model Performance : 
When I say m% token-level accuracy is achieved it means m% tokens are matched from the output produced by network.

I have used three different model architecture-
- LSTM 
- Bidirectional-LSTM 
- Bidirectional-LSTM with Attention 

Model name | LSTM | Bidirectional-LSTM | Bidirectional-LSTM with Attention 
--- | --- | --- | --- 
Accuracy | 91.10% | 93.20% | 96.0% 


### Tips :
I am providing some tips for you to get started. These are by no means exhaustive.
1) A neural network cannot be directly fed in with code (or in general, text). Thw provided data is already
tokenized. 
2) You can use pandas to inspect and analyze the CSV files. The field 'sourceTokens' (and
'targetTokens') is organized into a list of token-lists for your convenience where each
sublist corresponds to a line of code text. Note that in the CSV file, these nested lists will be stored as
strings. So to convert them to lists, you have to use ast.literal_eval().
3) You need to convert a token sequence into a sequence of indices into a vocabulary. For
this, you need to build a vocabulary by first iterating over your training data and
collecting all unique tokens.
4) You need to use some special tokens to indicate the end of an input sequence and the
start of an output sequence. You will have to fix the lengths of input and output
sequences when you are building the neural network. Longer sentences should be
truncated and shorted ones should be padded. It is therefore essential to create special
tokens to indicate these cases and reserve indices for them carefully. For example, 
the PAD_Token, SOS_Token and EOS_Token , they correspond to the token
used for padding shorter sequences, and to indicate start and end of sequences
respectively.
5) Once you create the vocabulary, you will know how many unique tokens appear in your
training data. Each token (index) will be mapped to a vector commonly called an
embedding in your neural network. With limited training data, learning good token
embeddings for all the tokens may not be feasible. It is therefore common to restrict the
vocabulary to top-k most frequent tokens for some value of k that you can decide
through experimentation (e.g., 250, 500, 1000, etc.). The rest of the tokens should be
mapped to an out-of-vocabulary token. For the purposes of uniformity, you should use
“OOV_Token” as the string representation of the OOV_Token. The vocabulary class at
this link does not reserve any index for OOV_Token, but you will need to do that.
6) We discussed “normalization” in one of the classes. It means instead of using the
original variable names, you map them to some predefined space of variable names. For
example, all integer variables could be mapped to VAR_INT_1, VAR_INT_2, etc. In fact,
you can simplify this further and ignore the type and just rename variables to VAR_1,
VAR_2, etc. Using the number of variables you typically observe in your training data,
you can decide on the bound on such variables. The advantage is that your vocabulary
now is much smaller and fixed -- it will include any language-specific tokens (e.g.,
opening and closing braces or parentheses, types such as int, char, etc.) plus the fixed
number of normalized variable names. In comparison, without normalization, your
vocabulary will have real variable names seen in the training data (e.g., x, my_function,
etc.). Normalized variable names will reduce OOV cases (which usually impact the
model accuracy) and would result in smaller vocabularies (e.g., considering that the
programs you are dealing with are small, it might suffice to have 50 normalized
variables). This can improve your model accuracy. However, the downside is that you
have to map from the actual program variables to the normalized variables during
training. Further, at test time, you need to do the same on the input program and then
map the normalized variables back to the original names (because the normalized
vocabulary is internal to your model). You have to ensure that if a variable x is mapped
to say VAR_1, then all occurrences of x should be renamed to VAR_1, and the other way
round (at the test time). This forward and reverse mapping code is the engineering cost.
It is upto you to decide whether you will stick to the original program vocabulary and
truncate it to some fixed length or use a normalized vocabulary.
7) During training, your neural network should take (x, y) pairs as examples, where x is the
token sequence for the buggy program or buggy program-line, depending on which
problem statement you are targeting. The sequence y is the expected output sequence,
that is, the tokenized version of the fixed line. You need to map the tokens to their
indices using the vocabulary. You can take a look at this tutorial - https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html to get started on
designing your neural network.
8) It is common to use teacher-forcing to train a sequence-to-sequence model. See the
discussion about this in the tutorial and elsewhere on the internet.
9) You should use the training data from train.csv to train your model and the validation
data from valid.csv to select hyper-parameters and checkpoints. Checkpoints are a
snapshot of the weights of all the layers of your neural network during some point of time
in your training procedure. Usually, checkpoints are saved after each epoch or some
interval of epochs (e.g., every 10th epoch). An epoch refers to one iteration through your
entire training dataset. You can train your model for as many epochs as you like.
However, more training epochs does not imply better performance because the model
may start to overfit your training data. So you should use the validation set to monitor
how the model behaves after each epoch and stop if the model starts to show worse
results on the validation set. Typically, you should see both training and validation loss
(or accuracy) decrease to ensure that your training progressive properly (of course,
some fluctuations can happen but the general trend should be decreasing).
10) You are free to use any neural architecture that can deal with sequences. These are
usually recurrent neural networks. Common recurrent neural networks are GRUs or
LSTMS (with and without attention), Transformers, etc. You can select any number of
layers. The dimensions of your layers are also up to you to decide. A small network may
not be able to model the training distribution properly, but a larger model is not always
better because it may overfit the training data. Again, you can use the validation data to
make these choices.

### Tips for run the code :
To run the inferrence code file -

I have prepared a demo script model_name_test.py. It can be invoked as follows
python  model_name_test.py <input-csv-file> <output-csv-file>

- This command will generate an output file output-csv-file, that includes the inputs from the input-csv-file(i.e. test.csv) and a new column fixedTokens 
  corresponding to your model prediction. 

I have already evaluted Bidirectional-LSTM model on 'test.csv' file and 'test_output.csv' can be found in root directory.

</div>

------------------------------------------

### Future Work :
- One can try `attention-transformers` based RNN architectures for this problem.
- Better tokenization mechansism for C code.
- Better representations / Embeddings for code.
- Better initialisations / biases for network.  

------------------------------------------
### Acknoledgements :

- Major ideas of this work are derived from [DeepFix](http://www.iisc-seal.net/deepfix).
   
