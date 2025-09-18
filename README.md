## Project path 

Need to run 
    export KAGGLEHUB_CACHE=$PWD/data/brazilian_lit

since kaggle doesnt have parameter to determine path to download dataset in user machine. 

### Tokenizer

The Tokenizer is an essential component, which creates a vocabulary for the model. Since computers only understands numbers, it is crucial to transform text into numbers. The tokenizer uses a technique called

BPE (Byte-Pair Encoding) which "learns" the most frequent sequences in the training dataset and transforms them into tokens(numbers), which the main model will use to make sense of them.

It is important to notice that this step only produces the tokens, we still need to implement a way to determine the position of each word in a sentence, since the position of each word can change the meaning of

the sentence.



### Positional encoder 

It introduces the concept of word position in a sentence. #TODO: 



### Transformer architecture

The main component of this project, which uses the concept of self-attention to improve model performance. #TODO: 



### Training loop

It is the loop to train the model to understand text. #TODO





### Inference and text generation





## Chat application
