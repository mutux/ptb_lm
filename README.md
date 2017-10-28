# ptb_word_lm
A lot of TensorFLow **beginners** feel that [the code](https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py) of the rnn language model supporting [the official tutorial](https://www.tensorflow.org/tutorials/recurrent) is rather obscure. It has actually stopped some people from diving deeper into deep learning. As one of them, I also spent a lot of time to figure out what the code is trying to say, since it's quite different from what they describe in the tutorial.

So after read through the codes, I **reorganized** the code into several pieces and **rename** a couple of confusing variables to make the code more understandable.

Before diving into the code for a beginner like me, there are some tricks and concepts need to pay attention to, which they used without explanation.
```
 - tf.train.range_input_producer
 - tf.contrib.cudnn_rnn.CudnnLSTM
 - tf.train.Supervisor.managed_session
 - epoch
 - batch
 - slice
```
## Explanation
 ### tf.train.range_input_producer
 While producing data slice to feed the model, the tutorial use this special **producer**. It's special because it will generate a QueueRunner and add it to the current Graph silently. This QueueRunner will start a separate thread managing a data queue like a server to for the main thread to request data slices from it iteratively. That's why it has no explict data slice feeding operations in the code, and without explanation, it makes the procedure confusing.

 ### tf.contrib.cudnn_rnn.CudnnLSTM
 CudnnLSTM is used for computing LSTM on GPUs, there's also a similar one named CudnnGRU. They are almost the same, except that LSTM use two memory variables c and h, it will return them as a tuple LSTMStateTuple(h=h, c=**c**), while GRU only need one memory variable h, it will return it as a Tensor h, in order to make to code compatible, I force it to return a similar tuple structure as LSTMStateTuple(h=h, c=**h**).
 Unlike to basic/block LSTM, both of them do **not** need you to explictly chain cells and/or layers up to buid the Graph, the operators will do it automatically with parameters like `num_layers`, `hidden_size` specified.

 ### tf.train.Supervisor.managed_session()
 tf.train.Supervisor is a small wrapper around a SessionManager, a Saver and a Coordinator to take care of the common needs of TensorFlow program during training process.
 Within the `with sv.managed_session()` block, all variables in the graph are initialized, and services are started to save checkpoint of the model and summaries to the log file.

### Some Concepts
#### epoch
Training process is consisted with multiple epochs, during each epoch, training process with consume the whole training dataset once *(in this case, no guarantee in other cases)*.

#### batch
The whole training dataset is a single big text file. A unique word is firstly given a unique integer as its index, then the file is transformed into a single row vector, with each element holding a index a word, the order of the index integer in the vector is exactly the same of that of the word in the text file.
During training process, we will feed the data in a batch-wise way to the program, so the vector are reshaped into a matrix with dimensions (batch_size, num_batch).

#### slice
Honestly, I have difficulty in understanding the concept of the epochs in the original code at the begining, partly because of the lack of knowledge on dnn training routines. Other reasons like not familiar with `range_input_producer` and the confusing naming conventions (`num_steps, epoch_size, batch_len, batch_size`) are also spikes in my feet. 
I decided to rename some fo the variables to lessen the confusion in my head, especially the ones associated with the concept epoch.


Training process will iteratively consume the data multiple times, each time can be called an epoch. During each epoch, in order to capture the long-distant dependencies, arbitrarily distant inputs should be fed to model, unfortunately, this will make the backpropagation computation difficult. In order to make the learning process tractable, a fixed number of inputs becomes common practice. The fixed number of inputs is actually a fix-sized slice of the batch matrix, say every 20 batches as a slice, then there would be `num_slices = num_batches/slice_size`. The `range_input_producer` will generate `num_slice` slices during each `epoch` of the training to exhaust the whole dataset, as is seen in the function `run_epoch`.


## The structure
Overall diagram looks like this
```
   --------------
  |     Input    |
   --------------
         |
         V
   --------------        --------------
  |  Lang_Model  | <--- |    Config    |
   --------------        --------------
         |
         v
   --------------
  |     Train    |
   --------------
```

## Run
### Note
This programs can only be run on machine with **GPUs** at the moment. You can change the code easily to make it work on CPUs too.


### Data
The data required for this tutorial is in the `data/` directory of the [PTB dataset](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz) from Tomas Mikolov's webpage.
Downlaod and unzip it.


### Config
Specify the `data_path` where you unzip the data in, `save_path` where you want the log files be put.


### Command
```
  python train.py
```


## Finally
Have fun!
