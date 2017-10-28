# -*- coding: utf-8 -*-
class Config(object):
	'''
	lstm config:
	Speed: 61566 wps
	Train Perplexity: 43.167
	Valid Perplexity: 116.618
	Test Perplexity: 111.888
	'''
    # init_scale = 0.1
    # learning_rate = 1.0
    # max_grad_norm = 5
    # num_layers = 2
    # slice_size = 20
    # hidden_size = 200
    # max_epoch = 13
    # keep_prob = 1.0
    # lr_const_epoch = 4
    # lr_decay = 0.5
    # batch_size = 20
    # vocab_size = 10000
    # rnn_model = "lstm"
    # data_path = "../simple-examples/data/"
    # save_path = "../out/cudnn/lstm/"

	'''
	gru config:
	Speed: 76442 wps
	Train Perplexity: 51.346
	Valid Perplexity: 111.628
	Test Perplexity: 106.113
	'''
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    slice_size = 30
    hidden_size = 200
    max_epoch = 13
    keep_prob = 0.8
    lr_const_epoch = 4
    lr_decay = 0.7
    batch_size = 30
    vocab_size = 10000
    rnn_model = "gru"
    data_path = "../simple-examples/data/"
    save_path = "../out/cudnn/gru/"
