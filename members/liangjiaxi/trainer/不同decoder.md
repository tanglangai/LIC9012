
+ 最初的模型结构
```
 "seq2seq_encoder": {
            "type": "intra_sentence_attention",
            "input_dim": 220,
            "output_dim": 220,

            "combination": "1,2"

        },
        "sentence_encoder": {
            "type": "lstm",
            "input_size": 220,
            "hidden_size": 110,
            "bidirectional": true
        },
//        "seq2seq_encoder": {
//            "type": "gated-cnn-encoder",
//            "input_dim": 170,
//            "layers": [ [[4, 170]], [[4, 170], [4, 170]], [[4, 170], [4, 170]]]
//        },

        "matrix_attention": {
            "type": "bilinear",
            "matrix_1_dim": 440,
            "matrix_2_dim": 440,
            "label_dim": 51,
            "use_input_biases": true,
            "activation": null
        }
    },
```
+ 结果

```
ssh://liangjx@218.17.122.106:9111/home/liangjx/anaconda3/bin/python3.6 -u /home/liangjx/teamextraction/members/liangjiaxi/trainer/train_decoder.py
Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.
5612it [00:33, 166.28it/s]
100%|##########| 635974/635974 [00:05<00:00, 121625.53it/s]
precision: 0.0156, recall: 0.0014, f1_score: 0.0025, loss: 3.4813 ||: : 44it [02:37,  3.52s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:13,  1.62s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [02:40,  3.47s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:10,  1.53s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [02:38,  3.49s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:10,  1.63s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [02:39,  3.63s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:11,  1.52s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [02:38,  3.49s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:12,  1.60s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [02:38,  3.59s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:10,  1.51s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [02:36,  3.50s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:10,  1.51s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [02:36,  3.36s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:11,  1.62s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [02:36,  3.40s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:12,  1.54s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [02:38,  3.51s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:11,  1.66s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [02:35,  3.52s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:09,  1.55s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [02:37,  3.51s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:13,  1.63s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [02:36,  3.33s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:10,  1.59s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [02:37,  3.62s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:08,  1.49s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [02:39,  2.74s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:47,  1.02s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [02:07,  2.89s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:47,  1.02s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [02:06,  2.81s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:47,  1.02s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [02:04,  2.71s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:47,  1.03s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [02:00,  2.55s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:46,  1.03s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:56,  2.53s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:47,  1.04s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:56,  2.53s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:46,  1.03s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:55,  2.51s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:46,  1.02s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:56,  2.56s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:46,  1.04s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:55,  2.56s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:47,  1.02s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:56,  2.59s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:47,  1.02s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:56,  2.57s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:46,  1.02s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:56,  2.61s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:47,  1.05s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:56,  2.55s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:46,  1.02s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:56,  2.56s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:47,  1.04s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:57,  2.57s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:47,  1.07s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:57,  2.54s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:47,  1.11s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:56,  2.52s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:47,  1.05s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:56,  2.53s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:47,  1.06s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:57,  2.54s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:47,  1.09s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:56,  2.54s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:46,  1.02s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:58,  2.63s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:46,  1.02s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:56,  2.52s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:47,  1.02s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:56,  2.67s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:46,  1.02s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:57,  2.54s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:46,  1.01s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:56,  2.57s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:46,  1.02s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:56,  2.57s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:46,  1.04s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:56,  2.59s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:47,  1.04s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:56,  2.53s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:46,  1.05s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:56,  2.55s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:47,  1.02s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:57,  2.54s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:47,  1.03s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:58,  2.63s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:47,  1.05s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [01:56,  2.56s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:46,  1.03s/it]

```


+ 去掉matrix attention，换成全连接
```angular2html
ssh://liangjx@218.17.122.106:9111/home/liangjx/anaconda3/bin/python3.6 -u /home/liangjx/teamextraction/members/liangjiaxi/trainer/train_decoder.py
Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.
5612it [00:29, 191.16it/s]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4835 ||: : 44it [00:47,  1.05s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:21,  2.04it/s]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:48,  1.06s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:21,  2.13it/s]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:49,  1.11s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:21,  2.10it/s]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:48,  1.09s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:21,  2.12it/s]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:49,  1.08s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:21,  1.98it/s]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:53,  1.22s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:22,  1.99it/s]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:53,  1.17s/it]
precision: 0.0000, recall: 0.0000, f1_score: 0.0000, loss: 3.4650 ||: : 44it [00:22,  1.98it/s]


```