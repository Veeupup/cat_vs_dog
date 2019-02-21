# dog vs cat
A neural network for a Kaggle's competition.

It is written according to the [article](https://zhuanlan.zhihu.com/p/29024978).

You can download the train data and test data from [kaggle](https://www.kaggle.com/c/dogs-vs-cats/data)
And then put the train dataset and test dataset in the same folder.

You can train it and test it by the steps.
And the start up is written in fire and you can train it by shell.

> simple_rcnn.py is a simple rcnn.
> it can not just run because of the data format.

Install.
```shell
pip install -r requirements.txt
```

Start the visdom.
```shell
python -m visdom.serve
```

Train
```shell
python main.py train --train-data-root=./data/train --use-gpu=False --env=classifier
```

Help
```
python main.py help
```

Test
```
python main.py test --data-root=./data/test --use-gpu=False --batch-size=256
```
