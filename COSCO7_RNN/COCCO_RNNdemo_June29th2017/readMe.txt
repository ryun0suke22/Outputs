[learning model]
mt.py : encoder-decoder model for training and make models(mt-xx.model) each epoc.

[prediction model]
test-mt.py : this module uses learning model and predict the test-data from jp-test.txt

[tuning data]
jp.txt contains 500 sentances written in japanese.
eng.txt	contains 500 sencentances written in english. (correspond with every sentances in jp.txt)

[test data]
jp-test.txt : contains 10 sentances written in japanese.

First you have to run the learning model(mt.py)
After finish learning,
You can run test-mt.py

Since the learning model(mt-xx.model) is too large, they were removed. 


**** Requirements is as following
Python 2.7
Chainer.__version__ == 1.10.0


If you have questions or unclear, please tell me by e-mail(m5211126@u-aizu.ac.jp)

Ryunosuke Murakami  // 2017/07/05