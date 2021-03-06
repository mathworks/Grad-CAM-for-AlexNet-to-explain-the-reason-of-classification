# Grad-CAM-for-AlexNet-to-explain-the-reason-of-classification

![result image](https://jp.mathworks.com/matlabcentral/mlc-downloads/downloads/f45c673d-1cd7-4047-9f1a-7cad68c22b72/7e70e809-9956-4d1f-b49d-349bb61b10e5/images/screenshot.JPG)


## Overview
Class Activation Mapping(CAM) is a good method to explain why the model classify the object as that.
https://jp.mathworks.com/matlabcentral/fileexchange/69357-class-activation-mapping
But network models which can be applied for CAM are limited.
Grad-CAM is the method to generalize CAM to work with many kinds of networks.

Through this demo, you can learn workflow from retraining model(AlexNet) to applying Grad-CAM on it.

[Japanese]
CNNを用いたディープラーニングによる分類の判定精度は非常に高く、多くの領域での画像自動判定に利用されています。一方で、内部がブラックボックスで「なぜその判定になったのかわからない」点に不安を感じる方もいます。Class Activation Mapping(CAM)は判定要因の可視化に非常に便利ですが、適用できるネットワークに制限があります。

Grad-CAMはGradietを利用して任意のネットワーク・層でCAMを一般化した方法です。
このサンプルでAlexNetでの転移学習からGrad-CAMの適用までのコードを確認できます。


## Paper
Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.
Ramprasaath R. Selvaraju, etc
https://arxiv.org/abs/1610.02391


Copyright 2019-2020 The MathWorks, Inc. 

[![View Grad-CAM for AlexNet to explain the reason of classification on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://jp.mathworks.com/matlabcentral/fileexchange/72850-grad-cam-for-alexnet-to-explain-the-reason-of-classification)
