clear; close all; imtool close all; clc;rng('default')

%% 画像セットを指定
imds = imageDatastore('images', 'IncludeSubfolders',true,'LabelSource', 'foldernames')
% imageBrowser(imds)
imds.ReadFcn = @(filename) readFunctionTrain(filename);
%% 学習時に適用する画像拡張を指定
augmenter = imageDataAugmenter('RandXReflection',true,'RandYReflection',true,...
    'RandRotation', [-180 180]);
datasource = augmentedImageSource([227 227],imds,'DataAugmentation',augmenter);

%% ネットワークを学習
net = alexnet();

layers = net.Layers;
layers(23) = fullyConnectedLayer(2); 
layers(25) = classificationLayer
% オプションの設定 
options = trainingOptions('sgdm','MaxEpochs',200, ...
    'InitialLearnRate',0.001,...
    'Plots','training-progress'); 
convnet = trainNetwork(datasource,layers,options);  
save('myalexnet.mat','convnet');

%% ネットワークをテストします
[YTest scores] = classify(convnet,imds);
TTest = imds.Labels;
accuracy = sum(YTest == TTest)/numel(YTest)

%% 正常のナット、不良のナットそれぞれの分類結果を表示
idx = 100;
img = readimage(imds, idx);
result = classify(convnet,img);
subplot(1,2,1),imshow(img)
title(string(result),'FontSize',20);
idx2 = 600;
img2 = readimage(imds, idx2);
result2 = classify(convnet,img2);
subplot(1,2,2),imshow(img2)
title(string(result2),'FontSize',20);shg

%% ネットワークの入力サイズに合うように画像サイズを修正する補助関数

function I = readFunctionTrain(filename)
% Resize the flowers images to the size required by the network.
I = imread(filename);
if ismatrix(I)          
    I = cat(3,I,I,I);
end
I = imresize(I, [227 227]);
end

%%  Copyright 2019 The MathWorks, Inc.