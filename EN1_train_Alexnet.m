clear; close all; imtool close all; clc;rng('default')

%% Create Datastore
% categ = {'Good', 'bad'};
imds = imageDatastore('images', 'IncludeSubfolders',true,'LabelSource', 'foldernames')
% imageBrowser(imds)
imds.ReadFcn = @(filename) readFunctionTrain(filename);
%% Augmentation
augmenter = imageDataAugmenter('RandXReflection',true,'RandYReflection',true,...
    'RandRotation', [-180 180]);
datasource = augmentedImageSource([227 227],imds,'DataAugmentation',augmenter);

%% Training the Network
net = alexnet();

layers = net.Layers;
layers(23) = fullyConnectedLayer(2); 
layers(25) = classificationLayer
% Training options 
options = trainingOptions('sgdm','MaxEpochs',200, ...
    'InitialLearnRate',0.001,...
    'Plots','training-progress'); 
convnet = trainNetwork(datasource,layers,options);  
save('myalexnet.mat','convnet');

%% Test the network
[YTest scores] = classify(convnet,imds);
TTest = imds.Labels;
accuracy = sum(YTest == TTest)/numel(YTest)

%%
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

%%

function I = readFunctionTrain(filename)
% Resize the flowers images to the size required by the network.
I = imread(filename);
if ismatrix(I)          
    I = cat(3,I,I,I);
end
I = imresize(I, [227 227]);
end

%%  Copyright 2019 The MathWorks, Inc.