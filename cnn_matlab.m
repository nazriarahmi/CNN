clear;
clc;

imageFolder = 'C:\Users\user\Documents\Kuliah\MatlabCNNModify\Binarydataset\After2_1'
imds = imageDatastore(imageFolder, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.2,'randomize');

layers = [
    imageInputLayer([100 100 1])
    
    convolution2dLayer(3,8)
    %batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2)
    
    convolution2dLayer(3,16)
    %batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2)
    
    convolution2dLayer(3,32)
    %batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

opts = trainingOptions('sgdm',...
      'LearnRateSchedule','piecewise',...
      'LearnRateDropFactor',0.2,... 
      'LearnRateDropPeriod',5,... 
      'MaxEpochs',20,... 
      'ExecutionEnvironment','cpu',... 
      'MiniBatchSize',300);


net = trainNetwork(imdsTrain,layers,opts);

YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)
