% imageSize = [28 28 1];
% pixelRange = [-4 4];
% imageAugmenter = imageDataAugmenter( ...
%     'RandXReflection',true, ...
%     'RandXTranslation',pixelRange, ...
%     'RandYTranslation',pixelRange);
% augimdsTrain = augmentedImageDatastore(imageSize,imgDataTrain,labelsTrain, ...
%     'DataAugmentation',imageAugmenter, ...
%     'OutputSizeMode','randcrop');

%% Prepare the dataset

% The MNIST dataset is a set of handwritten digits categorized 0-9 and is
% available at http://yann.lecun.com/exdb/mnist/.

% The following line will download (if necessary) and prepare the dataset
% to use in MATLAB.
%[imgDataTrain, labelsTrain, imgDataTest, labelsTest] = prepareData;

%% Need a starting point? Check the documentation!
% search "deep learning"
%web(fullfile(docroot, 'nnet/deep-learning-training-from-scratch.html'))
        
%% Attempt 1: Set training options and train the network
% layers1 = [
%     imageInputLayer([28 28 1])  %,'Normalization','rescale-zero-one'
%     
%     convolution2dLayer(3,8,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,16,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,32,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     fullyConnectedLayer(10)
%     softmaxLayer
%     classificationLayer];
%     
%     miniBatchSize = 512;

  categories = {'0','1','2','3','4','5','6','7','8','9'};
  rootFolder = 'E:\00 PhD\DataSets\MNISTbased\mnistasjpg\trainingSet\trainingSet' 
  imds = imageDatastore(fullfile(rootFolder, categories),'LabelSource', 'foldernames');

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',45, ...
    'Shuffle','every-epoch', ...
    'ValidationFrequency',30, ...
    'MiniBatchSize',miniBatchSize, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% Attempt 3: Change the network architecture

layers = [
    imageInputLayer([28 28 1])  %,'Normalization','rescale-zero-one'
    
    convolution2dLayer(3,8,'Padding','same')
    %batchNormalizationLayer
    LPrelu_1_Layer(8,'a1')
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    %batchNormalizationLayer
    LPrelu_1_Layer(16,'a2')
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    %batchNormalizationLayer
    LPrelu_1_Layer(32,'a3')
    
    fullyConnectedLayer(2)
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

    Lenet_only2_FC = trainNetwork(imgDataTrain, labelsTrain,layers, options);

