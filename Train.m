%  categories = {'airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'};
% rootFolder = 'C:\Users\30338304\Desktop\Project PhD\00 Phd\datasets\cifar10_dct_trainingSet';
% imds = imageDatastore(fullfile(rootFolder, categories),'LabelSource', 'foldernames');
% cifar10Data = tempdir;
%
% url = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';
%
% helperCIFAR10Data.download(url,cifar10Data);
%
% [trainingImages,trainingLabels,testImages,testLabels] = helperCIFAR10Data.load(cifar10Data);
%categories = {'10','1','2','3','4','5','6','7','8','9'};

%
%categories = {'airplane','automobile','bird','frog','ship','truck'};
categories = {'airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'};
rootFolder = 'C:\Users\mthossain\OneDrive - Federation University Australia\Desktop\cifar10\cifar 10 together';

imds = imageDatastore(fullfile(rootFolder, categories),'LabelSource', 'foldernames');


%Normal ResNet with LP-ReLU
k = 1;
netWidth = 16;
layers = [
    imageInputLayer([32 32 3],'Name','input')
    convolution2dLayer(3,netWidth,'Padding','same','Name','convInp')
    batchNormalizationLayer('Name','BNInp')
    %reluLayer('Name','reluInp')
    LPrelu_1_Layer(netWidth,'reluInp')
    
    convolutionalUnit(netWidth,1,'S1U1')
    additionLayer(2,'Name','add11')
    %reluLayer('Name','relu11')
    LPrelu_1_Layer(netWidth,'relu11')
    convolutionalUnit(netWidth,1,'S1U2')
    additionLayer(2,'Name','add12')
    %reluLayer('Name','relu12')
    LPrelu_1_Layer(netWidth,'relu12')
    
    convolutionalUnit(k*netWidth,2,'S2U1')
    additionLayer(2,'Name','add21')
    %reluLayer('Name','relu21')
    LPrelu_1_Layer(k*netWidth,'relu21')
    convolutionalUnit(k*netWidth,1,'S2U2')
    additionLayer(2,'Name','add22')
    %reluLayer('Name','relu22')
    LPrelu_1_Layer(k*netWidth,'relu22')
    
    convolutionalUnit(2*k*netWidth,2,'S3U1')
    additionLayer(2,'Name','add31')
    %reluLayer('Name','relu31')
    LPrelu_1_Layer(2*k*netWidth,'relu31')
    
    convolutionalUnit(2*k*netWidth,1,'S3U2')
    additionLayer(2,'Name','add32')
    %reluLayer('Name','relu32')
    LPrelu_1_Layer(2*k*netWidth,'relu32')
    
    averagePooling2dLayer(8,'Name','globalPool')
    fullyConnectedLayer(10,'Name','fcFinal')
    
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')
    ];

lgraph = layerGraph(layers);
% figure('Units','normalized','Position',[0.2 0.2 0.6 0.6]);
% plot(lgraph);
lgraph = connectLayers(lgraph,'reluInp','add11/in2');
lgraph = connectLayers(lgraph,'relu11','add12/in2');

% figure('Units','normalized','Position',[0.2 0.2 0.6 0.6]);
% plot(lgraph);

skip1 = [
    convolution2dLayer(1,k*netWidth,'Stride',2,'Name','skipConv1')
    batchNormalizationLayer('Name','skipBN1')];
lgraph = addLayers(lgraph,skip1);
lgraph = connectLayers(lgraph,'relu12','skipConv1');
lgraph = connectLayers(lgraph,'skipBN1','add21/in2');

lgraph = connectLayers(lgraph,'relu21','add22/in2');

skip2 = [
    convolution2dLayer(1,2*k*netWidth,'Stride',2,'Name','skipConv2')
    batchNormalizationLayer('Name','skipBN2')];
lgraph = addLayers(lgraph,skip2);
lgraph = connectLayers(lgraph,'relu22','skipConv2');
lgraph = connectLayers(lgraph,'skipBN2','add31/in2');

lgraph = connectLayers(lgraph,'relu31','add32/in2');

% figure('Units','normalized','Position',[0.2 0.2 0.6 0.6]);
% plot(lgraph)

numUnits = 9;
netWidth = 16;
%lgraph = residualCIFARlgraph(netWidth,numUnits,"standard");


% figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
% plot(lgraph)
miniBatchSize = 128;
learnRate = 0.1*miniBatchSize/128;
%valFrequency = floor(size(XTrain,4)/miniBatchSize)  'L2Regularization',.0005, ...
options = trainingOptions('sgdm', ...
    'Momentum', .9, ...
    'InitialLearnRate',.1, ...
    'L2Regularization', .0005, ...
    'MaxEpochs',160, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',50);

lgraph_WRN = myresidualCIFARlgraph(16,18,"standard",2);
WRN_40_2 = trainNetwork(imds, lgraph_WRN, options);

rootFolder1 = 'C:\Users\mthossain\OneDrive - Federation University Australia\Desktop\cifar10\test_batch';
imds_test = imageDatastore(fullfile(rootFolder1, categories),'LabelSource', 'foldernames');


labels = classify(WRN_40_2, imds_test);
confMat = my_confusionmat(imds_test.Labels, labels);
confMat = confMat./sum(confMat,2);
mean(diag(confMat))


function layers = convolutionalUnit(numF,stride,tag)
layers = [
    convolution2dLayer(3,numF,'Padding','same','Stride',stride,'Name',[tag,'conv1'])
    batchNormalizationLayer('Name',[tag,'BN1'])
    %reluLayer('Name',[tag,'relu1'])
    LPrelu_1_Layer(numF,[tag,'relu1'])
    convolution2dLayer(3,numF,'Padding','same','Name',[tag,'conv2'])
    batchNormalizationLayer('Name',[tag,'BN2'])];
end