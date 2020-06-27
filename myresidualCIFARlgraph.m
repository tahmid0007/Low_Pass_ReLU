% lgraph = residualCIFARlgraph(netWidth,numUnits,unitType) creates a layer
% graph for CIFAR-10 data with residual connections.
%
% netWidth is the network width, defined as the number of filters in the
% first 3-by-3 convolutional layers of the network.
%
% numUnits is the number of convolutional units in the main branch of
% network. Because the network consists of three stages with the same
% number of convolutional units, numUnits must be an integer multiple of 3.
%
% unitType is the type of convolutional unit, specified as "standard" or
% "bottleneck". A standard convolutional unit consists of two 3-by-3
% convolutional layers. A bottleneck convolutional unit consists of three
% convolutional layers: a 1-by-1 layer for downsampling in the channel
% dimension, a 3-by-3 convolutional layer, and a 1-by-1 layer for
% upsampling. Hence, a bottleneck convolutional unit has 50% more
% convolutional layers than a standard unit, but only half the number of
% spatial 3-by-3 convolutions. The computational complexity of the two unit
% types are approximately the same, but the total number of features
% propagating in the residual connections is four times larger when using
% the bottleneck units. The total depth, defined as the maximum number of
% sequential convolutional and fully connected layers, is 2*numUnits + 2
% for standard units and 3*numUnits + 2 for bottleneck units.
%
% Example: lgraph = residualCIFARlgraph(16,9,"standard") creates a residual
% layer graph for CIFAR-10 data that has width 16 and 9 standard
% convolutional units.


function lgraph = myresidualCIFARlgraph(netWidth,numUnits,unitType,k)


% Check inputs
assert(numUnits > 0 && mod(numUnits,3) == 0 ,...
    "Number of convolutional units must be an integer multiple of 3.");
unitsPerStage = numUnits/3;

if unitType == "standard"
    convolutionalUnit = @standardConvolutionalUnit;
elseif unitType == "bottleneck"
    convolutionalUnit = @bottleneckConvolutionalUnit;
else
    error("Residual block type must be either ""standard"" or ""bottleneck"".")
end


%% Create Main Network Branch

%
% Input section. Add the input layer and the first convolutional layer.
layers = [
    imageInputLayer([32 32 3],'Name','input')
    convolution2dLayer(3,netWidth,'Padding','same','Name','convInp')
    batchNormalizationLayer('Name','BNInp')
    %reluLayer('Name','reluInp')];
    logreluLayer(netWidth,'reluInp')];
% Stage one. Activation size is 32-by-32.
for i = 1:unitsPerStage
    layers = [layers
        convolutionalUnit(netWidth,1,['S1U' num2str(i) '_'])
        additionLayer(2,'Name',['add1' num2str(i)])
        %reluLayer('Name',['relu1' num2str(i)])
        logreluLayer(netWidth,['relu1' num2str(i)])];
end

% Stage two. Activation size is 16-by-16.
for i = 1:unitsPerStage
    if i==1
        stride = 2;
    else
        stride = 1;
    end
    layers = [layers
        convolutionalUnit(2*k*netWidth,stride,['S2U' num2str(i) '_'])
        additionLayer(2,'Name',['add2' num2str(i)])
        %reluLayer('Name',['relu2' num2str(i)])
         logreluLayer(2*k*netWidth,['relu2' num2str(i)])];
    
end

% Stage three. Activation size is 8-by-8
for i = 1:unitsPerStage
    if i==1
        stride = 2;
    else
        stride = 1;
    end
    layers = [layers
        convolutionalUnit(4*k*netWidth,stride,['S3U' num2str(i) '_'])
        additionLayer(2,'Name',['add3' num2str(i)])
        %reluLayer('Name',['relu3' num2str(i)])
         logreluLayer(4*k*netWidth,['relu3' num2str(i)])];
end

% Output section.
layers = [layers
    averagePooling2dLayer(8,'Name','globalPool')
    fullyConnectedLayer(10,'Name','fcFinal')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];

lgraph = layerGraph(layers);


%% Add shortcut connections
% Add shortcut connection around the convolutional units. Most shortcuts
% are identity connections.
for i = 1:unitsPerStage-1
    lgraph = connectLayers(lgraph,['relu1' num2str(i)],['add1' num2str(i+1) '/in2']);
    lgraph = connectLayers(lgraph,['relu2' num2str(i)],['add2' num2str(i+1) '/in2']);
    lgraph = connectLayers(lgraph,['relu3' num2str(i)],['add3' num2str(i+1) '/in2']);
end

% Shortcut connection from input section to first stage. If unitType equals
% "bottleneck", then the shortcut connection must upsample the channel
% dimension from netWidth to netWidth*4.
if unitType == "bottleneck"
    skip0 = [
        convolution2dLayer(1,netWidth*4,'Stride',1,'Name','skipConv0')
        batchNormalizationLayer('Name','skipBN0')];
    lgraph = addLayers(lgraph,skip0);
    lgraph = connectLayers(lgraph,'reluInp','skipConv0');
    lgraph = connectLayers(lgraph,'skipBN0','add11/in2');
else
    lgraph = connectLayers(lgraph,'reluInp','add11/in2');
end

% Shortcut connection from stage one to stage two.
if unitType == "bottleneck"
    numF =  netWidth*2*4;
else
    numF =  netWidth*2*k;
end
skip1 = [convolution2dLayer(1,numF,'Stride',2,'Name','skipConv1')
    batchNormalizationLayer('Name','skipBN1')];
lgraph = addLayers(lgraph,skip1);
lgraph = connectLayers(lgraph,['relu1' num2str(unitsPerStage)],'skipConv1');
lgraph = connectLayers(lgraph,'skipBN1','add21/in2');

% Shortcut connection from stage two to stage three.
if unitType == "bottleneck"
    numF =  netWidth*4*4;
else
    numF =  netWidth*4*k;
end
skip2 = [convolution2dLayer(1,numF,'Stride',2,'Name','skipConv2')
    batchNormalizationLayer('Name','skipBN2')];
lgraph = addLayers(lgraph,skip2);
lgraph = connectLayers(lgraph,['relu2' num2str(unitsPerStage)],'skipConv2');
lgraph = connectLayers(lgraph,'skipBN2','add31/in2');

return


end

%%
% layers = standardConvolutionalUnit(numF,stride,tag) creates a standard
% convolutional unit, containing two 3-by-3 convolutional layers with numF
% filters and a tag for layer name assignment.
function layers = standardConvolutionalUnit(numF,stride,tag)
layers = [
    convolution2dLayer(3,numF,'Padding','same','Stride',stride,'Name',[tag,'conv1'])
    batchNormalizationLayer('Name',[tag,'BN1'])
    %reluLayer('Name',[tag,'relu1'])
     logreluLayer(numF,[tag,'relu1'])
    convolution2dLayer(3,numF,'Padding','same','Name',[tag,'conv2'])
    batchNormalizationLayer('Name',[tag,'BN2'])];
end

%%
% layers = bottleneckConvolutionalUnit(numF,stride,tag) creates a
% bottleneck convolutional unit, containing two 1-by-1 convolutional layers
% and one 3-by-3 layer. The 3-by-3 layer has numF filters, while the final
% 1-by-1 layer upsamples the output to 4*numF channels. The stride is
% applied in the 3-by-3 convolution so that no input activations are
% completely discarded (the 3-by-3 filters are still overlapping).
function layers = bottleneckConvolutionalUnit(numF,stride,tag)
layers = [
    convolution2dLayer(1,numF,'Padding','same','Name',[tag,'conv1'])
    batchNormalizationLayer('Name',[tag,'BN1'])
    %reluLayer('Name',[tag,'relu1'])
    logreluLayer(numF,[tag,'relu1'])
    
    convolution2dLayer(3,numF,'Padding','same','Stride',stride,'Name',[tag,'conv2'])
    batchNormalizationLayer('Name',[tag,'BN2'])
    %reluLayer('Name',[tag,'relu2'])
    logreluLayer(numF,[tag,'relu2'])
    
    convolution2dLayer(1,4*numF,'Padding','same','Name',[tag,'conv3'])
    batchNormalizationLayer('Name',[tag,'BN3'])];
end


