
for i = 40006:40006
    act1(1,:,:,:) = activations(trainedNet,gaussian_blur(:,:,:,i),'reluInp');
    act2(1,:,:,:) = activations(trainedNet,gaussian_noise(:,:,:,i),'reluInp');
    act3(1,:,:,:) = activations(trainedNet,XValidation(:,:,:,6),'reluInp');

end
x1= act1(:);
x2= act2(:);
x3= act3(:);
v1 = nonzeros(x1*1.5);
v2 = nonzeros(x2*2.5);
v3 = nonzeros(x3*2);

h2 = histogram(v2,111,'FaceColor','c');
figure

h3 = histogram(v3,111,'FaceColor','k');
figure
h1 = histogram(v1,111,'FaceColor','r');
