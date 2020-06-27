
Image = imresize(imread('your_path'),[64 64]);

YCBCR = rgb2ycbcr(Im);
I1 = YCBCR(:,:,1);
I2 = YCBCR(:,:,2);
I3 = YCBCR(:,:,3);

J = dct2(I1);
J(abs(J) < randi([0 50])) = 0;
K = uint8(idct2(J));
%imshow(K);

rec(:,:,1) = uint8(K);
rec(:,:,2) = I2;
rec(:,:,3) = I3;
L = ycbcr2rgb(rec);


