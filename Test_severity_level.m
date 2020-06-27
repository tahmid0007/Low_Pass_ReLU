lev = 1;
for i = 0:10000:40000
    for j = 1:10000
        I(:,:,:,j) = zoom_blur(:,:,:,(i+j));
        labels_br(j,1) = my_labels((i+j),1);
    end
    labels = classify(WRN_40_2, I);
    confMat = my_confusionmat(labels_br, labels);
    confMat = confMat./sum(confMat,2);
    mean(diag(confMat))
    lev = lev+1;   
end
