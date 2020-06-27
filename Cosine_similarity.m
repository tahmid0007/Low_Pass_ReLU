k = 1;
for i = 1:1
    I(:,:,:,k) = XValidation(:,:,:,i);
    k = k+1;
    I(:,:,:,k) = brightness(:,:,:,i);
    k = k+1;
    I(:,:,:,k) = brightness(:,:,:,(10000+i));
    k = k+1;
    I(:,:,:,k) = brightness(:,:,:,(20000+i));
    k = k+1;
    I(:,:,:,k) = brightness(:,:,:,(30000+i));
    k = k+1;
    I(:,:,:,k) = brightness(:,:,:,(40000+i));
    k = k+1;
end
j = 1;
for i = 1:6
    for p = 1:6
    act1(1,:,:,:) = activations(WRN4002_ReLU_200ep,I(:,:,:,i),'reluInp');
    act2(1,:,:,:) = activations(WRN4002_ReLU_200ep,I(:,:,:,i),'relu11');
    act3(1,:,:,:) = activations(WRN4002_ReLU_200ep,I(:,:,:,i),'relu23');
    act4(1,:,:,:) = activations(WRN4002_ReLU_200ep,I(:,:,:,i),'relu35');
    
    my_act1(i,:,:,:) = activations(WRN_4002_myrelu_200ep,I(:,:,:,i),'reluInp');
    my_act2(i,:,:,:) = activations(WRN_4002_myrelu_200ep,I(:,:,:,i),'relu11');
    my_act3(i,:,:,:) = activations(WRN_4002_myrelu_200ep,I(:,:,:,i),'relu23');
    my_act4(i,:,:,:) = activations(WRN_4002_myrelu_200ep,I(:,:,:,i),'relu35');
    
        my_act2(i,:,:,:) = activations(WRN_3602_myRelu_3stage_v2,I(:,:,:,i),'relu11');
        my_act3(i,:,:,:) = activations(WRN_3602_myRelu_3stage_v2,I(:,:,:,i),'relu12');
        my_act4(i,:,:,:) = activations(WRN_3602_myRelu_3stage_v2,I(:,:,:,i),'relu23');
        my_act5(i,:,:,:) = activations(WRN_3602_myRelu_3stage_v2,I(:,:,:,i),'relu24');
        my_act6(i,:,:,:) = activations(WRN_3602_myRelu_3stage_v2,I(:,:,:,i),'relu34');
        my_act7(i,:,:,:) = activations(WRN_3602_myRelu_3stage_v2,I(:,:,:,i),'relu35');
         [c s]= classify(WRN_4002_myrelu_200ep2,I(:,:,:,i))
         [c s]= classify(WRN4002_ReLU_200ep,I(:,:,:,i))
    
    dmy_act1(i,:,:,:) = activations(WRN_4002_myrelu_200ep2_DCT,I(:,:,:,i),'reluInp');
    dmy_act2(i,:,:,:) = activations(WRN_4002_myrelu_200ep2_DCT,I(:,:,:,i),'relu11');
    dmy_act3(i,:,:,:) = activations(WRN_4002_myrelu_200ep2_DCT,I(:,:,:,i),'relu23');
    dmy_act4(i,:,:,:) = activations(WRN_4002_myrelu_200ep2_DCT,I(:,:,:,i),'relu35');
    
    end
sum(act1(:))
mean(act2(:))
mean(act3(:))
sum(act4(:))
end


j = 1;k=1;
for i = 0:6:1194
    for q = 1:6
        %dist1(j,1) = getCosineSimilarity(act1(i+1,:),act1(i+q,:));
        dist1(j,1) = getCosineSimilarity(act1(i+1,:),act1(i+q,:));
        dist2(j,1) = getCosineSimilarity(act2(i+1,:),act2(i+q,:));
        dist3(j,1) = getCosineSimilarity(act3(i+1,:),act3(i+q,:));
        dist4(j,1) = getCosineSimilarity(act4(i+1,:),act4(i+q,:));
        %dist3(j,1) = getCosineSimilarity(my_act1(i+1,:),my_act1(i+q,:));
        dist5(j,1) = getCosineSimilarity(my_act1(i+1,:),my_act1(i+q,:));
        dist6(j,1) = getCosineSimilarity(my_act2(i+1,:),my_act2(i+q,:));
        dist7(j,1) = getCosineSimilarity(my_act3(i+1,:),my_act3(i+q,:));
        dist8(j,1) = getCosineSimilarity(my_act4(i+1,:),my_act4(i+q,:));
        %dist5(j,1) = getCosineSimilarity(dmy_act1(i+1,:),dmy_act1(i+q,:));
        dist9(j,1) = getCosineSimilarity(dmy_act1(i+1,:),dmy_act1(i+q,:));
        dist10(j,1) = getCosineSimilarity(dmy_act2(i+1,:),dmy_act2(i+q,:));
        dist11(j,1) = getCosineSimilarity(dmy_act3(i+1,:),dmy_act3(i+q,:));
        dist12(j,1) = getCosineSimilarity(dmy_act4(i+1,:),dmy_act4(i+q,:));
        j =j+1;
    end
    
end
j = 1;k=1;
for i = 1:6
    j = 1;
    for q = i:6:1194
        a1(j,1) = dist1(q,1);
        a2(j,1) = dist2(q,1);
        a3(j,1) = dist3(q,1);
        a4(j,1) = dist4(q,1);
        
        a5(j,1) = dist5(q,1);
        a6(j,1) = dist6(q,1);
        a7(j,1) = dist7(q,1);
        a8(j,1) = dist8(q,1);
        
        a9(j,1) = dist9(q,1);
        a10(j,1) = dist10(q,1);
        a11(j,1) = dist11(q,1);
        a12(j,1) = dist12(q,1);
        j =j+1;
    end
    bmdist1(k,1) = mean(a1);
    bmdist2(k,1) = mean(a2);
    bmdist3(k,1) = mean(a3);
    bmdist4(k,1) = mean(a4);
    
    bmdist5(k,1) = mean(a5);
    bmdist6(k,1) = mean(a6);
    bmdist7(k,1) = mean(a7);
    bmdist8(k,1) = mean(a8);
    
    bmdist9(k,1) = mean(a9);
    bmdist10(k,1) = mean(a10);
    bmdist11(k,1) = mean(a11);
    bmdist12(k,1) = mean(a12);
    k = k+1;
end