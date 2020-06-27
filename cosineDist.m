
for j = 1:48
    D(j,1) = pdist2(act4(j,:),nact4(j,:),'cosine');
end
M = mean(D)
