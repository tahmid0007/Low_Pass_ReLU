shot_noise = readNPY('E:\00 PhD\DataSets\CIFAR-10-C\CIFAR-10-C\shot_noise.npy');
%test(1,1,2) = 100, 0;
%[labels scores] = classify(only2fc, test)

 for i = 1:50000
     if lbl(i,1) == 0
         my_labels(i,1) = 'airplane';
     end
     
     if lbl(i,1) == 1
         my_labels(i,1) = 'automobile';
     end
     
     if lbl(i,1) == 2
         my_labels(i,1) = 'bird';
     end
     
     if lbl(i,1) == 3
         my_labels(i,1) = 'cat';
     end
     
     if lbl(i,1) == 4
         my_labels(i,1) = 'deer';
     end
     
     if lbl(i,1) == 5
         my_labels(i,1) = 'dog';
     end
     
     if lbl(i,1) == 6
         my_labels(i,1) = 'frog';
     end
     
     if lbl(i,1) == 7
         my_labels(i,1) = 'horse';
     end
     
     if lbl(i,1) == 8
         my_labels(i,1) = 'ship';
     end
     
     if lbl(i,1) == 9
         my_labels(i,1) = 'truck';
     end
 end
 my_labels = categorical(my_labels);