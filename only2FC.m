
%theta = 0:0.05:2*pi;
theta = .70:.02:2.5;
rho = 5:5:100;
for i = 1:50
    [x,y] = pol2cart(theta(1,i),rho(1,2))
    
    a(1,1,1) = x;
    a(1,1,2) = y;
    
    [labels scores] = classify(Lenet_only2_FC, a);
    
end