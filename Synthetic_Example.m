clc
clear all

addpath('./utils')

%% Generate mixture Gaussians (convex shapes) synthetic dataset. Dataset has two views with 1502 samples and 3 dimensions, with 3 classes.
... There are 500 samples per class and 2 outliers. The samples are modelled to approximatelty live on a 2 dimensional subspace 
... this is done by modelling each class by  2-dimensional Gaussian distrubtion, then the 3rd dimension is added by unifrom random noise.
... Both views contain complementary information.
... In view 1: class 1 and 2 are overlapping and class 3 is distinguished from both classes 1 and 2.
... In view 2: class 1 and 3 are overlapping and class 2 is distinguished from both classes 1 and 3.


%%%% view 1 construction
N = 500; % number of samples
C1 = [1,0;0,1]; % covaraince matric of Class 1
m1 = [1;2]; % mean of Class 1


C2 = [1,0;0,1]; % covaraince matric of Class 2
m2 = [1;4]; % mean of Class 2

C3 = [1,0;0,1]; % covaraince matric of Class 3
m3 = [6;6]; % mean of Class 3

X1 = m1 + chol(C1)*randn(2,N); % 2-dimnesional Gaussian consturction
X1(3,:) = 0.5.*rand(1,N); % adding uniform random noise to construct 3rd dimension

X2 = m2 + chol(C2)*randn(2,N); 
X2(3,:) = 0.5.*rand(1,N);

X3 = m3 + chol(C3)*randn(2,N);
X3(3,:) = 0.5.*rand(1,N);

out1 = [2;4;1.5]; % outlier point 1
out2 = [3;4;-1.5]; % outlier point 2
X_v1 = [X1, X2, X3, out1, out2]; % final multi-view dataset



figure (1)
title('View 1')
plot3(X_v1(1,1:500),X_v1(2,1:500),X_v1(3,1:500),'bo')
hold on 
plot3(X_v1(1,501:1000),X_v1(2,501:1000),X_v1(3,501:1000),'ro')
hold on 
plot3(X_v1(1,1001:1500),X_v1(2,1001:1500),X_v1(3,1001:1500),'ko')
hold on
plot3(X_v1(1,1501:1502),X_v1(2,1501:1502),X_v1(3,1501:1502),'go')
title('View 1')
xlabel('x')
ylabel('y')
zlabel('z')
legend('C1','C2','C3','outliers')

%%%% view 2 construction
C1 = [1,0;0,1];
m1 = [1;2];


C2 = [1,0;0,1];
m2 = [6;6];

C3 = [1,0;0,1];
m3 = [1;4];

X1 = m1 + chol(C1)*randn(2,N);
X1(3,:)= 0.5 .* rand(1,N);

X2 = m2 + chol(C2)*randn(2,N);
X2(3,:)= 0.5 .* rand(1,N);

X3 = m3 + chol(C3)*randn(2,N);
X3(3,:)= 0.5 .* rand(1,N);

X_v2= [X1, X2, X3, out1, out2];

figure (2)
plot3(X_v2(1,1:500),X_v2(2,1:500),X_v2(3,1:500),'bo')
hold on 
plot3(X_v2(1,501:1000),X_v2(2,501:1000),X_v2(3,501:1000),'ro')
hold on 
plot3(X_v2(1,1001:1500),X_v2(2,1001:1500),X_v2(3,1001:1500),'ko')
hold on 
plot3(X_v2(1,1501:1502),X_v2(2,1501:1502),X_v2(3,1501:1502),'go')
title('View 2')
xlabel('x')
ylabel('y')
zlabel('z')
legend('C1','C2','C3','outliers')


%% 
X_data(:,:,1) = zeros(size(X_v1));
X_data(:,:,2) = zeros(size(X_v2));
X_Lap         = zeros(size(X_v1,2), size(X_v1,2), 2);
        
X_data(:,:,1) = X_v1;
X_data(:,:,2) = X_v2;

%%%%% normalizing each view 

%%% normalize mean to zero
for i = 1:size(X_data,1)
        
    X_data_m(i,:,1) = X_data(i,:,1)-mean(X_data(i,:,1));
    
end
    
for i = 1:size(X_data,1)
        
    X_data_m(i,:,2) = X_data(i,:,2) - mean(X_data(i,:,2));
    
end
%%%    

%%% normalize standard deviation to one    
for i=1:size(X_data,1)
        
    X_data_m(i,:,1) = X_data_m(i,:,1) / std(X_data_m(i,:,1));
    
end
    
for i = 1:size(X_data,1)
        
    X_data_m(i,:,2) = X_data_m(i,:,2) / std(X_data_m(i,:,2));
    
end
%%%

%%%%%

%%% Building K-NN Graph and computing Laplacian matrix.    
K=200;
[Lap_graph1, ~] = build_knn_graph(X_data_m(:,:,1)',K); % returns graph Laplacian matrix.
    
[Lap_graph2, ~] = build_knn_graph(X_data_m(:,:,2)',K); % returns graph Laplacian matrix.
%%%

X_Lap(:,:,1) = Lap_graph1;
X_Lap(:,:,2) = Lap_graph2;

%%% find shared latent space using CGRMSL    
lambda = [0.15,0.15];
alpha = 1;
gamma_v = [4,1]; 
[L_multi,S_multi,L_star,~,~,obj] = CGRMSL(X_data_m,X_Lap,lambda,alpha,gamma_v);
%%%

%%% plotting objective function of CGRMSL to show convergence
figure (3)
plot(obj,'-o')
title('Convergence Plot')
ylabel('objective function value')
xlabel('iteration')

% projecting L_star onto its column space, creating a two dimensional projection
[U,~,~] = svd(L_star);
Z_star  = U(:,1:2)'*L_star;
%

% plot Z_star for visualization
figure (4)
plot(Z_star(1,1:500),Z_star(2,1:500),'bo')
hold on
plot(Z_star(1,501:1000),Z_star(2,501:1000),'ro')
hold on 
plot(Z_star(1,1001:1500),Z_star(2,1001:1500),'ko')
hold on 
plot(Z_star(1,1501:1502),Z_star(2,1501:1502),'go')
legend('C1','C2','C3','outliers')



% projecting the low rank matrix of the first view ( L_multi(:,:,1) ) onto its column space, creating a two dimensional projection
[U,~,~] = svd(L_multi(:,:,1));
Z1      = U(:,1:2)'*L_multi(:,:,1);

% plot Z1 for visualization
figure(5)
plot(Z1(1,1:500),Z1(2,1:500),'bo')
hold on
plot(Z1(1,501:1000),Z1(2,501:1000),'ro')
hold on 
plot(Z1(1,1001:1500),Z1(2,1001:1500),'ko')
hold on 
plot(Z1(1,1501:1502),Z1(2,1501:1502),'go')
legend('C1','C2','C3','outliers')

% projecting the low rank matrix of the second view ( L_multi(:,:,2) ) onto its column space, creating a two dimensional projection
[U,~,~] =  svd(L_multi(:,:,2));
Z2      =  U(:,1:2)'*L_multi(:,:,2);

figure(6)
plot(Z2(1,1:500),Z2(2,1:500),'bo')
hold on
plot(Z2(1,501:1000),Z2(2,501:1000),'ro')
hold on 
plot(Z2(1,1001:1500),Z2(2,1001:1500),'ko')
hold on 
plot(Z2(1,1501:1502),Z2(2,1501:1502),'go')
legend('C1','C2','C3','outliers')

%%%% Finding outliers using reconstruction errors, for both views.

%%% view 1 reconstruction errors

err = sqrt(sum(  (X_data_m(:,:,1)-L_star).^2 ) );  % sample-wise reconstruction error

figure (11)
plot(1:1500,err(1:1500),'bx')
hold on 
plot(1501:1502,err(1501:1502),'r*')
title('View 1 Reconsturction error (CGRMSL)')
ylabel('Reconstruction error') 
xlabel('Sample Index')
legend('uncorrupted samples','outliers')
%%%%

%%% view 2 reconstruction errors

err = sqrt( sum( (X_data_m(:,:,2) - L_star).^2 ) );  % sample-wise reconstruction error
figure (12)
plot(1:1500,err(1:1500),'bx')
hold on 
plot(1501:1502,err(1501:1502),'r*')
title('View 2 Reconsturction error (CGRMSL)')
ylabel('Reconstruction error') 
xlabel('Sample Index')

%%%

%% CGMSL: non - robust CGRMSL.

clear X_data X_Lap    
X_data(:,:,1) = zeros(size(X_v1));
X_data(:,:,2) = zeros(size(X_v2));
X_Lap         = zeros(size(X_v1,2), size(X_v1,2), 2);

X_data(:,:,1) = X_v1;
X_data(:,:,2) = X_v2;

    
    
%%%%% normalizing each view 

%%% normalize mean to zero
for i = 1:size(X_data,1)
        
    X_data_m(i,:,1) = X_data(i,:,1) - mean(X_data(i,:,1));
    
end
    
for i = 1:size(X_data,1)
        
    X_data_m(i,:,2) = X_data(i,:,2) - mean(X_data(i,:,2));
    
end
%%%    
%%% normalize standard deviation to one     
for i = 1:size(X_data,1)
        
    X_data_m(i,:,1) = X_data_m(i,:,1) / std(X_data_m(i,:,1));
    
end
    
for i = 1:size(X_data,1)
        
    X_data_m(i,:,2) = X_data_m(i,:,2) / std(X_data_m(i,:,2));
    
end
%%%

%%%%%

%%% Building Graph and computing Laplacina matrix.    
K=200;
[Lap_graph1, ~] = build_knn_graph(X_data_m(:,:,1)',K); % returns graph Laplacian matrix.
    
[Lap_graph2, ~] = build_knn_graph(X_data_m(:,:,2)',K); % returns graph Laplacian matrix.
%%%
    
X_Lap(:,:,1)=Lap_graph1;
X_Lap(:,:,2)=Lap_graph2;

%%% find shared latent space using CGMSL
alpha = 1;
gamma_v = [4,1];
lambda = [2,2];
[L_hat, L_star, W_k, cnt, obj_func] = admm_algo_CGMSL(X_data_m,X_Lap,lambda,alpha,gamma_v);
%%%

%%% plotting objective function of CGRMSL to show convergence
figure (13)
plot(obj_func,'-o')
%%%
 
[P,~,~] = svd(L_star);
Z_star = P(:,1:2)'*L_star;

figure (14)
plot(Z_star(1,1:500),Z_star(2,1:500),'bo')
hold on
plot(Z_star(1,501:1000),Z_star(2,501:1000),'ro')
hold on 
plot(Z_star(1,1001:1500),Z_star(2,1001:1500),'ko')
hold on 
plot(Z_star(1,1501:1502),Z_star(2,1501:1502),'go')


%%% view 1 reconstruction errors

err = sqrt(sum(  (X_data_m(:,:,1)-L_star).^2 ) );  % sample-wise reconstruction error

figure (15)
plot(1:1500,err(1:1500),'bx')
hold on 
plot(1501:1502,err(1501:1502),'r*')
ylabel('Reconstruction error') 
xlabel('Sample Index')
legend('uncorrupted samples','outliers')
%%%%

%%% view 2 reconstruction errors

err = sqrt( sum( (X_data_m(:,:,2) - L_star).^2 ) ); % sample-wise reconstruction error
figure (16)
plot(1:1500, err(1:1500),'bx')
hold on 
plot(1501:1502, err(1501:1502),'r*')
ylabel('Reconstruction error') 
xlabel('Sample Index')

%%% 

