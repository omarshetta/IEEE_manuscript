function [L_hat,S_hat,L_star,W_k,cnt,obj_func_k]=CGRMSL(X,Phi,lambda,gamma,lambda_v)
eta=0.00000001; % 0.000001
p=size(X,1);
n=size(X,2);
M=size(X,3);
lambda_sum=sum(lambda_v);

L_k=rand(p,n,M);
W_k=rand(p,n,M);
 W_kp1=zeros(p,n,M);
%  W_kp1=randn(p,n,M);
S_k=rand(p,n,M);
L_star_k=rand(p,n);


for m=1:M
Z1_k(:,:,m)=X(:,:,m)-L_k(:,:,m)-S_k(:,:,m);
end
for m=1:M
Z2_k(:,:,m)=W_k(:,:,m)-L_k(:,:,m);
end
for m=1:M
P1_k(m)=norm_nuclear(L_k(:,:,m));
end

for m=1:M
P2_k(m)=lambda(m)*sum(sqrt(sum(S_k(:,:,m).^2)));
end

for m=1:M
P3_k(m)=gamma*trace(L_k(:,:,m)*Phi(:,:,m)*L_k(:,:,m)');
end


for m=1:M
P4_k(m)=lambda_v(m)*norm(W_k(:,:,m)-L_star_k,'fro')^2;
end
converged=0;
cnt=0;


r1_k=1;
r2_k=1;
cnt=0;
maxiter=1000;    


while (~converged) 

for m=1:M
H_1(:,:,m)=X(:,:,m)-S_k(:,:,m)+(Z1_k(:,:,m)/r1_k);
H_2(:,:,m)=W_k(:,:,m)+(Z2_k(:,:,m)/r2_k);
A(:,:,m)=(r1_k*H_1(:,:,m) + r2_k*H_2(:,:,m))/(r1_k + r2_k);
 r_k=(r1_k+r2_k)/2;    
%  r_k=(r1_k+r2_k);    
    
L_kp1(:,:,m)=prox_nuclear_norm(A(:,:,m),1/r_k);
S_kp1(:,:,m)=colmn_thresh((X(:,:,m)-L_kp1(:,:,m)+(Z1_k(:,:,m)/r1_k)),lambda(m)/r1_k);
% W_kp1=(L_kp1-(Z2_k/2))*(2*gamma*Phi+ones(n,n))^-1;
% W_kp1(:,:,m)=r2_k*(L_kp1(:,:,m)-(Z2_k(:,:,m)/r2_k)+ lambda_v(m)*L_star_k)*( gamma*Phi(:,:,m)+ ( r2_k+lambda_v(m) )*diag(ones(1,n)) )^-1;
W_kp1(:,:,m)=( r2_k*( L_kp1(:,:,m)-(Z2_k(:,:,m)/r2_k) )+ lambda_v(m)*L_star_k )*( gamma*Phi(:,:,m)+ ( r2_k+lambda_v(m) )*diag(ones(1,n)) )^-1;
temp=zeros(size(W_kp1(:,:,1)));
for j=1:M
%  T=  W_kp1(:,:,j)+temp;
 T= (lambda_v(j)/lambda_sum) * (W_kp1(:,:,j))  +  temp;
temp=T;
end
% L_star_kp1 = 1/( 2*lambda_v(m) ) * T;
   L_star_kp1 = T;


Z1_kp1(:,:,m)=Z1_k(:,:,m)+r1_k*(X(:,:,m)-L_kp1(:,:,m)-S_kp1(:,:,m));
Z2_kp1(:,:,m)=Z2_k(:,:,m)+r2_k*(W_kp1(:,:,m)-L_kp1(:,:,m));

P1_kp1(m)=norm_nuclear(L_kp1(:,:,m));
P2_kp1(m)=lambda(m)*sum(sqrt(sum(S_kp1(:,:,m).^2)));
P3_kp1(m)=gamma*trace(L_kp1(:,:,m)*Phi(:,:,m)*L_kp1(:,:,m)');
P4_kp1(m)=lambda_v(m)*norm( W_kp1(:,:,m)-L_star_kp1,'fro' );


%  obj_func_k(cnt)=P1_k+P2_k+P3_k;
% obj_func_kp1(cnt)=P1_kp1+P2_kp1+P3_kp1;



rel_err_1(m)= norm( P1_kp1(m)-P1_k(m),'fro' )^2/( norm(P1_k(m),'fro')^2);



 rel_err_2(m)=norm(P2_kp1(m)-P2_k(m),'fro')^2/(norm(P2_k(m),'fro')^2);



 rel_err_3(m)=norm(P3_kp1(m)-P3_k(m),'fro')^2/(norm(P3_k(m),'fro')^2); 
 
 
 rel_err_4(m)=norm(P4_kp1(m)-P4_k(m),'fro')^2/(norm(P4_k(m),'fro')^2); 



 rel_err_z1(m)=norm(Z1_kp1(m)-Z1_k(m),'fro')^2/(norm(Z1_k(m),'fro')^2);



rel_err_z2(m)=norm(Z2_kp1(m)-Z2_k(m),'fro')^2/(norm(Z2_k(m),'fro')^2);

%  rel_err_1<eta && rel_err_2<eta && rel_err_3<eta && rel_err_z1<eta && rel_err_z2<eta 
% norm(R1_k,'fro')<=0.001 && norm(R2_k,'fro')<=0.001 && norm(S1_k,'fro')<=0.001 && norm(S2_k,'fro')<=0.001


%  if(rel_err_1<eta && rel_err_2<eta && rel_err_3<eta && rel_err_z1<eta && rel_err_z2<eta || cnt>=maxiter )
%      converged=1;
%  else
  % get ready for new iteration    
  
      
  S_k(:,:,m)=S_kp1(:,:,m);
  L_k(:,:,m)=L_kp1(:,:,m);
  W_k(:,:,m)=W_kp1(:,:,m);
%      L_star_k=L_star_kp1;
  
%    r1_k=r1_kp1;
%    r2_k=r2_kp1;
  Z1_k(:,:,m)=Z1_kp1(:,:,m);
  Z2_k(:,:,m)=Z2_kp1(:,:,m);
  P1_k(m)=P1_kp1(m);
  P2_k(m)=P2_kp1(m);
  P3_k(m)=P3_kp1(m);
  P4_k(m)=P4_kp1(m);
  
end

      L_star_k=L_star_kp1;
 
 cnt=cnt+1
obj_func_k(cnt)=sum(P1_k)+sum(P2_k)+sum(P3_k)+sum(P4_k);

 for m=1:M
 ind = rel_err_1(m) < eta;
 ind = ind && (rel_err_2(m) < eta);
 ind = ind&&(rel_err_2(m) < eta) ;
 ind = ind&&(rel_err_3(m) < eta) ;
 ind = ind&&(rel_err_4(m) < eta) ;
 ind = ind&&(rel_err_z1(m) < eta) ;
 ind = ind&&(rel_err_z2(m) < eta) ;
 end
 if(ind || cnt>=maxiter )
     converged=1;
  
 end
    
    
    
end


L_hat=L_kp1;
S_hat=S_kp1;
L_star=L_star_kp1;


end



