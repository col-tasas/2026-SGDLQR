%% define the true system and the weighting matrix 
A=[1.01,0.01,0;0.01,1.01,0.01;0,0.01,1.01];% system matrix A
B=1*eye(3); % input matrix B
nx=size(A,2); % number of states
nu=size(B,2); % number of control inputs
Q=0.001*eye(3); % weighting matrix Q
R=1*eye(3); % weighting matrix R
[Pstar1,K_initial]=idare(A,B,10*Q,R);
K_initial=-K_initial;
Sigma_w=10^(-3)*eye(3);
Sigma_0=10^(-1)*eye(3);
r=[0.12,0.11,	0.1,	0.05,	0.01	0.005,	0.001,	0.0005,0.0001];
%r=0.1;
%l=[200,500,800,1000,1200,1500,1800];
num=500;
%nablaK=zeros(3,3,numel(r),num);
n=1;
l=800;
kcurrentBS=K_initial;
%% Direct
errorD=zeros(numel(r),numel(l));
variance=zeros(numel(r),numel(l));
for i=1:numel(r)
    for j=1:numel(l)
    errorgradient=zeros(num,1);
    %temp1=zeros(1000);
    %temp2=zeros(1000);
    for counter=1:num
        [nablaK(:,:,i,counter),~,errorgradient(counter),~]=estimationBS(kcurrentBS,l(j),r(i),n,A,B,Q,R,Sigma_w,Sigma_0,nx,nu);
        %temp1=norm(errorgradient(:,:,counter));
    end
    errorD(i,j)=mean(errorgradient,1);
    variance(i,j)=var(errorgradient);
    end
end
figure
plot(errorD)
hold on
plot(errorD4)
hold on
plot(variance)
hold on
plot(variance4)
%% Required functions
function [nablaK,SigmaK,errorgradient,errorcovariance]=estimationBS(kcurrent,l,r,n,A,B,Q,R,Sigma_w,Sigma_0,nx,nu)
    % true kernal P_K
    P=dlyap((A+B*kcurrent)',Q+kcurrent'*R*kcurrent);
    % true covariance Sigma_K
    Pw=dlyap((A+B*kcurrent),Sigma_w);
    C=2*((R+B'*P*B)*kcurrent+B'*P*A)*Pw;
    ckhistory=zeros(nx,nu);
    xkhistory=zeros(nx,nx);
    for j=1:n
        %disp([' Sample ', num2str(j)])
        xhistory=zeros(nx,l+1);
        xhistory(:,1)=mvnrnd(zeros(1,3),Sigma_0,1);
        X=randommatrix(kcurrent,r);
        K=kcurrent+X;
        ck=0;
        xk=zeros(nx,nx);
        for p=1:l
            xhistory(:,p+1)=realsystem(xhistory(:,p),K*xhistory(:,p),A,B,Sigma_w);
            ck=ck+xhistory(:,p)'*(Q+K'*R*K)*xhistory(:,p);
            xk=xk+xhistory(:,p)*xhistory(:,p)';
        end
        ckhistory=ckhistory+nx*nu/(r^2)*1/l*ck*X;
        xkhistory=xkhistory+1/l*xk;
    end
    nablaK=ckhistory/n;
    errorgradient=norm(nablaK-C);
    SigmaK=xkhistory/n;
    errorcovariance=norm(SigmaK-Pw);
end

% system dynamic
function xnew=realsystem(state,controlinput,A,B,Sigma_w) 
    xnew=A*state+B*controlinput+mvnrnd(zeros(3,1),Sigma_w,1)';
end
% generete the random matrix for exploration
function X=randommatrix(K,r)
  
    [a,b]=size(K);
      lower = -100000;
      upper = 100000;
    random = lower + (upper-lower).*rand(a*b,1);
    target_norm=r;
    X=reshape(random,[a,b]);
    current_norm=norm(X,'fro');
    X=X*(target_norm/current_norm);
end