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
% Indirect
t_0=50;
X0=[0;0;0];
x_current=X0;
Sigma_g=eye(3);
X1=zeros(6);
XY=zeros(3,6);
  P=dlyap((A+B*kcurrentBS)',Q+kcurrentBS'*R*kcurrentBS);
    % true covariance Sigma_K
    Pw=dlyap((A+B*kcurrentBS),Sigma_w);
    C=2*((R+B'*P*B)*kcurrentBS+B'*P*A)*Pw;
 error=zeros(500,300);
 for num=1:500
     X0=[0;0;0];
x_current=X0;
Sigma_g=eye(3);
X1=zeros(6);
XY=zeros(3,6);
for t=1:t_0
   u_t=kcurrentBS*x_current+mvnrnd(zeros(3,1),Sigma_g,1)';
   X1=X1+[x_current;u_t]*[x_current;u_t]';
   xnew=realsystem(x_current,u_t,A,B,Sigma_w);
   XY=XY+xnew*[x_current;u_t]';
   x_current=xnew;
end
Est=XY*inv(X1);
%rls
for i=1:300
    u_t=kcurrentBS*x_current+mvnrnd(zeros(3,1),Sigma_g,1)';
 X1=X1+[x_current;u_t]*[x_current;u_t]';
  xnew=realsystem(x_current,u_t,A,B,Sigma_w);
   %XY=XY+xnew*[x_current;u_t]';
   Est=Est+(xnew-Est*[x_current;u_t])*[x_current;u_t]'*inv(X1);
   Aest=Est(1:3,1:3);
   Best=Est(1:3,4:6);
      P=dlyap((Aest+Best*kcurrentBS)',Q+kcurrentBS'*R*kcurrentBS);
      Pw=dlyap((Aest+Best*kcurrentBS),Sigma_w);
    Cest=2*((R+Best'*P*Best)*kcurrentBS+Best'*P*Aest)*Pw;
   error(num,i)=norm(Cest-C);
   x_current=xnew;
end
 end
 Bias=mean(error,1);
 Var=var(error,1);
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