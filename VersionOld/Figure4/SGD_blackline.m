%% define the true system and the weighting matrix 
A=[1,-1.13,-0.65,-0.807,1.59; 0,0.77,0.32,-0.98,-2.97;0,0.12,0.02,0.0,-0.36;0,0.01,0.01,-0.03,-0.04;0,0.14,-0.09,0.29,0.76];% system matrix A
B=[89.20,-50.17,1.13,-19.35;5.22,6.36,0.23,-0.32;-9.47,5.93,-0.12,0.99;-0.32,0.32,-0.01,-0.01;-4.53,3.21,-0.14,0.09]; % input matrix B
nx=size(A,2); % number of states
nu=size(B,2); % number of control inputs
Q=1*eye(5); % weighting matrix Q
R=1*eye(4); % weighting matrix R
[Pstar1,K_initial]=idare(A,B,40*Q,R);
[Pstar,~]=idare(A,B,Q,R);
K_initial=-K_initial;
Sigma_w=10^(-3)*eye(nx);
Sigma_0=10^(-6)*eye(nx);
% r=[0.3,0.25,0.2,0.15,0.1,0.08,0.05,0.03,0.01,0.008,0.005];
% %r=0.1;
% l=[200,500,800,1000,2000,5000];
% num=500;
% nablaK=zeros(3,3,numel(r),num);
% n=1;
% %l=100;

% %% Direct
% errorD=zeros(numel(r),numel(l));
% variance=zeros(numel(r),numel(l));
% for i=1:numel(r)
%     for j=1:numel(l)
%     errorgradient=zeros(num,1);
%     %temp1=zeros(1000);
%     %temp2=zeros(1000);
%     for counter=1:num
%         [nablaK(:,:,i,counter),~,errorgradient(counter),~]=estimationBS(kcurrentBS,l(j),r(i),n,A,B,Q,R,Sigma_w,Sigma_0,nx,nu);
%         %temp1=norm(errorgradient(:,:,counter));
%     end
%     errorD(i,j)=mean(errorgradient,1);
%     variance(i,j)=var(errorgradient);
%     end
% end
% plot(errorD)
% Indirect
t_0=50;
X0=[0;0;0;0;0];
x_current=X0;
Sigma_g=eye(nu);
X1=zeros(nx+nu);
XY=zeros(nx,nx+nu);
%  P=dlyap((A+B*kcurrentBS)',Q+kcurrentBS'*R*kcurrentBS);
    % true covariance Sigma_K
%    Pw=dlyap((A+B*kcurrentBS),Sigma_w);
%    C=2*((R+B'*P*B)*kcurrentBS+B'*P*A)*Pw;
 error=zeros(500,300);
 iteration=100000;
 ck=zeros(1,iteration);
 for num=1:10
     X0=[0;0;0;0;0];
     kcurrentBS=K_initial;
x_current=X0;
Sigma_g=eye(nu);
X1=zeros(nx+nu);
XY=zeros(nx,nx+nu);
     num
for t=1:t_0
   u_t=kcurrentBS*x_current+mvnrnd(zeros(nu,1),Sigma_g,1)';
   X1=X1+[x_current;u_t]*[x_current;u_t]';
   xnew=realsystem(x_current,u_t,A,B,Sigma_w);
   XY=XY+xnew*[x_current;u_t]';
   x_current=xnew;
end
Est=XY*inv(X1);
%rls


for i=1:iteration
    u_t=kcurrentBS*x_current+mvnrnd(zeros(nu,1),Sigma_g,1)';
 X1=X1+[x_current;u_t]*[x_current;u_t]';
  xnew=realsystem(x_current,u_t,A,B,Sigma_w);
   %XY=XY+xnew*[x_current;u_t]';
   Est=Est+(xnew-Est*[x_current;u_t])*[x_current;u_t]'*inv(X1);
   Aest=Est(1:nx,1:nx);
   Best=Est(1:nx,nx+1:end);
      P=dlyap((Aest+Best*kcurrentBS)',Q+kcurrentBS'*R*kcurrentBS);
      Pw=dlyap((Aest+Best*kcurrentBS),Sigma_w);
    Cest=2*((R+Best'*P*Best)*kcurrentBS+Best'*P*Aest)*Pw;
   %error(num,i)=norm(Cest-C);
   x_current=xnew;
   pk=dlyap((A+B*kcurrentBS)',Q+kcurrentBS'*R*kcurrentBS);
   ck(num,i)=(trace(pk)-trace(Pstar))/trace(Pstar);
   %kcurrentBS=kcurrentBS-0.05 *Cest;
   kcurrentBS=kcurrentBS-0.05/ceil((i^(1/2+1/100))/50) *Cest;
end
 end
 plot(ck)
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
    xnew=A*state+B*controlinput+mvnrnd(zeros(5,1),Sigma_w,1)';
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