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
eig(A+B*K_initial)
Sigma_w=10^(-3)*eye(nx);
Sigma_0=10^(-3)*eye(nx);
% r=[0.3,0.25,0.2,0.15,0.1,0.08,0.05,0.03,0.01,0.008,0.005];
% %r=0.1;
% l=[200,500,800,1000,2000,5000];
% num=500;
% nablaK=zeros(3,3,numel(r),num);
% n=1;
% %l=100;
iteration=40000;
ck=zeros(1,iteration);
maxnum=100;
kcurrentBS=K_initial;
noise_var=0.0005;
for num=1:maxnum
    kcurrentBS=K_initial;
  for i=1:iteration
      beta=0.05/(i^(1/2));
    i
     %[nablaK,~,error,~]=estimationBS(kcurrentBS,l,r,n,A,B,Q,R,Sigma_w,Sigma_0,nx,nu);
     ctrue=comp_grad(A,B,Q,R,kcurrentBS,Sigma_w);
     nablaK=ctrue+random_matrix_mean_bounded(nu, nx, beta, noise_var);
     pk=dlyap((A+B*kcurrentBS)',Q+kcurrentBS'*R*kcurrentBS);
     ck(num,i)=(trace(pk)-trace(Pstar))/trace(Pstar);
kcurrentBS=kcurrentBS-0.05/ceil((i^(1/2+1/100))/100)*nablaK;
%kcurrentBS=kcurrentBS-0.05*nablaK;
  end
end
avg = truncated_mean_with_bounds(ck);
plot(avg)
%% Required functions
function X = random_matrix_mean_bounded(n, m, beta, noise_var)
    % E[X] = M, ||M||_F <= beta
    % E||X - E[X]||_F^2 = noise_var

    M0 = randn(n,m);
    M0 = M0 / norm(M0,'fro');   
    alpha = beta * rand();     % Uniform in [0, beta]
    M = alpha * M0;            % ||M||_F <= beta

    nu2 = noise_var / (n*m);
    W = sqrt(nu2) * randn(n,m);

    X = M + W;
end

function gradient=comp_grad(A,B,Q,R,K,Sigma_w)
     P=dlyap((A+B*K)',Q+K'*R*K);
     Pw=dlyap((A+B*K),Sigma_w);
     gradient=2*((R+B'*P*B)*K+B'*P*A)*Pw;
end

function avg = truncated_mean_with_bounds(X)
% TRUNCATED_MEAN_WITH_BOUNDS
% X(i,k): 第 i 条数据，第 k 个时刻
%
% 规则：
%   - 若 X(i,k) < 0：该条数据从 k 开始永久截断
%   - 若 X(i,k) > 5：仅丢弃该点
%   - 仅使用 0 <= X(i,k) <= 5 的点计算平均
%
% 输出：
%   avg(k): 第 k 个时刻的平均值

    [N, T] = size(X);

    alive = true(N,1);     % 是否尚未被负数截断
    avg   = nan(1,T);

    for k = 1:T
        % 先处理负数：更新存活状态
        alive = alive & (X(:,k) >= 0);

        % 当前时刻可用的数据
        valid = alive & (X(:,k) <= 100);

        if any(valid)
            avg(k) = mean(X(valid,k));
        else
            break;   % 所有数据都被截断了
        end
    end
end

function [nablaK,SigmaK,errorgradient,errorcovariance]=estimationBS(kcurrent,l,r,n,A,B,Q,R,Sigma_w,Sigma_0,nx,nu)
    % true kernal P_K
    P=dlyap((A+B*kcurrent)',Q+kcurrent'*R*kcurrent);
    % true covariance Sigma_K
    Pw=dlyap((A+B*kcurrent),Sigma_w);
    C=2*((R+B'*P*B)*kcurrent+B'*P*A)*Pw;
    ckhistory=zeros(nu,nx);
    xkhistory=zeros(nx,nx);
    for j=1:n
        %disp([' Sample ', num2str(j)])
        xhistory=zeros(nx,l+1);
        xhistory(:,1)=mvnrnd(zeros(1,nx),Sigma_0,1);
        X=randommatrix(kcurrent,r);
        K=kcurrent+X;
        ck=0;
        xk=zeros(nx,nx);
        for p=1:l
            xhistory(:,p+1)=realsystem(xhistory(:,p),K*xhistory(:,p),A,B,Sigma_w);
            ck=ck+xhistory(:,p)'*(Q+K'*R*K)*xhistory(:,p);
            xk=xk+xhistory(:,p)*xhistory(:,p)';
        end
        %size(nx*nu/(r^2)*1/l*ck*X)
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