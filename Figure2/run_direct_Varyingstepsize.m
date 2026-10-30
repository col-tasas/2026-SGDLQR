function run_SGD_decay()
% Zeroth-order SGD for LQR -- DECAYING / GROWING schedule
%   n_i   = n0 * ceil(i/Nn)
%   l_i   = l0 * ceil(i/Nl)
%   r_i   = r0 / ceil(sqrt(i)/cr)
%   eta_i = eta0 / ceil(i^(51/100)/ceta)
% Output: ck (numMC x iteration), relative cost gap (J(K_i)-J*)/J*.
% Diverged runs are padded with NaN -> use mean(ck,1,'omitnan'), var(ck,0,1,'omitnan').

%% ================= user settings =================
numMC      = 1;          % number of Monte Carlo runs
iteration  = 6600000;      % iterations per run
n0   = 100;  Nn   = 1100000;   % samples:  n_i = n0*ceil(i/Nn)
l0   = 10;   Nl   = 1100000;   % horizon:  l_i = l0*ceil(i/Nl)
r0   = 0.01; cr   = 3300000;  % radius:   r_i = r0/ceil(sqrt(i)/cr)
eta0 = 0.001; ceta = 1000000;      % step:     eta_i = eta0/ceil(i^0.51/ceta)
                             % (ceta = 1 as specified; your old commented code used 250)
usePar     = false;        % parfor over MC runs (Parallel Computing Toolbox); false = serial
printEvery = 5000;        % progress print interval
rng('shuffle'); seedBase = randi(2^31 - 1 - numMC);
outFile    = 'dataSGD_decay.mat';

%% ================= system =================
A = [1,-1.13,-0.65,-0.807,1.59; 0,0.77,0.32,-0.98,-2.97; 0,0.12,0.02,0.0,-0.36; 0,0.01,0.01,-0.03,-0.04; 0,0.14,-0.09,0.29,0.76];
B = [89.20,-50.17,1.13,-19.35; 5.22,6.36,0.23,-0.32; -9.47,5.93,-0.12,0.99; -0.32,0.32,-0.01,-0.01; -4.53,3.21,-0.14,0.09];
nx = size(A,2); nu = size(B,2);
Q = eye(nx); R = eye(nu);
[~,K0]    = idare(A,B,40*Q,R);  K0 = -K0;
[Pstar,~] = idare(A,B,Q,R);     trPstar = trace(Pstar);
Sigma_w = 1e-3*eye(nx);  Sigma_0 = 1e-6*eye(nx);
Lw = chol(Sigma_w,'lower');  L0 = chol(Sigma_0,'lower');

sched = @(i) deal(n0*ceil(i/Nn), l0*ceil(i/Nl), r0*sqrt(cr/(max(i,cr))), eta0*(ceta/max(i,ceta))^(51/100));

%% ================= Monte Carlo =================
ck = nan(numMC, iteration);
nWorkers = 0; if usePar, nWorkers = Inf; end
tic
for num = 1:numMC
    rng(seedBase + num, 'twister');
    row = oneRun(K0,A,B,Q,R,Lw,L0,trPstar,iteration,sched,printEvery,num);
    ck(num,:) = row;
    saveRun(sprintf('%s_run%02d.mat', outFile(1:end-4), num), row);   % per-run checkpoint
end
elapsed = toc;

params = struct('numMC',numMC,'iteration',iteration,'n0',n0,'Nn',Nn,'l0',l0,'Nl',Nl, ...
                'r0',r0,'cr',cr,'eta0',eta0,'ceta',ceta, ...
                'seedBase',seedBase,'K0',K0,'trPstar',trPstar,'elapsed',elapsed);
save(outFile, 'ck', 'params', '-v7.3');
fprintf('Done in %.1f s, saved to %s\n', elapsed, outFile);
end

%% ================= local functions =================
function ckRow = oneRun(K0,A,B,Q,R,Lw,L0,trPstar,iteration,sched,printEvery,num)
nx = size(A,1);
I2 = eye(nx^2);
ckRow = nan(1, iteration);
K = K0;
for i = 1:iteration
    Acl = A + B*K;
    if ~all(isfinite(Acl(:))) || max(abs(eig(Acl))) >= 1
        fprintf('run %d: closed loop unstable at i = %d, stopping (rest = NaN)\n', num, i);
        break
    end
    % J(K) = trace(P_K),  P_K = Acl' P_K Acl + Q + K'RK   (Kronecker solve, faster than dlyap)
    Qk = Q + K'*R*K;
    P  = reshape((I2 - kron(Acl',Acl')) \ Qk(:), nx, nx);
    ckRow(i) = (trace(P) - trPstar)/trPstar;

    [n,l,r,eta] = sched(i);
    K = K - eta*zoGradient(K,A,B,Q,R,Lw,L0,n,l,r);

    if mod(i,printEvery) == 0
        fprintf('run %d  i = %d  ck = %.4e  n=%d l=%d r=%.3g eta=%.3g\n', num, i, ckRow(i), n, l, r, eta);
    end
end
end

function g = zoGradient(K,A,B,Q,R,Lw,L0,n,l,r)
% Vectorised one-point zeroth-order estimate: all n rollouts simulated in parallel.
[nu,nx] = size(K); d = nu*nx;
U = randn(d,n);
U = U .* (r ./ sqrt(sum(U.^2,1)));          % uniform on Frobenius sphere of radius r
Kall = reshape(K(:) + U, nu, nx, n);        % perturbed gains K + U_j
x = L0*randn(nx,n);                         % x0 ~ N(0,Sigma_0)
c = zeros(1,n);
for p = 1:l
    u = reshape(sum(Kall .* reshape(x,1,nx,n), 2), nu, n);   % u_j = (K+U_j) x_j
    c = c + sum(x.*(Q*x),1) + sum(u.*(R*u),1);               % x'(Q+K'RK)x
    if p < l
        x = A*x + B*u + Lw*randn(nx,n);
    end
end
g = reshape(U*c.', nu, nx) * (d/(r^2*l*n));
end

function saveRun(fname, ckRow) %#ok<INUSD>
save(fname, 'ckRow');
end
