function run_noisy_grad()
% Policy gradient for LQR with ARTIFICIAL noise on the exact gradient:
%   K_{i+1} = K_i - eta_i * ( grad C(K_i) + X_i ),
%   X_i = M_i + W_i,  ||M_i||_F <= beta_i,  E||W_i||_F^2 = sigma2 (constant)
% Four cases:
%   1) eta const,  beta const
%   2) eta decay,  beta const
%   3) eta const,  beta decay
%   4) eta decay,  beta decay
% The variance sigma2 = E||W||_F^2 is kept CONSTANT in all cases.
% Runs only the cases in caseList; each case is saved to dataNoisyGrad_case<c>.mat
% (ckc: maxnum x iteration, avgc/vrc: mean/variance over MC runs, params).
% Plot with plot_noisy_grad.m after the cases are done.

%% ================= user settings =================
caseList   = 4;          % cases to run in this session, e.g. 1, [2 3], 1:4
maxnum     = 100;        % Monte Carlo runs per case
iteration  = 10000000;   % iterations per run
rng('shuffle');
% step size: eta_i = eta0 * (iEta2/max(i,iEta))^aEta  (const = eta0 up to iEta, then ~ i^-aEta)
eta0 = 0.05;   iEta = 5;  iEta2 = 5; aEta = 0.51;
% bias bound: beta_i = beta0 * (iN/max(i,iN))^aBeta ;  variance sigma2 constant
beta0 = 0.2;  iN = 5; iN2 = 5; aBeta = 0.5;
sigma2 = 0.00005;

% how the "mean" part M_i is generated:
%   'random'      : new random direction & magnitude every iteration (your original code)
%                   -> E[M_i] = 0, i.e. NOT a bias, just extra bounded zero-mean noise
%   'fixed'       : one random direction per run, ||M|| = beta_i  -> genuine (constant-direction) bias
%   'adversarial' : M_i = +beta_i * grad/||grad||, opposes descent  -> worst-case bias
biasMode = 'fixed';

capOutlier = 100;        % points with ck > capOutlier are dropped in the statistics

%% ================= system =================
A = [1,-1.13,-0.65,-0.807,1.59; 0,0.77,0.32,-0.98,-2.97; 0,0.12,0.02,0.0,-0.36; 0,0.01,0.01,-0.03,-0.04; 0,0.14,-0.09,0.29,0.76];
B = [89.20,-50.17,1.13,-19.35; 5.22,6.36,0.23,-0.32; -9.47,5.93,-0.12,0.99; -0.32,0.32,-0.01,-0.01; -4.53,3.21,-0.14,0.09];
nx = size(A,2); nu = size(B,2);
Q = eye(nx); R = eye(nu);
[~,K0]    = idare(A,B,40*Q,R);  K0 = -K0;
[Pstar,~] = idare(A,B,Q,R);     trPstar = trace(Pstar);
Sigma_w = 1e-3*eye(nx);

etaC  = @(i) eta0;
etaD  = @(i) eta0  * (iEta2/max(i,iEta))^aEta;
betaC = @(i) beta0;
betaD = @(i) beta0 * (iN2/max(i,iN))^aBeta;

cases = struct('name', {'\eta const, \beta const', '\eta decay, \beta const', ...
                        '\eta const, \beta decay', '\eta decay, \beta decay'}, ...
               'eta',  {etaC,  etaD,  etaC,  etaD}, ...
               'beta', {betaC, betaC, betaD, betaD});
nC = numel(cases);

%% ================= Monte Carlo =================
seeds = randperm(2^31-1, maxnum*nC);  seeds = reshape(seeds, maxnum, nC);   % 互不相同的随机种子
tic

for c = caseList
    etaF = cases(c).eta;  betaF = cases(c).beta;
    ckc = nan(maxnum, iteration);
    tc = tic;
    for num = 1:maxnum
        rng(seeds(num,c), 'twister');
        ckc(num,:) = oneRun(K0,A,B,Q,R,Sigma_w,trPstar,iteration,etaF,betaF,sigma2,biasMode);
        nAlive = nnz(~isnan(ckc(num,:)));
        if nAlive == iteration, st = 'ok';
        else,                   st = sprintf('diverged @ iter %d', nAlive+1);
        end
        fprintf('  case %d: run %3d/%d %s | %.1f s (case), %.1f s (total)\n', ...
                c, num, maxnum, st, toc(tc), toc);
    end
    [avgc, vrc] = truncatedStats(ckc, capOutlier);
    fprintf('case %d (%s) done, %.1f s elapsed, %d/%d runs diverged\n', ...
            c, cases(c).name, toc, sum(isnan(ckc(:,end))), maxnum);

    params = struct('maxnum',maxnum,'iteration',iteration,'seeds',seeds(:,c), ...
                    'eta0',eta0,'iEta',iEta,'iEta2',iEta2,'aEta',aEta, ...
                    'beta0',beta0,'iN',iN,'iN2',iN2,'aBeta',aBeta,'sigma2',sigma2, ...
                    'biasMode',biasMode,'capOutlier',capOutlier, ...
                    'caseName',cases(c).name,'elapsed',toc(tc));
    save(sprintf('dataNoisyGrad_case%d.mat', c), 'ckc', 'avgc', 'vrc', 'params', '-v7.3');
    fprintf('case %d saved, %.1f s elapsed\n', c, toc);
    clear ckc avgc vrc
end
end

%% ================= local functions =================
function ckRow = oneRun(K0,A,B,Q,R,Sigma_w,trPstar,iteration,etaF,betaF,sigma2,biasMode)
[nu,nx] = size(K0);
I2 = eye(nx^2);
ckRow = nan(1, iteration);
K = K0;
Mdir = randn(nu,nx);  Mdir = Mdir/norm(Mdir,'fro');      % used only in 'fixed' mode
for i = 1:iteration
    Acl = A + B*K;
    if ~all(isfinite(Acl(:))) || max(abs(eig(Acl))) >= 1
        break                                              % unstable -> rest stays NaN
    end
    % exact P_K, Sigma_K via Kronecker solves (replaces 3 dlyap calls per iteration)
    Qk = Q + K'*R*K;
    P  = reshape((I2 - kron(Acl',Acl')) \ Qk(:),      nx, nx);
    Pw = reshape((I2 - kron(Acl, Acl )) \ Sigma_w(:), nx, nx);
    ckRow(i) = (trace(P) - trPstar)/trPstar;
    grad = 2*((R + B'*P*B)*K + B'*P*A)*Pw;

    beta = betaF(i);
    switch biasMode
        case 'random'                                      % original generator
            M = randn(nu,nx);  M = (beta*rand()) * M/norm(M,'fro');
        case 'fixed'
            M = beta * Mdir;
        case 'adversarial'
            M = beta * grad/norm(grad,'fro');
    end
    W = sqrt(sigma2/(nu*nx)) * randn(nu,nx);              % E||W||_F^2 = sigma2

    K = K - etaF(i) * (grad + M + W);
end
end

function [avg, vr] = truncatedStats(X, cap)
% Same rule as your truncated_mean_with_bounds:
%   a run is dropped permanently from the first NaN (divergence) or negative value on,
%   single points with X > cap are skipped.  Also returns the variance over the same set.
[~, T] = size(X);
alive = true(size(X,1),1);
avg = nan(1,T);  vr = nan(1,T);
for k = 1:T
    alive = alive & ~isnan(X(:,k)) & (X(:,k) >= 0);
    valid = alive & (X(:,k) <= cap);
    if ~any(valid), break; end
    v = X(valid,k);
    avg(k) = mean(v);
    if numel(v) > 1, vr(k) = var(v); end
end
end
