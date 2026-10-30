function run_indirect()
% Indirect (RLS model-based) policy gradient for LQR, Monte Carlo.
%   Two cases:  1) eta const   2) eta decay (smooth, no jumps)
%   eta_i = eta0 * (iEta/max(i,iEta))^aEta
%   (your old 0.05/ceil(i^0.51/50) ~ smooth version with iEta = 50^(1/0.51) ~ 2148)
% A run that errors or becomes unstable is stopped (rest = NaN) and the next run starts.
% Saves RAW ck{c} (maxnum x iteration) + params -> outFile (after every case).
% Plot with plot_noisy_grad.m (set its file name to outFile); same data format.
%% ================= user settings =================
maxnum     = 100;         % Monte Carlo runs per case
iteration  = 200000;     % iterations per run
usePar     = true;       % parfor over MC runs; false = serial
outFile    = 'dataIndirect.mat';
rng('shuffle');
eta0 = 0.05;  iEta = 2200;  aEta = 0.51;   % step size
t_0  = 50;                                 % initial data collection length
Sigma_g = eye(4);                          % exploration noise covariance
%% ================= system =================
A = [1,-1.13,-0.65,-0.807,1.59; 0,0.77,0.32,-0.98,-2.97; 0,0.12,0.02,0.0,-0.36; 0,0.01,0.01,-0.03,-0.04; 0,0.14,-0.09,0.29,0.76];
B = [89.20,-50.17,1.13,-19.35; 5.22,6.36,0.23,-0.32; -9.47,5.93,-0.12,0.99; -0.32,0.32,-0.01,-0.01; -4.53,3.21,-0.14,0.09];
nx = size(A,2); nu = size(B,2);
Q = eye(nx); R = eye(nu);
[~,K0]    = idare(A,B,40*Q,R);  K0 = -K0;
[Pstar,~] = idare(A,B,Q,R);     trPstar = trace(Pstar);
Sigma_w = 1e-3*eye(nx);

etaC = @(i) eta0;
etaD = @(i) eta0 * (iEta/max(i,iEta))^aEta;
cases = struct('name', {'\eta const', '\eta decay'}, 'eta', {etaC, etaD});
nC = numel(cases);
%% ================= Monte Carlo =================
ck = cell(1,nC);
nWorkers = 0; if usePar, nWorkers = Inf; end
seeds = reshape(randperm(2^31-1, maxnum*nC), maxnum, nC);   % distinct random seeds
params = struct('maxnum',maxnum,'iteration',iteration,'seeds',seeds,'eta0',eta0, ...
                'iEta',iEta,'aEta',aEta,'t_0',t_0,'Sigma_g',Sigma_g,'Sigma_w',Sigma_w, ...
                'caseNames',{{cases.name}},'casesDone',0,'elapsed',0);
tic
for c = 1:nC
    etaF = cases(c).eta;
    ckc = nan(maxnum, iteration);
    parfor (num = 1:maxnum, nWorkers)
        rng(seeds(num,c), 'twister');
        ckc(num,:) = oneRun(K0,A,B,Q,R,Sigma_w,Sigma_g,trPstar,iteration,t_0,etaF,num);
    end
    ck{c} = ckc;
    fprintf('case %d (%s) done, %.1f s elapsed, %d/%d runs stopped early\n', ...
            c, cases(c).name, toc, sum(isnan(ckc(:,end))), maxnum);
    params.casesDone = c;  params.elapsed = toc;
    save(outFile, 'ck', 'params', '-v7.3');
end
fprintf('All ck saved to %s\n', outFile);
end
%% ================= local functions =================
function ckRow = oneRun(K0,A,B,Q,R,Sigma_w,Sigma_g,trPstar,iteration,t_0,etaF,num)
[nu,nx] = size(K0);  nz = nx+nu;
I2 = eye(nx^2);
Lw = chol(Sigma_w,'lower');  Lg = chol(Sigma_g,'lower');
ckRow = nan(1, iteration);
try
    K = K0;  x = zeros(nx,1);
    % ---- initial batch least squares ----
    X1 = zeros(nz);  XY = zeros(nx,nz);
    for t = 1:t_0
        u = K*x + Lg*randn(nu,1);
        z = [x; u];
        X1 = X1 + z*z';
        xnew = A*x + B*u + Lw*randn(nx,1);
        XY = XY + xnew*z';
        x = xnew;
    end
    Est  = XY / X1;
    Pinv = inv(X1);                       % RLS: Pinv = inv(X1), updated by Sherman-Morrison
    % ---- RLS + gradient steps ----
    for i = 1:iteration
        u = K*x + Lg*randn(nu,1);
        z = [x; u];
        xnew = A*x + B*u + Lw*randn(nx,1);
        Pz   = Pinv*z;
        Pinv = Pinv - (Pz*Pz')/(1 + z'*Pz);                % = inv(X1 + z z')
        Est  = Est + (xnew - Est*z)*(z'*Pinv);             % same update as your code
        Aest = Est(:,1:nx);  Best = Est(:,nx+1:end);
        x = xnew;

        % true cost of current K
        Acl = A + B*K;
        if ~all(isfinite(Acl(:))) || max(abs(eig(Acl))) >= 1
            fprintf('run %d: true closed loop unstable at i = %d -> next run\n', num, i);
            return
        end
        Qk = Q + K'*R*K;
        Ptrue = reshape((I2 - kron(Acl',Acl')) \ Qk(:), nx, nx);
        ckRow(i) = (trace(Ptrue) - trPstar)/trPstar;

        % gradient from estimated model
        Ae = Aest + Best*K;
        if max(abs(eig(Ae))) >= 1
            fprintf('run %d: estimated closed loop unstable at i = %d -> next run\n', num, i);
            return
        end
        P  = reshape((I2 - kron(Ae',Ae')) \ Qk(:),      nx, nx);
        Pw = reshape((I2 - kron(Ae, Ae )) \ Sigma_w(:), nx, nx);
        Cest = 2*((R + Best'*P*Best)*K + Best'*P*Aest)*Pw;

        K = K - etaF(i)*Cest;
    end
catch ME
    fprintf('run %d: error "%s" -> next run\n', num, ME.message);
end
end
