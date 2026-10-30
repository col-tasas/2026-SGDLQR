% Post-processing for run_noisy_grad.m: reads the big data files
% (dataNoisyGrad_case<c>.mat, ckc = maxnum x iteration) and writes small
% statistics files statsNoisyGrad_case<c>.mat used by plot_noisy_grad.m.
%
% Statistics on the FULL data, only the stored points are log-subsampled.
% Rule: a run is dropped from its first NaN / negative value on (no filling),
%       single points with value > capOutlier are skipped.
% Stored per case: idx, mu, s2, n (mean, variance, #valid runs at each idx),
%                  nDiv, nRun, params, capOutlier.
% The confidence level is NOT fixed here -> can be changed in the plot script.
%% ---------------- settings ----------------
caseIdx    = 1:4;
capOutlier = 2;          % single points above this are skipped
nPlot      = 500000;     % log-spaced stored points
redo       = false;      % false: skip cases whose stats file already exists
%% ---------------- statistics ----------------
for c = caseIdx
    fData  = sprintf('dataNoisyGrad_case%d.mat',  c);
    fStats = sprintf('statsNoisyGrad_case%d.mat', c);
    if ~exist(fData, 'file')
        fprintf('%s not found, skipped\n', fData);  continue
    end
    if ~redo && exist(fStats, 'file')
        fprintf('%s exists, skipped (set redo = true to recompute)\n', fStats);  continue
    end
    t0 = tic;
    S = load(fData, 'ckc', 'params');  X = S.ckc;  params = S.params;  clear S
    idx = unique(round(logspace(0, log10(size(X,2)), nPlot)));
    [mu, s2, n, nDiv] = truncStats(X, idx, capOutlier);
    nRun = size(X,1);
    clear X
    save(fStats, 'idx', 'mu', 's2', 'n', 'nDiv', 'nRun', 'params', 'capOutlier', 'nPlot');
    fprintf('case %d: %d/%d runs diverged, stats saved (%.1f s)\n', c, nDiv, nRun, toc(t0));
end
%% ---------------- local function ----------------
function [mu, s2, n, nDiv] = truncStats(X, idx, cap)
% mean / variance over the runs still alive at each column of idx (no filling)
[~, T] = size(X);
bad = isnan(X) | X < 0;                       % first NaN / negative per run (on full data)
[has, firstBad] = max(bad, [], 2);
clear bad
firstBad(~has) = T + 1;
nDiv  = sum(has);                             % runs that diverged at some point
Xs    = X(:, idx);
valid = (idx < firstBad) & (Xs <= cap);       % alive and not an outlier
n     = sum(valid, 1);
Xs(~valid) = 0;
mu    = sum(Xs, 1) ./ n;
s2    = sum(((Xs - mu).^2) .* valid, 1) ./ (n - 1);
mu(n == 0) = NaN;
s2(n < 2)  = NaN;
end
