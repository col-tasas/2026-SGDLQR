% Plot from the small statistics files written by stats_noisy_grad.m
% (statsNoisyGrad_case<c>.mat) in the style of Fig. 4:
% mean over the NON-diverged runs + confidence interval (shaded),
% and next to each curve the number of diverged runs.
% Only plotting here -> re-run freely to adjust the figure.
% Cases whose stats file does not exist yet are skipped.
%% ---------------- settings ----------------
caseIdx = 1:4;
% legend text (LaTeX): [] = generated from the saved params; or give 4 strings by hand
labels  = [];
colors = {[1 0 1], [0 0 1], [1 0 0], [0 0 0]};   % magenta, blue, red, black
styles = {'-.', '--', ':', '-'};
widths = [1.5, 1.5, 1.5, 1.5];
level      = 0.999;      % confidence level of the band
faceAlpha  = 0.15;       % transparency of the band
txtOffset  = [1.15, 1.0];      % text position = last point .* [x-factor, y-factor] (log axes)
txtFormat  = '%d/%d diverged'; % annotation text
yLim       = [0.05, 2];        % [] = automatic
nPlotFig   = 2000;             % points per curve actually drawn (stats files keep all points)
saveFig    = true;             % false: skip the .fig export
outName    = 'fig_sgd_bias';   % exported as .pdf and .fig
fs         = 12;               % font size for all text
%% ---------------- load statistics ----------------
nC   = numel(caseIdx);
have = false(1,nC);
mu = cell(1,nC);  lo = cell(1,nC);  hi = cell(1,nC);  idx = cell(1,nC);
nDiv = zeros(1,nC);  nRun = zeros(1,nC);  lab = cell(1,nC);
for k = 1:nC
    f = sprintf('statsNoisyGrad_case%d.mat', caseIdx(k));
    if ~exist(f, 'file')
        fprintf('%s not found, skipped\n', f);  continue
    end
    S = load(f);
    if exist('tinv', 'file')
        q = tinv((1+level)/2, max(S.n-1,1));
    else
        q = sqrt(2)*erfinv(level) * ones(size(S.n));
    end
    hw = q .* sqrt(S.s2 ./ S.n);
    % log-spaced subset of the stored points for drawing
    tgt = logspace(0, log10(double(S.idx(end))), nPlotFig);
    sel = unique(interp1(double(S.idx), 1:numel(S.idx), tgt, 'nearest'));
    idx{k} = S.idx(sel);  mu{k} = S.mu(sel);
    lo{k}  = S.mu(sel) - hw(sel);  hi{k} = S.mu(sel) + hw(sel);
    nDiv(k) = S.nDiv;  nRun(k) = S.nRun;
    if isempty(labels), lab{k} = makeLabel(caseIdx(k), S.params); else, lab{k} = labels{k}; end
    have(k) = true;
end
if ~any(have), error('no stats file found -- run stats_noisy_grad first'); end
%% ---------------- plot ----------------
fig = figure('Units','centimeters','Position',[2 2 17 8], 'Color','w');
ax  = axes(fig); hold(ax,'on');
hL  = gobjects(1,nC);
for k = find(have)                             % bands first
    ok = isfinite(lo{k}) & isfinite(hi{k});
    x  = idx{k}(ok);
    fill(ax, [x, fliplr(x)], [max(lo{k}(ok),1e-12), fliplr(hi{k}(ok))], colors{caseIdx(k)}, ...
         'FaceAlpha', faceAlpha, 'EdgeColor', 'none', 'HandleVisibility', 'off');
end
for k = find(have)                             % lines + annotation on top
    c     = caseIdx(k);
    okm   = isfinite(mu{k});
    hL(k) = plot(ax, idx{k}(okm), mu{k}(okm), styles{c}, 'Color', colors{c}, 'LineWidth', widths(c));
    jEnd  = find(okm, 1, 'last');
    if ~isempty(jEnd)
        text(ax, idx{k}(jEnd)*txtOffset(1), mu{k}(jEnd)*txtOffset(2), ...
             sprintf(txtFormat, nDiv(k), nRun(k)), 'Color', colors{c}, ...
             'Interpreter','latex', 'FontSize', fs, 'VerticalAlignment','middle');
    end
end
set(ax, 'XScale','log', 'YScale','log', 'TickLabelInterpreter','latex', 'FontSize', fs);
grid(ax,'on'); grid(ax,'minor'); box(ax,'on');
xlim(ax, [1, max(cellfun(@(v) v(end), idx(have)))]);
if ~isempty(yLim), ylim(ax, yLim); end
xlabel(ax, 'Iteration $i$', 'Interpreter','latex', 'FontSize', fs);
ylabel(ax, '$\frac{C(K_i)-C(K^*)}{C(K^*)}$', 'Interpreter','latex', 'FontSize', fs);
legend(ax, hL(have), lab(have), 'Interpreter','latex', 'Location','southwest', 'FontSize', fs);
if saveFig, savefig(fig, [outName '.fig']); end
exportgraphics(fig, [outName '.pdf'], 'ContentType','vector');
%% ---------------- local function ----------------
function s = makeLabel(c, p)
% legend string in the same layout as the hand-written labels, values taken from params
etaDec  = mod(c-1,2) == 1;                    % cases 2,4: eta decays
betaDec = c >= 3;                             % cases 3,4: beta decays
if etaDec, sEta = sprintf('$\\eta_i = %gO(i^{-%g})$', p.eta0, p.aEta);
else,      sEta = sprintf('$\\eta_i = %g$', p.eta0);
end
if betaDec, sBeta = sprintf('$\\|\\Delta_i\\| \\leq %gO(i^{-%g})$', p.beta0, p.aBeta);
else,       sBeta = sprintf('$\\|\\Delta_i\\| \\leq %g$', p.beta0);
end
if etaDec, s = [sEta '  ' sBeta];
else,      s = [sEta ' \qquad$~~~~~~' sBeta(2:end)];   % one math env, avoids "$$"
end
end