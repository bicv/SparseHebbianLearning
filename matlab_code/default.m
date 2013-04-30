function e=default(where)
% loads default experimental ('e') parameters

% e.where = directory where the results are stored

% this function loads default parameters (without the basis functions
% and the homeostatic vectors that were stored in a mat file. if no
% directory was provided, creates a new one.

% these parameters often correspond to realisitic and experimentally
% adequate values. 


%----------------------------------------------------------------
% sets where and what to do
%----------------------------------------------------------------
if  nargin < 1, % stores all results in a new directory
    e.where=['../results/' datestr(now, 30)]; % directory named with ISO date+hour
    disp('generating new directory for this experiment')
    mkdir(e.where) % create directory for data + figures
    mkdir([e.where '/tmp']) % for the PNGs
    e.version='mp_sparsenet_2.2';
										  
    % L = length image (therefore its surface) M=number of filters
    % L and M should be square resp. for numeric and graphic reasons.
    e.L=16^2; e.M=18^2; %
    e.num_trials=40001; % the number of learning steps
    e.eta_cgf=1/25; % learning rate
    e.eta_ssc=1/25; % learning rate
    e.switch_sym=1; % we use ON/OFF symmetry

    % experiments' parameters and monitoring
    %----------------------------------------------------------------
    e.image_base='../data/IMAGES.mat';%._yelmo';%';.mat';%_icabench'; %
    % name of the image database see get_patch.m for a description (or see diy.m)
    e.batch_size=100; % how many 'imagelets' do we take for each learning step?


    e.display_every=50; % delay between snapshots (to the screen or movie)

    e.n_mean=25; % smoothing parameter for graphs
    e.switch_verbosity=1; % verbosity level, in time scale 1= one message every
    % display_every learning, 2=every single learning, 4=every coding step
    e.switch_stats=1; % do I make statistics?
    e.video=0; % and video images?

    % learning parameters
    %----------------------------------------------------------------
    e.frac=.25; % we take *at most* frac active filters in the Matching Pursuit during the lerning phase
    e.noise_var_cgf= 0.03; % noise_var/VAR_GOAL gives the desired precision (*mean* energy of the residual) stopping criteria for CGF (it considers the residual is noise)
    e.noise_var_ssc= 0.002; % relative threshold for SSC corresponding to an
                           % estimate of the ratio of background
                           % noise over total signal energy

    % it's the same as in SPARSENET and should be computed in comparison
    % with experiment_stats_images (mean of the diagonal of the covariance
    % matrix gives the mean energy of the signal and the optimization in
    % the experiment_code_*.m files

    % homeostatic parameters (the inverse gives the time constant in learning step time)
    % for MP (modifying homeostatic gain control)
    e.var_eta_ssc = 1/20; % used to ensure that all filters
    % fire  equally the denominator gives the order of learning steps
    % used to estimate the probability
    e.switch_choice = e.var_eta_ssc > 0;
    e.n_quant = 512;
    e.switch_Mod=1;
    % for CG (we play on the fact that norm of the filters don't play a
    % role in MP)
    e.var_eta_cgf=1/1000; % used to ensure non degenerate solutions smoothing of the variance estimate
    e.alpha=0.02; % multiplicative hebbian learning rule learning constant for the homeostasy
    % desired variance of the S coefficients
    e.VAR_GOAL=.10; %
    % additional paremeters for cgf as in SparseNet
    e.beta=0.2; % /* prior steepness */
    e.sigma=0.1; %; % /* scaling parameter for prior */
    e.tol=.0031; % parameter to stop CG when the gradient is to low in amplitude (0.001 is the default in cgf_fits.m from Olshausen's SparseNet)
    save('-v7',[e.where '/default.mat'],'e')
else % use an existing set of parameters
    load([where '/default.mat'])
end
