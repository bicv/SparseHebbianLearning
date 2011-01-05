% experiment_simple.m : simple learning experiment
% ------------------------------------------------
% performs a simple (and quick) experiment
% demonstrates the modelization framework : experiment_*.m files provide the
% initialization and procedure, the other (such as sparsenet.m) provide the
% core computations

%## Author : Laurent Perrinet <Laurent.Perrinet@incm.cnrs-mrs.fr>
%## This software is distributed under the terms of the GPL

clear all; % yes, it clears ALL previous computations (safer side strategy)
path(path,'../src')
%----------------------------------------------------------------
% Initialization of the network, learning and simulations parameters
%----------------------------------------------------------------
%loads default parameters and creates a directory to put all results%
if 1,
    e=default;
    n.A = rand(e.L,e.M)-0.5; n.A = n.A*diag(1./sqrt(sum(n.A.*n.A)));
    n.gain=ones(e.M,1);
    n.S_var=e.VAR_GOAL*ones(e.M,1);
    n.Pz_j=1/e.n_quant*ones(e.n_quant,e.M); n.Mod=cumsum(n.Pz_j);
    prettyformat(e)
    %add_log(e,'Make simple learning experiment');
else, % or loads some data directory to see if learning is stable
    e.where='../results/20100322T151819'; %
    e=default(e.where);
    %e.eta_ssc = 1/7
    load([e.where '/MP.mat']),%
end

%----------------------------------------------------------------
% begins simulation
%----------------------------------------------------------------
%     e.num_trials=1001; % the number of learning steps
e.switch_verbosity=4;
e.display_every=10;
e.Method='ssc'% e.video = 0;
[n,s]=sparsenet(n,e);
