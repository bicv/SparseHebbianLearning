%  Learning image patches : comparison of 2 sparsification techniques
% -------------------------------------------------------------------
% Contents

%## Author : Laurent Perrinet <Laurent.Perrinet@incm.cnrs-mrs.fr>
%## This software is distributed under the terms of the GPL

clear all
% e.where='../results/'; %

path(path,'../src')
%----------------------------------------------------------------
% Initialization of the network, learning and simulations parameters
%----------------------------------------------------------------
if 1% new session
    e=default;% switches to run computation and figures
else% restores an old session
    e=default(e.where);
end

%load(e.image_base); % a stacked 3-d matrix of natural images
%global IMAGES

prettyformat(e)

%experiment_simple
%% with the default parameters do :
experiment_stats_images(e)

% generates a learning and computes statistics
experiment_learn(e)
% how efficient is the homeostatic constraint?
experiment_nonhomeo(e)
experiment_stability_homeo(e)

%% control experiments
% and if we allow the filter to be non-symmetric in response (no ON-OFF)
experiment_symmetric(e)

% compares final coding efficiency
experiment_code_efficiency(e)
experiment_code_histogram(e)
experiment_code_sparse(e)

% try to change some default parameters
experiment_stability_eta(e)
experiment_perturb_ssc(e)

% compares different parameters for cgf and ssc
experiment_stability_cgf(e)
experiment_stability_ssc(e)
% even more fun for your cluster
experiment_stability_imagebase(e)
experiment_stability_oc(e)
experiment_oomp(e)
