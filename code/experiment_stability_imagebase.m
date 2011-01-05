function experiment_stability_imagebase(e)
% experiment_stability_imagebase.m : learning with different databases
% --------------------------------------------------------------------

% TODO : study how reproducible a base is wrt initial A_rand even with a random choice of images.

% stored default parameters
e=default(e.where);
load(e.image_base); % a stacked 3-d matrix of natural images

%%  Learning
%% ----------
% generates 2 learning with all stats and video
nom_exp_MP_yelmo=[e.where '/MP_yelmo.mat'];
nom_exp_MP_icabench=[e.where '/MP_icabench_decorr.mat'];
if ~(switch_lock(e,nom_exp_MP_yelmo)==-1 && switch_lock(e,nom_exp_MP_icabench)==-1),
    add_log(e,'Make learning with experiment_stability_imagebase');

    A_rand = rand(e.L,e.M)-0.5; A_rand = A_rand*diag(1./sqrt(sum(A_rand.*A_rand)));
    gain_rand=sqrt(sum(A_rand.*A_rand))';
    if ~exist([e.where '/fig_map_rand.png'],'file'),
        imwrite((tile(A_rand)+1)/2,[e.where '/fig_map_rand.png'])
    end
    % MP
    if ~switch_lock(e,nom_exp_MP_yelmo),
        switch_lock(e,nom_exp_MP_yelmo,1) % lock
        e=default(e.where); %loads default parameters
        e.Method='ssc'; e.video = 1;
	e.image_base='../data/IMAGES_yelmo.mat';%_icabench'; %
        e.video = 0;
        n.A = A_rand;
        n.Pz_j=1/e.n_quant*ones(e.n_quant,e.M); n.Mod=cumsum(n.Pz_j);
        [n,s]=sparsenet(n,e);
        save('-v7',nom_exp_MP_yelmo,'n','s')
    end
    % MP
    if ~switch_lock(e,nom_exp_MP_icabench),
        switch_lock(e,nom_exp_MP_icabench,1) % lock
        e=default(e.where); %loads default parameters
        e.Method='ssc'; e.video = 1;
        e.image_base='../data/IMAGES_icabench_decorr.mat';%_icabench'; %
        e.video = 0;

        n.A = A_rand;
        n.Pz_j=1/e.n_quant*ones(e.n_quant,e.M); n.Mod=cumsum(n.Pz_j);
        [n,s]=sparsenet(n,e);
        save('-v7',nom_exp_MP_icabench,'n','s')
    end
    cmd= ['rm -f ' e.where '/fig_stability_imagebase_* '];
    unix(cmd)
    add_log(e,'Made learning with experiment_stability_imagebase');

end;



if switch_lock(e,nom_exp_MP_yelmo)==-1 && switch_lock(e,nom_exp_MP_icabench)==-1,
    load(nom_exp_MP_yelmo),
    if ~exist([e.where '/fig_stability_imagebase_yelmo.png'],'file'),
        add_log(e,'Generate image of the basis functions');
        imwrite((tile(n.A)+1)/2,[e.where '/fig_stability_imagebase_yelmo.png'])
    end

    load(nom_exp_MP_icabench),
    if ~exist([e.where '/fig_stability_imagebase_icabench.png'],'file'),
        add_log(e,'Generate image of the basis functions');
        imwrite((tile(n.A)+1)/2,[e.where '/fig_stability_imagebase_icabench.png'])
    end

end;
