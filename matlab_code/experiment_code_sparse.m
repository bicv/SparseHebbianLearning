function experiment_code_sparse(e)
% experiment_code_sparse.m : compares coding after learning with both methods
% ---------------------------------------------------------------------------
% depends on the learning done in experiment_learn.m

% this experiment, we use the learnt basis functions from
% experiemnt_learn.m and compute the resulting L1-sparsity of the code

% this is meant to get the best v_noise_var parameter for the Olshausen
% cost

%## Author : Laurent Perrinet <Laurent.Perrinet@incm.cnrs-mrs.fr>
%## This software is distributed under the terms of the GPL

% compare efficiency of coding with the final basis functions
e=default(e.where); %loads default parameters


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% in this experiment, we modify the compromise between precision and
% sparsity (by tweaking noise_var) to compare the relative efficiency of both methods
nom_exp=[e.where '/L1_stats.mat'];
if switch_lock(e,[e.where '/CG.mat'])==-1 && switch_lock(e,[e.where '/MP.mat'])==-1 && ...
        ~switch_lock(e,nom_exp),
    switch_lock(e,nom_exp,1) % lock
    add_log(e,'Make learning with experiment_code_sparse');
    % gets a batch of imagelets
    e.batch_size=100*e.batch_size; % a big batch
    X=get_patch(e);
    res.total=sum(X(:).^2)/e.L/e.batch_size;
    e.eta_ssc=0; e.eta_cgf=0;  e.num_trials=1; % no learning, thank you
    
    
    % effect of the noise_var parameter
    v_noise_var = exp(log(10)*(-2:.5:2)); % range of different coding parameters 10**something
    
    for i_noise_var=1:length(v_noise_var)
        
        % sparsify by Conjugate Gradient
        S=zeros(e.M,e.batch_size); % initialize coeffs for LGM
        load([e.where '/CG.mat']),
        S=cgf_fitS(n.A,X,e.noise_var_cgf*v_noise_var(i_noise_var),e.beta,e.sigma,e.tol);
        E=X-n.A*S; % residual
        res.cg(i_noise_var)=sum(E(:).^2)/e.L/e.batch_size;
        sparseness.cg(i_noise_var)=e.beta/e.sigma*sum(log(1+S(:).^2/e.sigma^2))/e.M/e.batch_size;
        
        if e.switch_verbosity>=3,
            add_log(e,['res_cg=' num2str(res.cg(i_noise_var)) ';sparseness_cg=' num2str(sparseness.cg(i_noise_var)) ]);
        end;
        
        % sparsify by Matching Pursuit
        S=zeros(e.M,e.batch_size); % initialize coeffs for MP
        load([e.where '/MP.mat']),
        e.switch_Mod=0;
        e.switch_choice = 0; % we compare here the *coding* methods, not the full algorithm (CG has no spike decoding scheme
        S=mp_fitS(n.A,X,e.noise_var_ssc*v_noise_var(i_noise_var),1,e.switch_choice,n.Mod,e.switch_Mod, 0,e.switch_sym);%
        
        E=X-n.A*S; % residual
        
        L0_mp(i_noise_var)=sum(S(:)~=0)/e.M/e.batch_size;
        res.mp(i_noise_var)=sum(E(:).^2)/e.L/e.batch_size;
        sparseness.mp(i_noise_var)=e.beta/e.sigma*sum(log(1+S(:).^2/e.sigma^2))/e.M/e.batch_size;
        
        if e.switch_verbosity>=3,
            add_log(e,['res_mp=' num2str(res.mp(i_noise_var)) ';sparseness_mp=' num2str(sparseness.mp(i_noise_var)) ]);
        end;
        
    end
    
    cmd= ['rm -f ' e.where '/fig_sparse_* ' ];
    unix(cmd)
    
    save('-v7',nom_exp,'res','v_noise_var','sparseness','L0_mp')
    add_log(e,'Finished  with experiment_code_sparse');
    
end;
if switch_lock(e,nom_exp)==-1,
    
    load(nom_exp),%
    
    color_cg = 'b--';%'k--';%
    color_mp = 'r';%'k';%
    
    if ~exist([e.where '/fig_sparse_energy.eps'],'file'),
        
        figure(5),set(gcf, 'visible', 'off'),
        plot(e.noise_var_cgf*v_noise_var,res.cg/res.total,color_cg,...
            e.noise_var_ssc*v_noise_var,res.mp/res.total,color_mp),
        axis tight,%
        xlabel('Noise variance parameter'), ylabel('Energy (L2 norm)'),legend('cg','ssc','Location','East')
        fig2pdf([e.where '/fig_sparse_energy'],12,10)
    end
    if ~exist([e.where '/fig_sparse_olshausen.eps'],'file'),
        
        figure(6),set(gcf, 'visible', 'off'),
        plot(e.noise_var_cgf*v_noise_var,sparseness.cg,color_cg,...
            v_noise_var,sparseness.mp,color_mp),
        axis tight,%
        xlabel('Noise variance parameter'), ylabel('Olshausen''s sparseness'),
        legend('cg','ssc','Location','East'),hold off
        fig2pdf([e.where '/fig_sparse_olshausen'],12,10)
    end
    
    if ~exist([e.where '/fig_sparse_L0.eps'],'file'),
        
        figure(6),set(gcf, 'visible', 'off'),
        plot(e.noise_var_cgf*v_noise_var,L0_mp),
        axis tight,%
        xlabel('Noise variance parameter'), ylabel('L0 mp'),
        fig2pdf([e.where '/fig_sparse_L0'],12,10)
    end
    
    
    if ~exist([e.where '/fig_sparse_cost.eps'],'file'),
        
        % I don't really know if it's on the sum or on the mean : the sum
        % appears to be used in the code but it's certainly a bad habit since
        % the code is dependent on the overcompleteness of the representation
        % (that is the ratio e.M/e.L). Let's use here the sum and check at the
        % same how the efficiency varries with the noise_var parameter
        
        cost_mp=(1./(e.noise_var_cgf*v_noise_var)).*res.mp + sparseness.mp*e.M/e.L;
        cost_cg=(1./(e.noise_var_cgf*v_noise_var)).*res.cg + sparseness.cg*e.M/e.L;
        [cost_mp_opt ind_mp]=min(cost_mp);
        [cost_cg_opt ind_cg]=min(cost_cg);
        
        figure(7),set(gcf, 'visible', 'off'),
        semilogx(e.noise_var_cgf*v_noise_var,cost_cg,color_cg,...
            e.noise_var_ssc*v_noise_var,cost_mp,color_mp,...
            e.noise_var_cgf*v_noise_var(ind_cg),cost_cg(ind_cg),'r*',e.noise_var_ssc*v_noise_var(ind_mp),cost_mp(ind_mp),'b*')
        hold on, plot(0,0,''), hold off
        axis tight,  grid on, 
        
        text(e.noise_var_cgf*v_noise_var(ind_cg),cost_cg(ind_cg),[' v_{noise} = ' num2str(e.noise_var_cgf*v_noise_var(ind_cg))])
        text(e.noise_var_ssc*v_noise_var(ind_mp),cost_mp(ind_mp),[' v_{noise} = ' num2str(e.noise_var_ssc*v_noise_var(ind_mp))])
        
        xlabel('Noise variance parameter'),ylabel('Olshausen''s cost'),legend('cg','ssc')
        fig2pdf([e.where '/fig_sparse_cost'],10,10)
    end
    
    if ~exist([e.where '/fig_sparse_phase.eps'],'file'),
        
        figure(8),set(gcf, 'visible', 'off'),
        plot(sparseness.cg,res.cg,color_cg,sparseness.mp,res.mp,color_mp)%, 
        hold on, plot(0,0,''), hold off
        axis tight,grid on,%
        xlabel('Olshausen''s sparseness'), ylabel('Residual Energy (L2 norm)'),
        legend('cg','ssc','Location','NorthEast'),hold off
        fig2pdf([e.where '/fig_sparse_phase'],10,10)
    end
    
end
