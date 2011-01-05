function experiment_stability_homeo(e)
% experiment_stability_homeo : compares the efficiency of the sparsification
% at the end of the learning for different homeostasis parameters

%## Author : Laurent Perrinet <Laurent.Perrinet@incm.cnrs-mrs.fr>
%## This software is distributed under the terms of the GPL

% stored default parameters
e=default(e.where);

nom_exp=[e.where '/stability_homeo.mat'];
v_param=logspace(-1,1,9);%
v_param(1) =0;
%v_param=(10.^(-2:.5:2)) .* ((10.^(-2:.5:2))>0.015);

if ~(switch_lock(e,nom_exp)==-1),
    add_log(e,'Make learning with experiment_stability_homeo');
    var_eta_cgf=e.var_eta_cgf*v_param; % used in CGF
    var_eta_ssc=e.var_eta_ssc*v_param; % used in SSC
    X=get_patch(e);
    res.total=sum(X(:).^2)/e.L/e.batch_size;
    for i_param=1:length(v_param),
        % sets a similar initialization for weights
        A_rand = rand(e.L,e.M)-0.5; A_rand = A_rand*diag(1./sqrt(sum(A_rand.*A_rand)));
        gain_rand=sqrt(sum(A_rand.*A_rand))';
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Sparsenet / MP_Sparsenet %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for i_method=1:2
            e=default(e.where);
            if i_method==1,
                e.var_eta_cgf = var_eta_cgf(i_param);
                e.Method='cgf';
                n.gain=gain_rand;
                n.S_var=e.VAR_GOAL*ones(e.M,1);
            else
                e.var_eta_ssc = var_eta_ssc(i_param);
                e.Method='ssc';
                n.Pz_j=1/e.n_quant*ones(e.n_quant,e.M); n.Mod=cumsum(n.Pz_j);
            end

            nom_exp_=[e.where '/stability_homeo_' e.Method '_' num2str(i_param) '.mat'];
            if ~switch_lock(e,nom_exp_),
                switch_lock(e,nom_exp_,1) %lock   
                add_log(e,['experiment_stability_homeo - ' e.Method ' -i_param= ' num2str(i_param) '/' num2str(length(v_param))]);
                n.A = A_rand;
                [n,s]=sparsenet(n,e);
                save('-v7',nom_exp_,'n','s');
            end
        end
    end
end

% checking that everything is finished
if ~(switch_lock(e,nom_exp)==-1),
    switch_lock(e,nom_exp,0) % unlock by default
    for i_param=1:length(v_param),
        for i_method=1:2
            if i_method==1,
                e.Method='cgf';
            else;
                e.Method='ssc';
            end

            nom_exp_=[e.where '/stability_homeo_' e.Method '_' num2str(i_param) '.mat'];
            if ~(switch_lock(e,nom_exp_)==-1),
                switch_lock(e,nom_exp_) % displays who locks the experiment
                switch_lock(e,nom_exp,1) % lock it for this run
            end
        end
    end
end


if ~switch_lock(e,nom_exp),
    switch_lock(e,nom_exp,1) %lock   

    add_log(e,'Make coding with experiment_stability_homeo');
    clear res sparseness_end L0_end
    var_eta_cgf=e.var_eta_cgf*v_param; % used in CGF
    var_eta_ssc=e.var_eta_ssc*v_param; % used in SSC
    X=get_patch(e);
    res.total=sum(X(:).^2)/e.L/e.batch_size;
    for i_param=1:length(v_param),
        % sets a similar initialization for weights
        A_rand = rand(e.L,e.M)-0.5; A_rand = A_rand*diag(1./sqrt(sum(A_rand.*A_rand)));
        gain_rand=sqrt(sum(A_rand.*A_rand))';
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Sparsenet / MP_Sparsenet %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for i_method=1:2
            e=default(e.where);
            if i_method==1,
                e.Method='cgf';
            else;
                e.Method='ssc';
            end

            nom_exp_=[e.where '/stability_homeo_' e.Method '_' num2str(i_param) '.mat'];
            load(nom_exp_)
            
            if ~exist([e.where '/fig_stability_homeo_' e.Method '_' num2str(i_param) '.png'],'file'),
                imwrite((tile(n.A)+1)/2,[e.where '/fig_stability_homeo_' e.Method '_' num2str(i_param) '.png'])
            end
            
            % now test the efficiency
            e.batch_size=10000; % a big batch
            X=get_patch(e);
            if i_method==1,
                S=cgf_fitS(n.A,X,e.noise_var_cgf,e.beta,e.sigma,e.tol);
                for i_batch=1:e.batch_size,
                    max_S=max(abs(S(:,i_batch)));
                    % second sparsification using cgf to be fair compared to MP
                    ind=find(abs(S(:,i_batch))>e.frac*max_S); % the basis functions we use
                    S_(ind,i_batch)=cgf_fitS(n.A(:,ind),X(:,i_batch),e.noise_var_cgf,e.beta,e.sigma,e.tol);
                end
            else
                e.switch_Mod = 0;
                S=mp_fitS(n.A,X,e.noise_var_ssc,e.frac,e.switch_choice,n.Mod,e.switch_Mod, 0, e.switch_sym);      
            end
            E=X-n.A*S;

            res.end(i_param,i_method)=sum(E(:).^2)/e.L/e.batch_size;
            sparseness_end(i_param,i_method)=e.beta/e.sigma*sum(log(1+S(:).^2/e.sigma^2))/e.M/e.batch_size;
            L0_end(i_param,i_method)=sum(S(:)~=0)/e.M/e.batch_size;
        end
    end

    save('-v7',nom_exp,'res','sparseness_end','L0_end','v_param')
    add_log(e,'Made coding with experiment_stability_homeo');

end
if switch_lock(e,nom_exp)==-1;

    load(nom_exp)
    %%%%% PLOTS
    % displays end of learning perfromance for different learnig rates
    % information transmission
    if ~exist([e.where '/fig_stability_homeo_energy.eps'],'file'),
        figure(3),set(gcf, 'visible', 'off'),
        semilogx(v_param,res.end'),
        axis tight,
        xlabel('parameter'),ylabel('Mean residual Energy'),
        legend('cg','ssc','Location','East')
        fig2pdf([e.where '/fig_stability_homeo_energy'],10,10)
    end
    if ~exist([e.where '/fig_stability_homeo_sparseness.eps'],'file'),
        % Sparseness
        figure(4),set(gcf, 'visible', 'off'),
        semilogx(v_param,sparseness_end),
        axis tight,
        xlabel('parameter'),ylabel('Olshausen''s sparseness'),
        legend('cgf','ssc','Location','East')
        fig2pdf([e.where '/fig_stability_homeo_sparseness'],10,10)
    end
    if ~exist([e.where '/fig_stability_homeo_cost.eps'],'file'),

        figure(5),set(gcf, 'visible', 'off'),
        semilogx(v_param,(1/e.noise_var_cgf)*res.end + sparseness_end*e.M/e.L),
        axis tight,
        xlabel('parameter'),ylabel('Olshausen''s cost'),
        legend('cgf','ssc','Location','East')
        fig2pdf([e.where '/fig_stability_homeo_cost'],10,10)
    end

end
