function experiment_stability_eta(e)
% compares the efficiency of the sparsification
% at the end of the learning for different learning rates

%## Author : Laurent Perrinet <Laurent.Perrinet@incm.cnrs-mrs.fr>
%## This software is distributed under the terms of the GPL

nom_exp=[e.where '/stability_eta.mat'];
v_eta=logspace(-1,1,9);
if ~(switch_lock(e,nom_exp)==-1),

    add_log(e,'Make learning with experiment_stability_eta');
    for i_eta=1:length(v_eta),
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
                n.gain=gain_rand;
                n.S_var=e.VAR_GOAL*ones(e.M,1);
                e.eta_cgf = e.eta_cgf * v_eta(i_eta);
            else;
                e.Method='ssc';
                n.Pz_j=1/e.n_quant*ones(e.n_quant,e.M); n.Mod=cumsum(n.Pz_j);
                e.eta_ssc = e.eta_ssc * v_eta(i_eta);
            end

            nom_exp_=[e.where '/stability_eta_' e.Method '_' num2str(i_eta) '.mat'];
            if ~switch_lock(e,nom_exp_),
                switch_lock(e,nom_exp_,1) %lock
                add_log(e,['experiment_stability_eta - ' e.Method ' -i_eta= ' num2str(i_eta) '/' num2str(length(v_eta))]);
                n.A = A_rand;
                [n,s]=sparsenet(n,e);
                save('-v7',nom_exp_,'n','s')
            end
        end
    end
end;

if ~(switch_lock(e,nom_exp)==-1),
% while running, things may have changed on the cluster: check if one simulation is still running
switch_lock(e,nom_exp, 0) % unlock by default   
for  i_method=1:2
    if i_method==1,
        e.Method='cgf';
    else;
        e.Method='ssc';
    end

    for i_eta=1:length(v_eta),
        nom_exp_=[e.where '/stability_eta_' e.Method '_' num2str(i_eta) '.mat'];
        if ~(switch_lock(e,nom_exp_)==-1),
            switch_lock(e,nom_exp, 1) % lock if one simulation is missing or running   
        end
    end
end
if ~switch_lock(e,nom_exp)
    add_log(e,'Made learning with experiment_stability_eta');
end
end

if ~switch_lock(e,nom_exp),
    switch_lock(e,nom_exp,1) %lock

    add_log(e,'Make coding with experiment_stability_eta');

    for i_eta=1:length(v_eta),
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Sparsenet / MP_Sparsenet %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        e=default(e.where);
        e.batch_size=100*e.batch_size; % a big batch
        X=get_patch(e);
        for i_method=1:2

            if i_method==1,
                e.Method='cgf';
            else;
                e.Method='ssc';
            end

            nom_exp_=[e.where '/stability_eta_' e.Method '_' num2str(i_eta) '.mat'];
            load(nom_exp_)
            if ~exist([e.where '/fig_stability_eta_' e.Method '_' num2str(i_eta) '.png'],'file'),
                imwrite((tile(n.A)+1)/2,[e.where '/fig_stability_eta_' e.Method '_' num2str(i_eta) '.png'])
            end

            if i_method==1, % CG
                S=cgf_fitS(n.A,X,e.noise_var_cgf,e.beta,e.sigma,e.tol);
                for i_batch=1:e.batch_size,
                    max_S=max(abs(S(:,i_batch)));
                    % second sparsification using cgf to be fair compared to MP
                    ind=find(abs(S(:,i_batch))>e.frac*max_S); % the set of basis functions we use
                    S_(ind,i_batch)=cgf_fitS(n.A(:,ind),X(:,i_batch),e.noise_var_cgf,e.beta,e.sigma,e.tol);
                end

            else, % MP
                S=mp_fitS(n.A,X)%
            end
            E=X-n.A*S;

            res.end(i_eta,i_method)=sum(E(:).^2)/e.L/e.batch_size;%
            sparseness_end(i_eta,i_method)=e.beta/e.sigma*sum(log(1+S(:).^2/e.sigma^2))/e.M/e.batch_size;
            L0_end(i_eta,i_method)=sum(S(:)~=0)/e.M/e.batch_size;%
        end
    end

    save('-v7',nom_exp,'res','sparseness_end','L0_end','v_eta') %
    add_log(e,'Made coding with experiment_stability_eta');

end;

if switch_lock(e,nom_exp)==-1,
    load(nom_exp)

    v_eta_ =v_eta;


    %%%%% PLOTS
    % displays end of learning perfromance for different learnig rates
    % information transmission

    if ~exist([e.where '/fig_stability_eta_energy.eps'],'file'),
        figure(3),set(gcf, 'visible', 'off'),
        semilogx(v_eta_, res.end'),
        axis tight,
        xlabel('factor of learning rate'),ylabel('Mean residual Energy'),
        legend('cgf x2','ssc','Location','East')
        fig2pdf([e.where '/fig_stability_eta_energy'],10,10)

    end
    if ~exist([e.where '/fig_stability_eta_sparseness.eps'],'file'),
        % Sparseness
        figure(4),set(gcf, 'visible', 'off'),
        semilogx(v_eta_,sparseness_end),
        axis tight, 
        xlabel('factor of learning rate'),ylabel('Olshausen''s sparseness'),
        legend('cgf x2','ssc','Location','East')
        fig2pdf([e.where '/fig_stability_eta_sparseness'],10,10)

    end
    if ~exist([e.where '/fig_stability_eta_ol_cost.eps'],'file'),
        figure(5),set(gcf, 'visible', 'off'),
        semilogx(v_eta_,(1/e.noise_var_cgf)*res.end + sparseness_end*e.M/e.L),
        axis tight, 
        xlabel('factor of learning rate'),ylabel('Olshausen''s cost'),
        legend('cgf x2','ssc','Location','East')
        fig2pdf([e.where '/fig_stability_eta_ol_cost'],10,10)

    end

end
