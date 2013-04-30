function experiment_stability_ssc(e)
% compares the efficiency of the sparsification
% at the end of the learning for different frac and noise threshold

% dumbily copying-pasting code for both variables...


%## Author : Laurent Perrinet <Laurent.Perrinet@incm.cnrs-mrs.fr>
%## This software is distributed under the terms of the GPL


nom_exp=[e.where '/stability_ssc_frac.mat'];
v_frac= logspace(-.2,-.0,9);%

if ~(switch_lock(e,nom_exp)==-1),
    
    add_log(e,'Make learning with experiment_stability_ssc');
    for i_frac=1:length(v_frac),
        % sets a similar initialization for weights
        A_rand = rand(e.L,e.M)-0.5; A_rand = A_rand*diag(1./sqrt(sum(A_rand.*A_rand)));
        gain_rand=sqrt(sum(A_rand.*A_rand))';
        %%%%%%%%
        % aSSC %
        %%%%%%%%
        e=default(e.where);
        e.Method='ssc';
        n.Pz_j=1/e.n_quant*ones(e.n_quant,e.M); n.Mod=cumsum(n.Pz_j);
        e.frac_ssc = v_frac(i_frac);
        
        
        nom_exp_=[e.where '/stability_ssc_frac_' num2str(i_frac) '.mat'];
        if ~switch_lock(e,nom_exp_),
            switch_lock(e,nom_exp_,1) %lock
            add_log(e,['experiment_stability_ssc-i_frac= ' num2str(i_frac) '/' num2str(length(v_frac))]);
            n.A = A_rand;
            [n,s]=sparsenet(n,e);

            e.batch_size=100*e.batch_size; % a big batch
            X=get_patch(e);
            e.switch_Mod = 0; % we compare here the *coding* methods, not the full algorithm (CG has no spike decoding scheme)
            e.switch_choice = 0; % we compare here the *coding* methods, not the full algorithm (CG has no spike decoding scheme)
            S=mp_fitS(n.A,X,0,1,e.switch_choice,n.Mod,e.switch_Mod, 0, e.switch_sym);%
            
            % TODO : use this stats in a figure
            for i_frac=1:length(v_frac)
                S_=zeros(size(S)); %
                for i_batch=1:e.batch_size,
                    max_S=max(abs(S(:,i_batch)));
                    ind=find(abs(S(:,i_batch)) > v_frac(i_frac)*max_S); % the basis functions we use
                    S_(ind,i_batch)=S(ind,i_batch);
                end
                E=X-n.A*S_;
                residual(i_frac)=sum(E(:).^2)/e.L/e.batch_size;
                L0(i_frac)=sum(S_(:)~=0)/e.M/e.batch_size;
            end
            save('-v7',nom_exp_,'n','s','residual','L0')
        end
    end
end;

if ~(switch_lock(e,nom_exp)==-1),
    % while running, things may have changed on the cluster: check if one simulation is still running
    switch_lock(e,nom_exp, 0) % unlock by default
    e.Method='ssc';
    
    for i_frac=1:length(v_frac),
        nom_exp_=[e.where '/stability_ssc_frac_' num2str(i_frac) '.mat'];
        if ~(switch_lock(e,nom_exp_)==-1),
            switch_lock(e,nom_exp, 1) % lock if one simulation is missing or running
        end
    end
    if ~switch_lock(e,nom_exp)
        add_log(e,'Made learning with experiment_stability_ssc');
    end
end

if ~switch_lock(e,nom_exp),
    switch_lock(e,nom_exp,1) %lock
    
    add_log(e,'Make coding with experiment_stability_ssc');
    
    for i_frac=1:length(v_frac),
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Sparsenet / MP_Sparsenet %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        e=default(e.where);
        e.batch_size=100*e.batch_size; % a big batch
        X=get_patch(e);
        
        e.Method='ssc';
        
        nom_exp_=[e.where '/stability_ssc_frac_' num2str(i_frac) '.mat'];
        load(nom_exp_)
        if ~exist([e.where '/fig_stability_ssc_frac_' num2str(i_frac) '.png'],'file'),
            imwrite((tile(n.A)+1)/2,[e.where '/fig_stability_ssc_frac_' num2str(i_frac) '.png'])
        end
        
        S=mp_fitS(n.A,X,e.noise_var_ssc,e.frac,0,n.Mod,0);%
        E=X-n.A*S;
        
        residual(i_frac)=sum(E(:).^2)/e.L/e.batch_size;%
        sparseness_end(i_frac)=1/e.sigma*sum(log(1+S(:).^2/e.sigma^2))/e.M/e.batch_size;
        L0_end(i_frac)=sum(S(:)~=0)/e.M/e.batch_size;%
        
    end
    
    save('-v7',nom_exp,'residual','sparseness_end','L0_end','v_frac') %
    add_log(e,'Made coding with experiment_stability_ssc');
    
end;

if switch_lock(e,nom_exp)==-1,
    load(nom_exp)
    
    v_frac_ =v_frac;
    
    
    for i_frac=1:length(v_frac),
        nom_exp_=[e.where '/stability_ssc_frac_' num2str(i_frac) '.mat'];
        load(nom_exp_)
        if ~exist([e.where '/fig_stability_ssc_frac_' num2str(i_frac) '.png'],'file'),
            imwrite((tile(n.A)+1)/2,[e.where '/fig_stability_ssc_frac_' num2str(i_frac) '.png'])
        end
    end
    
    %%%%% PLOTS
    % displays end of learning perfromance for different learnig rates
    % information transmission
    
    if ~exist([e.where '/fig_stability_ssc_frac_energy.eps'],'file'),
        figure(3),set(gcf, 'visible', 'off'),
        semilogx(v_frac, residual'),
        axis tight,%title('Learning : energy of residual'),
        xlabel('L0 threshold'),ylabel('Mean residual Energy'),
        fig2pdf([e.where '/fig_stability_ssc_frac_energy'],10,10)
        
    end
    
    if ~exist([e.where '/fig_stability_ssc_frac_L0.eps'],'file'),
        figure(3),set(gcf, 'visible', 'off'),
        semilogx(v_frac, L0_end'),
        axis tight,%title('Learning : energy of residual'),
        xlabel('L0 threshold'),ylabel('Mean residual L0'),
        fig2pdf([e.where '/fig_stability_ssc_frac_L0'],10,10)
        
    end

    if ~exist([e.where '/fig_stability_ssc_frac_sparseness.eps'],'file'),
        % Sparseness
        figure(4),set(gcf, 'visible', 'off'),
        semilogx(v_frac,sparseness_end),
        axis tight,%title('Learning: sparseness of representation'),
        xlabel('L0 threshold'),ylabel('Olshausen''s sparseness'),
        fig2pdf([e.where '/fig_stability_ssc_frac_sparseness'],10,10)
        
    end
    if ~exist([e.where '/fig_stability_ssc_frac_ol_cost.eps'],'file'),
        figure(5),set(gcf, 'visible', 'off'),
        semilogx(v_frac,(1/e.noise_var_cgf)*residual + sparseness_end*e.M/e.L),
        axis tight,%title('Learning: sparseness of representation'),*e.L
        xlabel('L0 threshold'),ylabel('Olshausen''s cost'),
        fig2pdf([e.where '/fig_stability_ssc_frac_ol_cost'],10,10)
        
    end
end



nom_exp=[e.where '/stability_ssc_noise.mat'];
v_noise= logspace(-3,-1,9);% logspace(-.01,-.4,9); % 10.^(-3:0.25:-1);%
if ~(switch_lock(e,nom_exp)==-1),
    
    add_log(e,'Make learning with experiment_stability_ssc_noise');
    for i_noise=1:length(v_noise),
        % sets a similar initialization for weights
        A_rand = rand(e.L,e.M)-0.5; A_rand = A_rand*diag(1./sqrt(sum(A_rand.*A_rand)));
        gain_rand=sqrt(sum(A_rand.*A_rand))';
        %%%%%%%%
        % aSSC %
        %%%%%%%%
        e=default(e.where);
        e.Method='ssc';
        n.Pz_j=1/e.n_quant*ones(e.n_quant,e.M); n.Mod=cumsum(n.Pz_j);
        e.noise_var_ssc = v_noise(i_noise);
        
        nom_exp_=[e.where '/stability_ssc_noise_' num2str(i_noise) '.mat'];
        if ~switch_lock(e,nom_exp_),
            switch_lock(e,nom_exp_,1) %lock
            add_log(e,['experiment_stability_ssc-i_noise= ' num2str(i_noise) '/' num2str(length(v_noise))]);
            n.A = A_rand;
            [n,s]=sparsenet(n,e);
            save('-v7',nom_exp_,'n','s')
        end
    end
end;

if ~(switch_lock(e,nom_exp)==-1),
    % while running, things may have changed on the cluster: check if one simulation is still running
    switch_lock(e,nom_exp, 0) % unlock by default
    e.Method='ssc';
    
    for i_noise=1:length(v_noise),
        nom_exp_=[e.where '/stability_ssc_noise_' num2str(i_noise) '.mat'];
        if ~(switch_lock(e,nom_exp_)==-1),
            switch_lock(e,nom_exp, 1) % lock if one simulation is missing or running
        end
    end
    if ~switch_lock(e,nom_exp)
        add_log(e,'Made learning with experiment_stability_ssc_noise');
    end
end

if ~switch_lock(e,nom_exp),
    switch_lock(e,nom_exp,1) %lock
    
    add_log(e,'Make coding with experiment_stability_ssc_noise');
    
    for i_noise=1:length(v_noise),
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Sparsenet / MP_Sparsenet %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        e=default(e.where);
        e.batch_size=100*e.batch_size; % a big batch
        X=get_patch(e);
        
        e.Method='ssc';
        
        nom_exp_=[e.where '/stability_ssc_noise_' num2str(i_noise) '.mat'];
        load(nom_exp_)
        if ~exist([e.where '/fig_stability_ssc_noise_' num2str(i_noise) '.png'],'file'),
            imwrite((tile(n.A)+1)/2,[e.where '/fig_stability_ssc_noise_' num2str(i_noise) '.png'])
        end
        
        S=mp_fitS(n.A,X,e.noise_var_ssc,e.frac,0,n.Mod,0);%
        E=X-n.A*S;
        
        residual(i_noise)=sum(E(:).^2)/e.L/e.batch_size;%
        sparseness_end(i_noise)=1/e.sigma*sum(log(1+S(:).^2/e.sigma^2))/e.M/e.batch_size;
        L0_end(i_noise)=sum(S(:)~=0)/e.M/e.batch_size;%
        
    end
    
    save('-v7',nom_exp,'residual','sparseness_end','L0_end','v_noise') %
    add_log(e,'Made coding with experiment_stability_ssc_noise');
    
end;

if switch_lock(e,nom_exp)==-1,
    load(nom_exp)
    
    v_noise_ =v_noise;
    
    for i_noise=1:length(v_noise),
        nom_exp_=[e.where '/stability_ssc_noise_' num2str(i_noise) '.mat'];
        load(nom_exp_)
        if ~exist([e.where '/fig_stability_ssc_noise_' num2str(i_noise) '.png'],'file'),
            imwrite((tile(n.A)+1)/2,[e.where '/fig_stability_ssc_noise_' num2str(i_noise) '.png'])
        end
    end
    
    %%%%% PLOTS
    % displays end of learning perfromance for different learnig rates
    % information transmission
    
    if ~exist([e.where '/fig_stability_ssc_noise_energy.eps'],'file'),
        figure(3),set(gcf, 'visible', 'off'),
        semilogx(v_noise, residual'),
        axis tight,%title('Learning : energy of residual'),
        xlabel('noise threshold'),ylabel('Mean residual Energy'),
        fig2pdf([e.where '/fig_stability_ssc_noise_energy'],10,10)
        
    end
    if ~exist([e.where '/fig_stability_ssc_noise_L0.eps'],'file'),
        figure(3),set(gcf, 'visible', 'off'),
        semilogx(v_noise, L0_end'),
        axis tight,%title('Learning : energy of residual'),
        xlabel('noise threshold'),ylabel('Mean residual L0'),
        fig2pdf([e.where '/fig_stability_ssc_noise_L0'],10,10)
        
    end
    if ~exist([e.where '/fig_stability_ssc_noise_sparseness.eps'],'file'),
        % Sparseness
        figure(4),set(gcf, 'visible', 'off'),
        semilogx(v_noise,sparseness_end),
        axis tight,%title('Learning: sparseness of representation'),
        xlabel('noise threshold'),ylabel('Olshausen''s sparseness'),
        fig2pdf([e.where '/fig_stability_ssc_noise_sparseness'],10,10)
        
    end
    if ~exist([e.where '/fig_stability_ssc_noise_ol_cost.eps'],'file'),
        figure(5),set(gcf, 'visible', 'off'),
        semilogx(v_noise,(1/e.noise_var_cgf)*residual + sparseness_end*e.M/e.L),
        axis tight,%title('Learning: sparseness of representation'),*e.L
        xlabel('noise threshold'),ylabel('Olshausen''s cost'),
        fig2pdf([e.where '/fig_stability_ssc_noise_ol_cost'],10,10)
        
    end
end
