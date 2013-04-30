function experiment_code_efficiency(e)
% experiment_code_efficiency.m : compares coding after learning with both methods
% -------------------------------------------------------------------------------
% depends on the learning done in experiment_learn.m

%## Author : Laurent Perrinet <Laurent.Perrinet@incm.cnrs-mrs.fr>
%## This software is distributed under the terms of the GPL

% compare coding with the final basis functions
e=default(e.where); %loads default parameters

% gets a batch of imagelets
e.batch_size=10*e.batch_size; % a big batch
% in this experiment, we modify the number of active coefficients and
% compute the SNR with both methods
nom_exp=[e.where '/L0_stats.mat'];
%unix(['rm -f ' nom_exp])
if switch_lock(e,[e.where '/CG.mat'])==-1 && switch_lock(e,[e.where '/MP.mat'])==-1 && ...
        ~switch_lock(e,nom_exp),
    
    switch_lock(e,nom_exp,1) %lock
    add_log(e,'Make learning with experiment_code_efficiency');
    clear res L0
    X=get_patch(e);
    res.total=sum(X(:).^2)/e.L/e.batch_size;
    v_frac = 0:0.01:0.99;%
    
    %%%%%%%%%%%%%
    %    MP     %
    %%%%%%%%%%%%%
    load([e.where '/MP.mat']),%
    e.switch_Mod = 0; % we compare here the *coding* methods, not the full algorithm (CG has no spike decoding scheme)
    e.switch_choice = 0; % we compare here the *coding* methods, not the full algorithm (CG has no spike decoding scheme)
    S=mp_fitS(n.A,X,0,1,e.switch_choice,n.Mod,e.switch_Mod, 0, e.switch_sym);%
    
    for i_frac=1:length(v_frac)
        S_=zeros(size(S)); %
        for i_batch=1:e.batch_size,
            max_S=max(abs(S(:,i_batch)));
            ind=find(abs(S(:,i_batch)) > v_frac(i_frac)*max_S); % the basis functions we use
            S_(ind,i_batch)=S(ind,i_batch);
        end
        E=X-n.A*S_;
        res.mp(i_frac)=sum(E(:).^2)/e.L/e.batch_size;
        variance.mp(i_frac)=var(E(:).^2);
        L0.mp(i_frac)=sum(S_(:)~=0)/e.M/e.batch_size;
        sparseness.mp(i_frac)=e.beta/e.sigma*sum(log(1+S_(:).^2/e.sigma^2))/e.M/e.batch_size;
        kurt.mp(i_frac)=(sum((S_(:)-( sum(S_(:))/e.M/e.batch_size )).^4)/e.M/e.batch_size)/...
            (sum((S_(:)-( sum(S_(:))/e.M/e.batch_size )).^2)/e.M/e.batch_size)^2 -3; % Kurtosis
        kurt_e.mp(i_frac)=(sum((S_(:)-( sum(S_(:))/e.M/e.batch_size )).^4)/e.M/e.batch_size)/...
            (sum((S_(:)-( sum(S_(:))/e.M/e.batch_size )).^2)/e.M/e.batch_size)^2 -3; % Kurtosis
    end
    % using OOMP with MP filters
    if 1,
        S = perform_omp(n.A,X);
        
        for i_frac=1:length(v_frac)
            S_=zeros(size(S)); %
            for i_batch=1:e.batch_size,
                max_S=max(abs(S(:,i_batch)));
                ind=find(abs(S(:,i_batch)) > v_frac(i_frac)*max_S); % the basis functions we use
                S_(ind,i_batch)=S(ind,i_batch);
            end
            E=X-n.A*S_;
            res.mp_oomp(i_frac)=sum(E(:).^2)/e.L/e.batch_size;
            L0.mp_oomp(i_frac)=sum(S_(:)~=0)/e.M/e.batch_size;
            variance.mp_oomp(i_frac)=var(E(:).^2);
            sparseness.mp_oomp(i_frac)=e.beta/e.sigma*sum(log(1+S_(:).^2/e.sigma^2))/e.M/e.batch_size;
        end
    end
    % cg with mp filters
    S=cgf_fitS(n.A,X,e.noise_var_cgf,e.beta,e.sigma,e.tol);
    
    for i_frac=1:length(v_frac),
        S_=zeros(size(S)); S__=S_; %
        for i_batch=1:e.batch_size,
            max_S=max(abs(S(:,i_batch)));
            % second sparsification using cgf to be fair compared to MP
            ind=find(abs(S(:,i_batch))>v_frac(i_frac)*max_S); % the basis functions we use
            S__(ind,i_batch)=S(ind,i_batch);
            S_(ind,i_batch)=cgf_fitS(n.A(:,ind),X(:,i_batch),e.noise_var_cgf,e.beta,e.sigma,e.tol);
        end
        E=X-n.A*S_;
        res.cg_with_mp(i_frac)=sum(E(:).^2)/e.L/e.batch_size;
        variance.cg_with_mp(i_frac)=var(E(:).^2);
        L0.cg_with_mp(i_frac)=sum(S_(:)~=0)/e.M/e.batch_size;
    end
    
    %%%%%%%%%%%%%
    %    CGF    %
    %%%%%%%%%%%%%
    load([e.where '/CG.mat']),
    S=cgf_fitS(n.A,X,e.noise_var_cgf,e.beta,e.sigma,e.tol);
    
    
    for i_frac=1:length(v_frac),
        S_=zeros(size(S)); S__=S_; %
        for i_batch=1:e.batch_size,
            max_S=max(abs(S(:,i_batch)));
            % second sparsification using cgf to be fair compared to MP
            ind=find(abs(S(:,i_batch))>v_frac(i_frac)*max_S); % the basis functions we use
            S__(ind,i_batch)=S(ind,i_batch);
            S_(ind,i_batch)=cgf_fitS(n.A(:,ind),X(:,i_batch),e.noise_var_cgf,e.beta,e.sigma,e.tol);
        end
        E=X-n.A*S_;
        res.cg(i_frac)=sum(E(:).^2)/e.L/e.batch_size;
        variance.cg(i_frac)=var(E(:).^2);
        L0.cg(i_frac)=sum(S_(:)~=0)/e.M/e.batch_size;
        sparseness.cg(i_frac)=e.beta/e.sigma*sum(log(1+S__(:).^2/e.sigma^2))/e.M/e.batch_size;
        kurt.cg(i_frac)=(sum((S_(:)-( sum(S_(:))/e.M/e.batch_size )).^4)/e.M/e.batch_size)/...
            (sum((S_(:)-( sum(S_(:))/e.M/e.batch_size )).^2)/e.M/e.batch_size)^2 -3; % Kurtosis
    end
    
    % using OOMP with CGF filters
    S = perform_omp(n.A,X);
    
    for i_frac=1:length(v_frac)
        S_=zeros(size(S)); %
        for i_batch=1:e.batch_size,
            max_S=max(abs(S(:,i_batch)));
            ind=find(abs(S(:,i_batch)) > v_frac(i_frac)*max_S); % the basis functions we use
            S_(ind,i_batch)=S(ind,i_batch);
        end
        E=X-n.A*S_;
        res.cg_oomp(i_frac)=sum(E(:).^2)/e.L/e.batch_size;
        variance.cg_oomp(i_frac)=var(E(:).^2);
        L0.cg_oomp(i_frac)=sum(S_(:)~=0)/e.M/e.batch_size;
    end
    
    % using MP with CGF filters
    n.Pz_j=1/e.n_quant*ones(e.n_quant,e.M); n.Mod=cumsum(n.Pz_j);
    S=mp_fitS(n.A,X,0,1,e.switch_choice,n.Mod,e.switch_Mod, 0,e.switch_sym);%
    
    for i_frac=1:length(v_frac)
        S_=zeros(size(S)); S__=S_; %
        for i_batch=1:e.batch_size,
            max_S=max(abs(S(:,i_batch)));
            % second sparsification using cgf to be fair compared to MP
            ind=find(abs(S(:,i_batch))>v_frac(i_frac)*max_S); % the basis functions we use
            S__(ind,i_batch)=S(ind,i_batch);
            S_(ind,i_batch)=cgf_fitS(n.A(:,ind),X(:,i_batch),e.noise_var_cgf,e.beta,e.sigma,e.tol);
        end
        E=X-n.A*S_;
        res.mp_with_cg(i_frac)=sum(E(:).^2)/e.L/e.batch_size;
        variance.mp_with_cg(i_frac)=var(E(:).^2);
        L0.mp_with_cg(i_frac)=sum(S_(:)~=0)/e.M/e.batch_size;
    end
    
    
    unix(['rm -f ' e.where '/fig_efficiency_*'])
    save('-v7',nom_exp,'res','v_frac','L0','kurt','sparseness','variance');
    add_log(e,'Made learning with experiment_code_efficiency');
end;

% unix(['rm -f ' e.where '/fig_efficiency*'])

if switch_lock(e,nom_exp)==-1,
    
    load(nom_exp),
    
    color_cg = 'b--';%'k--';%
    color_mp = 'r';%'k';%
    color_cg_oomp = 'go:';%'k';%
    color_mp_oomp = 'gs:';%'k';%
    
    if ~exist([e.where '/fig_efficiency_L0.eps'],'file'),
        figure(7),set(gcf, 'visible', 'off'),
        plot([  L0.cg],[  res.cg/res.total],color_cg,...
            [  L0.mp],[  res.mp/res.total],color_mp,'LineWidth',1.5)%
        hold on,
        errorbar(L0.mp(1:8:end),res.mp(1:8:end)/res.total,sqrt(variance.mp(1:8:end)),'r','LineStyle','none')%
        errorbar(L0.cg(1:8:end),res.cg(1:8:end)/res.total,sqrt(variance.cg(1:8:end)),'b','LineStyle','none')%
        plot(0,0,''),  plot(0,1,''), hold off
        axis([0, .3, 0, 1])% 
        grid on, %
        xlabel('Sparseness (L0-norm)'),ylabel('Residual Energy (L2 norm)'),
        fig2pdf([e.where '/fig_efficiency_L0'],10,10)
    end
    
    if ~exist([e.where '/fig_efficiency_L0_omp.eps'],'file'),
        figure(10),set(gcf, 'visible', 'off'),
        plot(L0.cg,res.cg/res.total,color_cg, ...
            L0.mp,res.mp/res.total,color_mp, ...
            L0.cg_oomp(1:8:end),res.cg_oomp(1:8:end)/res.total,color_cg_oomp, ...
            L0.mp_oomp(1:8:end),res.mp_oomp(1:8:end)/res.total,color_mp_oomp,'LineWidth',1.5)%
        hold on,
        errorbar(L0.mp(1:8:end),res.mp(1:8:end)/res.total,sqrt(variance.mp(1:8:end)),'r','LineStyle','none')%
        errorbar(L0.cg(1:8:end),res.cg(1:8:end)/res.total,sqrt(variance.cg(1:8:end)),'b','LineStyle','none')%
        errorbar(L0.cg_oomp(1:8:end),res.cg_oomp(1:8:end)/res.total,sqrt(variance.cg_oomp(1:8:end)),'g','LineStyle','none')%
        errorbar(L0.mp_oomp(1:8:end),res.mp_oomp(1:8:end)/res.total,sqrt(variance.mp_oomp(1:8:end)),'g','LineStyle','none')%
        plot(0,0,''), plot(0,1,''),  hold off
        legend('SparseNet (SN)','aSSC', 'SN with OOMP','aSSC with OOMP'),
        axis([0, .3, 0, 1])% 
        grid on,
        xlabel('Sparseness (L0-norm)'),ylabel('Residual Energy (L2 norm)'),
        fig2pdf([e.where '/fig_efficiency_L0_omp'],10,10)
    end
    
    if ~exist([e.where '/fig_efficiency_L0_crossed.eps'],'file'),
        figure(9),set(gcf, 'visible', 'off'),
        plot(L0.cg,res.cg/res.total,color_cg, ...
            L0.mp,res.mp/res.total,color_mp, ...
            L0.mp_with_cg,res.mp_with_cg/res.total,'g', ...
            L0.cg_with_mp,res.cg_with_mp/res.total,'g--','LineWidth',1.5)%
        hold on,
        errorbar(L0.mp(1:8:end),res.mp(1:8:end)/res.total,sqrt(variance.mp(1:8:end)),'r','LineStyle','none')%
        errorbar(L0.cg(1:8:end),res.cg(1:8:end)/res.total,sqrt(variance.cg(1:8:end)),'b','LineStyle','none')%
        plot(0,0,''),  plot(0,1,''),  hold off
        legend('SparseNet (SN)','aSSC','aSSC with CG', 'SN with SSC'),
        axis([0, .3, 0, 1])%
        grid on, %
        xlabel('Sparseness (L0-norm)'),ylabel('Residual Energy (L2 norm)'),
        fig2pdf([e.where '/fig_efficiency_L0_crossed'],10,10)
    end
    
    if ~exist([e.where '/fig_efficiency_L1.eps'],'file'),
        
        figure(8),set(gcf, 'visible', 'off'),
        plot(sparseness.cg,res.cg/res.total,color_cg,...
            sparseness.mp,res.mp/res.total,color_mp,'LineWidth',1.5)%,
        hold on, plot(0,0,''),  plot(0,1,''), hold off
        set(gca,'XTick',[])
        grid on, %
        xlabel('Olshausen''s sparseness'), ylabel('Res. Energy (L2 norm)'),
        legend('SparseNet','aSSC','Location','NorthEast'),hold off
        fig2pdf([e.where '/fig_efficiency_L1'],5,5)
    end
    
    if ~exist([e.where '/fig_efficiency_kurt.eps'],'file'),
        figure(9),set(gcf, 'visible', 'off'), %
        plot(L0.cg,kurt.cg,color_cg,L0.mp,kurt.mp,color_mp,'LineWidth',1.5)
        hold on, plot(0,0,''), hold off
        legend('SparseNet','aSSC'), 
        axis tight,  grid on,
        xlabel('Sparseness (L0 norm)'),ylabel('Kurtosis'),
        fig2pdf([e.where '/fig_efficiency_kurt'],10,10)
    end
    
end
