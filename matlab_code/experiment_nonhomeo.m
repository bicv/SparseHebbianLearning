function experiment_nonhomeo(e)
% experiment_nonhomeo.m : compares coding efficiency with or without the
% homeostatic constraint in Matching Pursuit

% generates a set of filters without the homeostatic rule and then compare
% both coding schemes.
% this is limited to MP, since CG *requires* homeo to work, but we cand do
% it with CG filters and COMP sparse coding

% depends on the learning done in experiment_learn.m

%## Author : Laurent Perrinet <Laurent.Perrinet@incm.cnrs-mrs.fr>
%## This software is distributed under the terms of the GPL


nom_exp=[e.where '/MP_nonhomeo.mat'];
nom_exp_learn =[e.where '/MP_nonhomeo_learn.mat'];
nom_exp_lut =[e.where '/MP_nonhomeo_lut.mat'];


if ~switch_lock(e,nom_exp_learn),
    switch_lock(e,nom_exp_learn,1) %lock
    add_log(e,'Make learning with experiment_nonhomeo');
    e=default(e.where);
    e.Method='amp';
    A_rand = rand(e.L,e.M)-0.5; n.A = A_rand*diag(1./sqrt(sum(A_rand.*A_rand)));
    n.Pz_j=1/e.n_quant*ones(e.n_quant,e.M); n.Mod=cumsum(n.Pz_j);
    e.video=1; % record during AMP
    e.var_eta_ssc=0; % no homeostasis learning
    e.switch_choice=0; % and no homeostasis (not necessary in theory since Pz_j are equal, but faster in practice)
    [n,s]=sparsenet(n,e);
    
    unix(['rm -f ' e.where '/fig_nonhomeo*'])
    unix(['rm -f ' nom_exp_lut])
    save('-v7',nom_exp_learn,'n','s')
    add_log(e,'Made learning with experiment_nonhomeo');
end

if ~switch_lock(e,nom_exp) && ~switch_lock(e,nom_exp_lut) &&  switch_lock(e,[e.where '/MP.mat'])==-1 && ...
        switch_lock(e,[e.where '/CG.mat'])==-1 && switch_lock(e,nom_exp_learn)==-1,
    switch_lock(e,nom_exp_lut,1)   %lock
    add_log(e,'Make lut with experiment_nonhomeo');
    e=default(e.where);
    % during learning, we did not record the modulation function, so we
    % make a small trip to compute initial probabilities for later use:
    e.eta_ssc=0;
    e.eta_cgf=0;
    e.switch_Mod = 0; % switched off anyway in sparsenet.m
    e.switch_choice = 1;
    e.num_trials=ceil(e.num_trials/20);%100)%DEBUG
    e.Method='ssc';
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  ROC with AMP filters   %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    load(nom_exp_learn)
    n.Pz_j=1/e.n_quant*ones(e.n_quant,e.M); n.Mod=cumsum(n.Pz_j);
    [n_amp,s]=sparsenet(n,e);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  ROC with SSC filters   %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    load([e.where '/MP.mat']),%
    n.Pz_j=1/e.n_quant*ones(e.n_quant,e.M); n.Mod=cumsum(n.Pz_j);
    [n_ssc,s]=sparsenet(n,e);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  ROC with CGF filters   %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    load([e.where '/CG.mat']),%
    n.Pz_j=1/e.n_quant*ones(e.n_quant,e.M); n.Mod=cumsum(n.Pz_j);
    [n_cg,s]=sparsenet(n,e);
    
    unix(['rm -f ' e.where '/fig_nonhomeo*'])
    unix(['rm -f ' nom_exp])
    save('-v7',nom_exp_lut,'n_cg','n_ssc','n_amp')
    add_log(e,'Made lut with experiment_nonhomeo');
end

%    unix(['rm -f ' e.where '/fig_nonhomeo_lut*'])
if ~switch_lock(e,nom_exp) && switch_lock(e,nom_exp_lut)==-1,
    
    % MP - with different homeo
    switch_lock(e,nom_exp,1) %lock
    add_log(e,'Make coding with experiment_nonhomeo');
    e=default(e.where);
    load(nom_exp_lut)
    e.batch_size=100*e.batch_size; % a big batch%DEBUG
    X=get_patch(e,1); % DEBUG use norm?
    res_total=sum(X(:).^2)/e.L/e.batch_size;
    
    % test out the efficiency of the non homeostatic method
    
    %%%%%%%%%%%%%
    %   AMP    %
    %%%%%%%%%%%%%
    e.switch_Mod = 0; % no quantization
    e.switch_choice = 0; % and no homeostasis
    S=mp_fitS(n_amp.A,X,0,1,e.switch_choice,n_ssc.Mod,e.switch_Mod,0,e.switch_sym);

    for i_L0=1:e.M,
        S_=zeros(size(S));
        for i_batch=1:e.batch_size,
            [S_sorted, S_ind] = sort(abs(S(:,i_batch)), 'descend');
            S_(S_ind(1:i_L0),i_batch)=S(S_ind(1:i_L0),i_batch);
        end;
        E=X-n_amp.A*S_;
        res_amp(i_L0)=sum(E(:).^2)/e.L/e.batch_size;
        L0_amp(i_L0)=i_L0/e.M;%
    end
    
    e.switch_Mod = 1 ; % quantization
    e.switch_choice = 0; % and homeostasis
    S=mp_fitS(n_amp.A,X,0,1,e.switch_choice,n_amp.Mod,e.switch_Mod,mean(sort(abs(S),1,'descend'),2),e.switch_sym);%
    S_ = abs(S); % sparse correlation values (with normA = 1 )
    lut = mean(sort(S_,1,'descend'),2);
    
    for i_L0=1:e.M,
        S_=zeros(size(S));
        for i_batch=1:e.batch_size,
            [S_sorted, S_ind] = sort(abs(S(:,i_batch)), 'descend');
            S_(S_ind(1:i_L0),i_batch)=sign(S(S_ind(1:i_L0),i_batch)).*lut(1:i_L0);%
        end
        E=X-n_amp.A*S_;
        res_amp_quant(i_L0)=sum(E(:).^2)/e.L/e.batch_size;
        L0_amp_quant(i_L0)=i_L0/e.M;%
    end
    %%%%%%%%%%%%%
    %    aSSC    %
    %%%%%%%%%%%%%
    e.switch_Mod = 0; % no quantization
    e.switch_choice = 0; % and no homeostasis
    S=mp_fitS(n_ssc.A,X,0,1,e.switch_choice,n_ssc.Mod,e.switch_Mod,0,e.switch_sym);
    
    for i_L0=1:e.M,
        S_=zeros(size(S));
        for i_batch=1:e.batch_size,
            [S_sorted, S_ind] = sort(abs(S(:,i_batch)), 'descend');
            S_(S_ind(1:i_L0),i_batch)=S(S_ind(1:i_L0),i_batch);
        end
        E=X-n_ssc.A*S_;
        res_ssc(i_L0)=sum(E(:).^2)/e.L/e.batch_size;
        L0_ssc(i_L0)=i_L0/e.M;
    end
    
    %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    e.switch_Mod = 1 ; % quantization
    e.switch_choice = 1; % and homeostasis (this is not important normally since at the end of the learning f_i curves are similar)
    S=mp_fitS(n_ssc.A,X,0,1,e.switch_choice,n_ssc.Mod,e.switch_Mod,mean(sort(abs(S),1,'descend'),2),e.switch_sym);%
    S_ = abs(S); % sparse correlation values (with normA = 1 )
    lut = mean(sort(S_,1,'descend'),2);
    
    S=mp_fitS(n_ssc.A,X,0,1,e.switch_choice,n_ssc.Mod,e.switch_Mod,lut,e.switch_sym);%
    for i_L0=1:e.M,
        S_=zeros(size(S));
        for i_batch=1:e.batch_size,
            [S_sorted, S_ind] = sort(abs(S(:,i_batch)), 'descend');
            S_(S_ind(1:i_L0),i_batch)=sign(S(S_ind(1:i_L0),i_batch)).*lut(1:i_L0);%
        end
        E=X-n_ssc.A*S_;
        res_ssc_quant(i_L0)=sum(E(:).^2)/e.L/e.batch_size;
        L0_ssc_quant(i_L0)=i_L0/e.M;%
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  ROC with CGF filters   %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    %   e.switch_Mod = 0;
    %   e.switch_choice = 0;
    S = mp_fitS(n_cg.A,X);
    
    for i_L0=1:e.M,
        S_=zeros(size(S));
        for i_batch=1:e.batch_size,
            [S_sorted, S_ind] = sort(abs(S(:,i_batch)), 'descend');
            S_(S_ind(1:i_L0),i_batch)=S(S_ind(1:i_L0),i_batch);
        end
        E=X-n_cg.A*S_;
        res_cg(i_L0)=sum(E(:).^2)/e.L/e.batch_size;
        L0_cg(i_L0)=i_L0/e.M;%
    end
    
    e.switch_Mod = 1 ; % quantization
    e.switch_choice= 0; % and homeostasis (this is not important normally since at the end of the learning f_i curves are similar)
    S=mp_fitS(n_cg.A,X,0,1,e.switch_choice,n_cg.Mod,e.switch_Mod,mean(sort(abs(S),1,'descend'),2),e.switch_sym);%
    S_ = abs(S); % sparse correlation values (with normA = 1 )
    lut = mean(sort(S_,1,'descend'),2);
    
    S=mp_fitS(n_cg.A,X,0,1,e.switch_choice,n_cg.Mod,e.switch_Mod, lut, e.switch_sym);%
    for i_L0=1:e.M,
        S_=zeros(size(S));
        for i_batch=1:e.batch_size,
            [S_sorted, S_ind] = sort(abs(S(:,i_batch)), 'descend');
            S_(S_ind(1:i_L0),i_batch)=sign(S(S_ind(1:i_L0),i_batch)).*lut(1:i_L0);
        end
        E=X-n_cg.A*S_;
        res_cg_quant(i_L0)=sum(E(:).^2)/e.L/e.batch_size;
        L0_cg_quant(i_L0)=i_L0/e.M;
    end
    
    unix(['rm -f ' e.where '/fig_nonhomeo_L0* ' e.where '/fig_nonhomeo_bits* ' e.where '/fig_nonhomeo_quant*'])
    save('-v7',nom_exp,'res_total','L0_amp','res_amp','L0_amp_quant','res_amp_quant',...
        'L0_ssc','res_ssc','L0_ssc_quant','res_ssc_quant',...
        'L0_cg','res_cg','L0_cg_quant','res_cg_quant');
    add_log(e,'Made coding with experiment_nonhomeo');
    
end;

%    unix(['rm -f ' e.where '/fig_nonhomeo_bits* '])

if switch_lock(e,nom_exp)==-1 && switch_lock(e,nom_exp_lut)==-1,
    
    e=default(e.where);
    
    
    if ~exist([e.where '/fig_map_ssc_nonhomeo.png'],'file'),
        load(nom_exp_learn);
        add_log(e,'Generate image of the basis functions');
        imwrite((tile(n.A)+1)/2,[e.where '/fig_map_ssc_nonhomeo.png'])
    end
    
    color_amp = 'r';%'k';%
    color_ssc = 'r-.';%'k--';%
    color_cg = 'b--';%'k-.';%
    
    
    if ~exist([e.where '/fig_nonhomeo_lut.eps'],'file'),
        load(nom_exp_lut)
        sub = 10;
        z = 0:sub/e.n_quant:1-1/e.n_quant ;
        figure(2),set(gcf, 'visible', 'off'),
        subplot(131), plot(z,n_amp.Mod(1:sub:e.n_quant,:))
        axis tight,xlabel('coeff'),ylabel('z'),
        title('AMP ')
        subplot(132), plot(z,n_cg.Mod(1:sub:e.n_quant,:))%
        axis tight,xlabel('coeff'),%
        title('SparseNet')
        subplot(133), plot(z,n_ssc.Mod(1:sub:e.n_quant,:))%
        axis tight,xlabel('coeff'),%
        title('aSSC')
        fig2pdf([e.where '/fig_nonhomeo_lut'],5,13)
    end
    
    load(nom_exp)
    if ~exist([e.where '/fig_nonhomeo_L0.eps'],'file'),
        figure(8),set(gcf, 'visible', 'off'),
        bps = log2(e.M);
        plot(L0_amp, res_amp/res_total,color_amp,...
            L0_ssc, res_ssc/res_total,color_ssc,...
            L0_cg, res_cg/res_total,color_cg,'LineWidth',1.5)%
        hold on, plot(0,0,''), plot(0,1,''), hold off,  legend('aSSC','AMP','SN')%
        axis tight,
        grid on, %
        xlabel('Code Length (L0 norm)'),
        ylabel('Residual Energy (L2 norm)'),
        fig2pdf([e.where '/fig_nonhomeo_L0'],5,5)
    end
    if ~exist([e.where '/fig_nonhomeo_quant.eps'],'file'),
        figure(8),set(gcf, 'visible', 'off'),
        bps = log2(e.M);
        plot(L0_amp, (res_amp_quant-res_amp)/res_total,color_amp,...
            L0_ssc, (res_ssc_quant-res_ssc)/res_total,color_ssc,...
            L0_cg, (res_cg_quant-res_cg)/res_total,color_cg,'LineWidth',1.5)%
        hold on, plot(0,0,''),hold off,  legend('aSSC','AMP','SN', 'Location','SouthEast')%
        axis tight,
        grid on, %
        xlabel('Code Length (L0 norm)'),
        ylabel('Residual Energy (L2 norm)'),
        fig2pdf([e.where '/fig_nonhomeo_quant'],10,10)
    end
    if ~exist([e.where '/fig_nonhomeo_bits.eps'],'file'),
        figure(9),set(gcf, 'visible', 'off'),
        bps = log2(e.M)/e.L;
        plot(L0_amp_quant, res_amp_quant/res_total,color_amp, ...
            L0_ssc_quant, res_ssc_quant/res_total,color_ssc,...
            L0_cg_quant, res_cg_quant/res_total,color_cg,'LineWidth',1.5)%
        hold on, plot(0,0,''),plot(0,1,''), hold off,
        legend('aSSC','AMP','SN', 'Location','NorthEast')%
        axis tight,
        grid on, %
        xlabel('Code Length (L0 norm)'),
        ylabel('Residual Energy (L2 norm)'),
        fig2pdf([e.where '/fig_nonhomeo_bits'],10,10)
    end
    
end
