function experiment_code_histogram(e)

% experiment_code_histogram : compares the efficiency of the sparsification
% at the end of the learning for different learning parameters
% as no influence on MP, so we do it only on CGF

%## Author : Laurent Perrinet <Laurent.Perrinet@incm.cnrs-mrs.fr>
%## This software is distributed under the terms of the GPL

nom_exp=[e.where '/hist.mat'];
%unix(['rm -f ' nom_exp])
if switch_lock(e,[e.where '/CG.mat'])==-1 && switch_lock(e,[e.where '/MP.mat'])==-1 && ...
        ~switch_lock(e,nom_exp),
    switch_lock(e,nom_exp,1) %lock
    add_log(e,'Make learning with experiment_code_histogram');
    % gets a batch of imagelets
    e=default(e.where);
    e.batch_size=1e5; % a big batch
    X=get_patch(e);
    energy = sum(X.^2,1);
    A_rand = rand(e.L,e.M)-0.5; A_rand = A_rand*diag(1./sqrt(sum(A_rand.*A_rand)));
    
    n_quant=50;
    frac = .2;
    % histogram of coefficients before and after
    % CG BEFORE
    S=zeros(e.M,e.batch_size); % initialize coeffs for CGF
    S=cgf_fitS(A_rand,X,e.noise_var_cgf,e.beta,e.sigma,e.tol);
    
    z = abs(S); % sparse correlation values (with normA = 1 )
    for i_batch=1:e.batch_size,
        z(:,i_batch) = z(:,i_batch) / sqrt(energy(i_batch));
    end;
    
    sv.cg0=mean(abs(S(:)));
    [proba.cg0, bin.cg0]=hist(z(:),n_quant);%
    
    % MP BEFORE
    S=zeros(e.M,e.batch_size); % initialize coeffs for MP
    Pz_j=1/e.n_quant*ones(e.n_quant,e.M); Mod=cumsum(Pz_j);
    switch_Mod=0;  switch_choice = 0;
    
    S=mp_fitS(A_rand,X,0,frac,switch_choice,Mod,switch_Mod,0,e.switch_sym);%
    
    z = abs(S); % sparse correlation values (with normA = 1 )
    for i_batch=1:e.batch_size,
        z(:,i_batch) = z(:,i_batch) / sqrt(energy(i_batch));
    end;
    sv.mp0=mean(abs(S(:)));
    [proba.mp0, bin.mp0]=hist(z(:),n_quant);
    
    
    % CG AFTER
    S=zeros(e.M,e.batch_size); % initialize coeffs for LGM
    load([e.where '/CG.mat']),
    A_cg = n.A*diag(1./sqrt(sum(n.A.*n.A)));
    
    S=cgf_fitS(A_cg,X,e.noise_var_cgf,e.beta,e.sigma,e.tol);
    z = abs(S); % sparse correlation values (with normA = 1 )
    for i_batch=1:e.batch_size,
        z(:,i_batch) = z(:,i_batch) / sqrt(energy(i_batch));
    end;
    sorted.cg =sort(z,'descend');
    sv.cg=mean(abs(S(:)));
    [proba.cg, bin.cg]=hist(z(:),n_quant);
    
    
    % MP AFTER
    S=zeros(e.M,e.batch_size); % initialize coeffs for MP
    load([e.where '/MP.mat']),
    switch_Mod=0;
    S=mp_fitS(n.A,X,0,frac,e.switch_choice,n.Mod,switch_Mod,0,e.switch_sym);%
    
    z = abs(S); % sparse correlation values (with normA = 1 )
    for i_batch=1:e.batch_size,
        z(:,i_batch) = z(:,i_batch) / sqrt(energy(i_batch));
    end;
    sorted.mp =sort(z,'descend');
    
    sv.mp = mean(abs(S(:)));
    
    [proba.mp, bin.mp]=hist(z(:),n_quant);
    
    unix(['rm -f ' e.where '/fig_coeff.*']);
    unix(['rm -f ' e.where '/fig_hist.*']);
    
    m_sorted.cg = mean(sorted.cg,2);
    m_sorted.mp = mean(sorted.mp,2);
    save('-v7',nom_exp,'proba','bin','sv','m_sorted')
    add_log(e,'Made experiment_code_histogram');
    
end;
%  unix(['rm -f ' e.where '/fig_coeff.*'])
%  unix(['rm -f ' e.where '/fig_hist.*'])
if switch_lock(e,nom_exp)==-1,
    load(nom_exp)
    
    
    
    color_cg = 'b--';%'k--';%
    color_mp = 'r';%'k';%
    color_cg_init = 'b+';%'k-.';%
    color_mp_init = 'r+';%'k+';%
    
    if ~exist([e.where '/fig_coeff.eps'],'file'),
        figure(6), set(gcf, 'visible', 'off'),
        hold on, plot(0,0,''),
        plot((1:e.M)/e.M,m_sorted.cg,color_cg,...
            (1:e.M)/e.M,m_sorted.mp,color_mp,'LineWidth',1.5)%
        hold off
        axis([0, 1, 1e-6, 1])% tight,%
        grid on, %
        xlabel('Rank (L0-norm)'),ylabel('Sparse Coefficient'),
        set(gca,'Color','none'),
        fig2pdf([e.where '/fig_coeff'],5,5)
    end
    
    if ~exist([e.where '/fig_hist.eps'],'file'),
        figure(7),set(gcf, 'visible', 'off'),
        proba.cg0= proba.cg0 / sum(proba.cg0);%
        proba.cg= proba.cg / sum(proba.cg); %
        proba.mp0= proba.mp0 / sum(proba.mp0);%
        proba.mp= proba.mp / sum(proba.mp);%
        plot(bin.cg0,proba.cg0,color_cg_init, ...
            bin.cg,proba.cg,color_cg,...
            bin.mp0,proba.mp0,color_mp_init,...
            bin.mp,proba.mp,color_mp,'LineWidth',1.5)%
        axis([0, 1, 1e-6, 1])%
        legend('SN-init','SN','aSSC-init','aSSC','Location','NorthEast'),
        grid on,
        set(gca,'YScale','log')
        
        xlabel('Sparse Coefficient'), ylabel('Probability'),
        fig2pdf([e.where '/fig_hist'],10,10)
        axis([0, .3, 1e-3, 1])%
        legend('SN-init','SN','aSSC-init','aSSC','Location','NorthEast'),
        set(gca,'YScale','linear')
        fig2pdf([e.where '/fig_hist_zoom'],10,10)
    end
end
