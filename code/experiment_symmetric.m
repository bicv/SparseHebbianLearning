function experiment_symmetric(e)
% experiment_symmetric.m : compares with or without the symmetric constraint
%-------------------------

% generates a set of filters without the symmetry of filters (ON-OFF) constraint
% for the Matching Pursuit scheme

%## Author : Laurent Perrinet <Laurent.Perrinet@incm.cnrs-mrs.fr>
%## This software is distributed under the terms of the GPL


%%  Learning
%% ----------
% generates the missing learning with all stats and no video
nom_exp=[e.where '/MP_no_sym.mat'];
nom_exp_ =[e.where '/MP_no_sym_learn.mat'];
if ~switch_lock(e,nom_exp_),
    switch_lock(e,nom_exp_,1) %lock
    
    e=default(e.where);
    add_log(e,'Make learning with experiment_symmetric');
    e.Method='ssc';
    
    % MP - without symmetry (but same degree of freedom, that is twice the number of filters)
    e.M=2*e.M;
    A_rand = rand(e.L,e.M)-0.5; A_rand = A_rand*diag(1./sqrt(sum(A_rand.*A_rand)));
    n.A = A_rand;
    n.Pz_j=1/e.n_quant*ones(e.n_quant,e.M); n.Mod=cumsum(n.Pz_j);
    
    e.switch_sym=0; % a switch in mp_fitS.m
    
    [n,s]=sparsenet(n,e)
    
    save('-v7',nom_exp_,'n','s')
    
end

if switch_lock(e,nom_exp_)==-1 && ~switch_lock(e,nom_exp),
    switch_lock(e,nom_exp,1) %lock
    
    load(nom_exp_)
    
    e=default(e.where);
    e.M=2*e.M;
    add_log(e,'Make coding with experiment_symmetric');
    e.Method='ssc';
    e.switch_sym=0; % a switch in mp_fitS.m
    
    e.batch_size=100*e.batch_size; % a big batch
    X=get_patch(e);
    
    % test out the efficiency of the methods
    e.switch_Mod = 0 ;
    S=mp_fitS(n.A,X,0,1,e.switch_choice,n.Mod,e.switch_Mod, 0,e.switch_sym);%
    S_=zeros(size(S));
    v_frac = 1-exp(log(10)*(-2.5:.01:-.0)); % fraction of minimal activity for the threshold
    for i_frac=1:length(v_frac)
        for i_batch=1:e.batch_size,
            max_S=max(abs(S(:,i_batch)));
            ind=find(abs(S(:,i_batch))>v_frac(i_frac)*max_S); % the basis functions we use
            S_(ind,i_batch)=S(ind,i_batch);
        end
        E=X-n.A*S_;
        res_sym(i_frac)=sum(E(:).^2)/e.L/e.batch_size;
        L0_sym(i_frac)=sum(S_(:)~=0)/e.M/e.batch_size;
        proba_j=sum(S_~=0,2);proba_j=proba_j/sum(proba_j);
        entropy_sym(i_frac)= sum(- log2(proba_j+eps) .* proba_j );
    end
    
    cmd= ['rm -f ' e.where '/fig_sym_* ' ];
    unix(cmd)
    
    save('-v7',nom_exp,'v_frac','L0_sym','res_sym','entropy_sym');
    add_log(e,'Made learning with experiment_symmetric');
end;
if switch_lock(e,nom_exp)==-1 && switch_lock(e,[e.where '/MP.mat'])==-1 && switch_lock(e,[e.where '/L0_stats.mat'])==-1,
    e=default(e.where);
    
    nom_exp_ =[e.where '/MP_no_sym_learn.mat'];
    load(nom_exp_), s_mp_non_sym=s;
    if ~exist([e.where '/fig_sym_ssc_map.png'],'file'),
        add_log(e,'Generate image of symmetric map');
        tile_=tile(n.A(:,1:e.M));
        imwrite(([tile_(:,1:sqrt(e.M)*(sqrt(e.L) +1)) tile(n.A(:,(e.M+1):(2*e.M)))] +1)/2, [e.where '/fig_sym_ssc_map.png'])
    end
    
    
    load([e.where '/MP.mat']), s_mp=s;
    
    % Smoothing the data accross learning steps
    if ~exist([e.where '/fig_sym_phase.eps'],'file'),
        figure(7),set(gcf, 'visible', 'off'),
        v_smooth=1:e.n_mean:(e.num_trials-e.n_mean); % where we smooth
        res_mp_non_sym_=smooth(s_mp_non_sym.res,e.n_mean);  res_mp_=smooth(s_mp.res,e.n_mean);
        L0_non_sym_=smooth(s_mp_non_sym.L0,e.n_mean);    L0_mp_=smooth(s_mp.L0,e.n_mean);
        
        axis ij, hold on
        plot(L0_non_sym_,res_mp_non_sym_,'r',L0_mp_,res_mp_,'b')
        hold on, plot(0,0,''), hold off
        
        legend('no sym','sym','Location','SouthEast')%
        axis xy, grid on,
%        set(gca,'XTick',[1 50 100]),    set(gca,'XTickLabel',ceil([0 ol_max/2  ol_max]*100)/100)
%        set(gca,'YTick',[1 50 100]),    set(gca,'YTickLabel',ceil([0 E_max/2  E_max]*100)/100)
        hold off
        xlabel('Sparseness (L0 norm)'),ylabel('Residual Energy'),
        fig2pdf([e.where '/fig_sym_phase'],10,10)
    end
    
    load([e.where '/L0_stats.mat'])
    load(nom_exp)
    
    color_sym = 'b--';%'k--';%
    color_mp = 'r';%'k';%
    
    if ~exist([e.where '/fig_sym_efficiency_L0.eps'],'file'),
        figure(9),set(gcf, 'visible', 'off'),
        plot([  L0_sym],[  res_sym/res.total],color_sym,...
            [  L0.mp],[  res.mp/res.total],color_mp,'LineWidth',1.5)%
        hold on, plot(0,0,''), hold off
        legend('no sym','ssc','Location','NorthEast')%
        axis([0, .5, 0, 1])%
        grid on, xlabel('Sparseness (L0-norm)'),ylabel('Residual Energy (L2 norm)'),
        fig2pdf([e.where '/fig_sym_efficiency_L0'],10,10)
    end
    
    
end
