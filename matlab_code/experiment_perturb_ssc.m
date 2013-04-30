function experiment_perturb_ssc(e)
%---------------------------------
% Trying to see what happens if we reinitialize just one filter: 
%  * do we get back a similar edge: is the learning stable to this pertiurbation?
%  * is it the same filter?
%  * are other filters also changed? 

%## Author : Laurent Perrinet <Laurent.Perrinet@incm.cnrs-mrs.fr>
%## This software is distributed under the terms of the GPL

nom_exp=[e.where '/perturb.mat'];

if ~switch_lock(e,nom_exp) && switch_lock(e,[e.where '/MP.mat'])==-1 ,
    switch_lock(e,nom_exp,1) %lock
    e=default(e.where);
    e.video = 0;
    load(e.image_base); % a stacked 3-d matrix of natural images
    load([e.where '/MP.mat']),%

    % perturb
    n.A(:,1)=randn(e.L,1);
    n.A(:,1)=n.A(:,1)/sqrt(sum(n.A(:,1).^2));
    n.Pz_j(:,1)=1/e.n_quant*ones(e.n_quant,1); n.Mod=cumsum(n.Pz_j);
    
    % small trip to compute initial probabilities (just for the first figure
    % in fact)
    e.eta_ssc=0;
    e.switch_choice=1;
    e.num_trials=ceil(e.num_trials/8);
    e.Method='ssc'
    [n_init,s]=sparsenet(n,e);

    % fixes the broken RF from the perturbed n
    e=default(e.where);
    e.num_trials=ceil(e.num_trials/3);
    e.Method='ssc';
    [n_final,s]=sparsenet(n,e);

    save('-v7',nom_exp,'n_init','n_final')
    unix(['rm -f ' e.where '/fig_perturb_*'])

end;

%    unix(['rm -f ' e.where '/fig_perturb_*'])

if switch_lock(e,nom_exp) == -1,

    load(nom_exp)
    e=default(e.where);
    sub = 10;
    z = 0:sub/e.n_quant:1-1/e.n_quant ;
    if ~exist([e.where '/fig_perturb_init_P.eps'],'file'),
        figure(2),set(gcf, 'visible', 'off'), 
        subplot(211),  plot(z,n_init.Mod(1:sub:e.n_quant,2:e.M),'.'),
        hold on, plot(z,n_init.Mod(1:sub:e.n_quant,1)), hold off
        axis tight,xlabel('coeff'),ylabel('Mod'),
        subplot(212), semilogy(z,n_init.Pz_j(1:sub:e.n_quant,2:e.M),'.'),
        hold on, semilogy(z,n_init.Pz_j(1:sub:e.n_quant,1)), hold off
        axis([0, 1, 1e-5, 1])
        xlabel('coeff'),ylabel('Proba'),
        fig2pdf([e.where '/fig_perturb_init_P'],12,10)
    end
    if ~exist([e.where '/fig_perturb_final_P.eps'],'file'),
        figure(2),set(gcf, 'visible', 'off'),
        subplot(211),  plot(z,n_final.Mod(1:sub:e.n_quant,2:e.M),'.'),
        hold on, plot(z,n_final.Mod(1:sub:e.n_quant,1)), hold off
        axis tight,xlabel('coeff'),ylabel('Mod'), 
        subplot(212), semilogy(z,n_final.Pz_j(1:sub:e.n_quant,2:e.M),'.'),

        hold on, semilogy(z,n_final.Pz_j(1:sub:e.n_quant,1)), hold off
        axis([0, 1, 1e-5, 1])%
        xlabel('coeff'),ylabel('Proba*M'),
        fig2pdf([e.where '/fig_perturb_final_P'],12,10)
    end
    if ~exist([e.where '/fig_map_perturb_init.png'],'file'),
        n_init.A(:,1)=randn(e.L,1); 
        n_init.A(:,1)=n_init.A(:,1)/sqrt(sum(n_init.A(:,1).^2));

        imwrite((tile(n_init.A)+1)/2,[e.where '/fig_map_perturb_init.png'])
    end
    if ~exist([e.where '/fig_map_perturb_final.png'],'file'),
        imwrite((tile(n_final.A)+1)/2, [e.where '/fig_map_perturb_final.png'])
    end
end
