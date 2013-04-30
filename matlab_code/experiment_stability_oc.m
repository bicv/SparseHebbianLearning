function experiment_stability_oc(e)
% experiment_stability_oc : compares the efficiency of the sparsification
% at the end of the learning for different over_completenesses

%## Author : Laurent Perrinet <Laurent.Perrinet@incm.cnrs-mrs.fr>
%## This software is distributed under the terms of the GPL

nom_exp=[e.where '/stability_oc.mat'];
%unix(['rm -f ' nom_exp])
%oc=[5 8 13 21 ].^2; % squared golden spiral
oc=[5 8 13 21 34].^2; % squared golden spiral

if ~(switch_lock(e,nom_exp)==-1),

    % stored default parameters
    e=default(e.where);
    for i_oc=1:length(oc),
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Sparsenet / MP_Sparsenet %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%

        nom_exp_=[e.where '/stability_oc_' num2str(i_oc) '.mat'];
        if ~exist(nom_exp_,'file'),
            switch_lock(e,nom_exp_, 1) % unlock by default
            add_log(e,['experiment_stability_oc_' num2str(i_oc) '/' num2str(length(oc)) '; oc = ' num2str(oc(i_oc)) ]);
            % sets a similar initialization for weights
            e=default(e.where);
            M = e.M; % old number of filters
            e.M = oc(i_oc); % new number of filters
            A_rand = rand(e.L,e.M)-0.5; A_rand = A_rand*diag(1./sqrt(sum(A_rand.*A_rand)));
            n.A = A_rand;
            n.Pz_j=1/e.n_quant*ones(e.n_quant,e.M); n.Mod=cumsum(n.Pz_j);
            e.Method='ssc';
            e.num_trials = ceil(e.num_trials/e.L*e.M); % increase learning time proportionnaly to the number of filters to be fair
            [n,s]=sparsenet(n,e);
            save('-v7',nom_exp_,'n','s')
        end
    end
end


if ~(switch_lock(e,nom_exp)==-1),
    % while running, things may have changed on the cluster: check if one simulation is still running
    switch_lock(e,nom_exp, 0) % unlock by default
    for i_oc=1:length(oc),
        nom_exp_=[e.where '/stability_oc_' num2str(i_oc) '.mat'];
        if ~(switch_lock(e,nom_exp_)==-1),
            switch_lock(e,nom_exp, 1) % lock if one simulation is missing or running
        end
    end
end


if ~switch_lock(e,nom_exp),
    switch_lock(e,nom_exp,1) %lock

    add_log(e,'Make coding with experiment_stability_oc');
    % stored default parameters
    e=default(e.where);
    e.batch_size=1e4; % a big batch
    X=get_patch(e);
    res_total=sum(X(:).^2);%
    %oc=[5 8 13 21 34].^2; % squared golden spiral
    mse = zeros(length(oc),max(oc)+1)*nan;
    L0 = mse;
    for i_oc=1:length(oc),
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Sparsenet / MP_Sparsenet %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        e=default(e.where);
        e.M = oc(i_oc);
        e.Method='ssc';

        nom_exp_=[e.where '/stability_oc_' num2str(i_oc) '.mat'];
        load(nom_exp_)

        if ~exist([e.where '/fig_stability_oc_' e.Method '_' num2str(i_oc) '.png'],'file'),
            imwrite((tile(n.A)+1)/2,[e.where '/fig_stability_oc_' e.Method '_' num2str(i_oc) '.png'])
        end

        e.switch_Mod = 0; % we compare here the *coding* methods, not the full algorithm (CG has no spike decoding scheme)
        S=mp_fitS(n.A,X,0,1,e.switch_choice,n.Mod,e.switch_Mod, 0,e.switch_sym);%

        [S_sort, S_ind] = sort(-abs(S));

        for i_frac = 0:e.M,
            S_=S;
            if i_frac < e.M,
                for i_batch=1:size(X,2),
                    to_zero = S_ind((i_frac+1):e.M,i_batch);
                    S_(to_zero,i_batch)=0;
                end
            end
            E = X - n.A*S_;
            mse(i_oc,i_frac+1)=sum(E(:).^2)/res_total/size(X,2);%/e.L/e.batch_size;
            L0(i_oc,i_frac+1)=sum(S_(:)~=0)/e.M/size(X,2);
        end
    end

    unix(['rm -f ' e.where '/fig_stability_oc_*eps'])
    save('-v7',nom_exp,'mse','L0','oc','res_total'); %
    add_log(e,'Made coding with experiment_stability_oc');

end;

% unix(['rm -f ' e.where '/fig_stability_oc_*eps'])

if switch_lock(e,nom_exp) == -1,
    e=default(e.where);
    load(nom_exp)
    bits = (log2(oc) + 1)/e.L;

    for i_oc=1:length(oc),    
        nom_exp_=[e.where '/stability_oc_' num2str(i_oc) '.mat'];
        load(nom_exp_)
        if ~exist([e.where '/fig_stability_oc_ssc_' num2str(i_oc) '.png'],'file'),
            imwrite((tile(n.A)+1)/2,[e.where '/fig_stability_oc_ssc_' num2str(i_oc) '.png'])
        end
    end 

    %%%%% PLOTS
    % displays end of learning perfromance for different learnig rates
    % information transmission
    if ~exist([e.where '/fig_stability_oc_energy.eps'],'file'),
        figure(3),set(gcf, 'visible', 'off'),
        plot(mse'),        hold on, plot(0,0,''), hold off
        axis tight,%
        xlabel('full L0'),
        ylabel('Mean residual Energy'),
        legend('5^2','8^2','13^2','21^2','34^2','Location','NorthEast')
        fig2pdf([e.where '/fig_stability_oc_energy'],10,10)
    end
    % Sparseness
    if ~exist([e.where '/fig_stability_oc_L0.eps'],'file'),
        figure(4),set(gcf, 'visible', 'off'),
        plot((L0.*(oc'*ones(1,max(oc)+1)))'),        hold on, plot(0,0,''), hold off
        axis tight,%
        xlabel('L0_{frac}'),
        ylabel('L0 sparseness'),
        legend('5^2','8^2','13^2','21^2','34^2','Location','NorthEast')
        fig2pdf([e.where '/fig_stability_oc_L0'],10,10)
    end

    if ~exist([e.where '/fig_stability_oc_L0_res.eps'],'file'),
        figure(5),set(gcf, 'visible', 'off'),
        semilogx((L0.*((oc .*bits)'*ones(1,max(oc)+1)))',mse'),        hold on, plot(0,0,''), hold off
        axis tight,%
        xlabel('bits per pixel '),ylabel('Mean residual Energy'),
        legend('5^2','8^2','13^2','21^2','34^2','Location','NorthEast')
        fig2pdf([e.where '/fig_stability_oc_L0_res'],10,10)

    end

    if ~exist([e.where '/fig_stability_oc_occam.eps'],'file'),

        figure(6),set(gcf, 'visible', 'off'),
        compression = zeros(length(oc),1);
        for i_oc=1:length(oc),        
            compression(i_oc) = L0(i_oc,oc(i_oc))* bits(i_oc) / oc(i_oc) / mse(i_oc,oc(i_oc));
        end
        
        bar(oc/e.L,compression),
        axis tight,
        xlabel('Over-completeness factor'),ylabel('Compression '),
        fig2pdf([e.where '/fig_stability_oc_occam'],10,10)
    end

end
