function experiment_stability_cgf(e)
% experiment_stability_cgf : compares the efficiency of the SPARSENET learning for
% different parameters

%## Author : Laurent Perrinet <Laurent.Perrinet@incm.cnrs-mrs.fr>
%## This software is distributed under the terms of the GPL

% stored default parameters
e=default(e.where);
n_param=4; n_amp =9;
nom_exp=[e.where '/stability_cgf.mat'];
% running all sub-experimentss
if ~(switch_lock(e,nom_exp)==-1),
    for i_param=1:n_param,
        v_amp=logspace(-1,1,n_amp); % (9 columns) amplification factor for the parameters
        for i_amp=1:length(v_amp),
            nom_exp_=[e.where '/stability_cgf_' num2str(i_param) '_' num2str(i_amp) '.mat'];
            if ~switch_lock(e,nom_exp_),
                switch_lock(e,nom_exp_,1) %lock
                add_log(e,['experiment_stability_cgf_' num2str(i_param) '/' num2str(n_param) '_' num2str(i_amp) '/' num2str(length(v_amp))]);
                e=default(e.where); %retrieves default parameters
                % and changes ONE
                switch i_param,
                    case 1, e.beta=v_amp(i_amp)*e.beta;
                    case 2, e.sigma=v_amp(i_amp)*e.sigma;
                    case 3, e.tol=v_amp(i_amp)*e.tol;
                    case 4, e.alpha=v_amp(i_amp)*e.alpha;
                end
                % sets a similar initialization for weights
                A_rand = rand(e.L,e.M)-0.5; A_rand = A_rand*diag(1./sqrt(sum(A_rand.*A_rand)));
                gain_rand=sqrt(sum(A_rand.*A_rand))';
                %%%%%%%%%%%%%
                % Sparsenet %
                %%%%%%%%%%%%%
                e.Method='cgf';
                n.A = A_rand; n.gain=gain_rand;
                n.S_var=e.VAR_GOAL*ones(e.M,1);
                [n,s]=sparsenet(n,e);
                save('-v7',nom_exp_,'n','s')
            end
        end
    end
end

if ~(switch_lock(e,nom_exp)==-1),
    % while running, things may have changed on the cluster: check if one simulation is still running
    switch_lock(e,nom_exp, 0) % unlock by default
    for i_param=1:n_param,
        for i_amp=1:n_amp,
            nom_exp_=[e.where '/stability_cgf_' num2str(i_param) '_' num2str(i_amp) '.mat'];
            if ~(switch_lock(e,nom_exp_)==-1),
                switch_lock(e,nom_exp, 1) % lock if one simulation is missing or running
            end
        end
    end
end
if ~switch_lock(e,nom_exp),
    switch_lock(e,nom_exp,1) %lock
    add_log(e,'Made learning with experiment_stability_cgf');
    add_log(e,'Make coding with experiment_stability_cgf');
    
    
    v_amp=logspace(-1,1,n_amp); % (9 columns) amplification factor for the parameters
    % additional paremeters for cgf (see their signification in cgf.c)
    % Defaults :
    %(1) e.beta=2.2;
    %(2) e.sigma=0.316;
    %(3) e.tol=.01; % parameter to stop CG when the gradient is to low in amplitude
    for i_param=1:n_param,
        for i_amp=1:length(v_amp),
            e=default(e.where); %retrieves default parameters
            % and changes ONE
            switch i_param,
                case 1, e.beta=v_amp(i_amp)*e.beta;
                case 2, e.sigma=v_amp(i_amp)*e.sigma;
                case 3, e.tol=v_amp(i_amp)*e.tol;
                case 4, e.alpha=v_amp(i_amp)*e.alpha;
            end
            nom_exp_=[e.where '/stability_cgf_' num2str(i_param) '_' num2str(i_amp) '.mat'];
            load(nom_exp_)
            
            % now test coding
            e.batch_size=100*e.batch_size % a big batch
            X=get_patch(e);
            
            S=zeros(e.M,e.batch_size); % initialize coeffs for LGM
            S=cgf_fitS(n.A,X,e.noise_var_cgf,e.beta,e.sigma,e.tol);
            E=X-n.A*S;
            
            % do some stats
            res_end(i_amp,i_param)=(1/e.noise_var_cgf)*sum(E(:).^2)/e.L/e.batch_size;%
            sparseness_end(i_amp,i_param)=e.beta/e.sigma*sum(log(1+S(:).^2/e.sigma^2))/e.M/e.batch_size;
            sparseness_default(i_amp,i_param)=e.beta/e.sigma*sum(log(1+S(:).^2/e.sigma^2))/e.M/e.batch_size;
            L1_end(i_amp,i_param)=sum(abs(S(:)))/e.M/e.batch_size;
            cost(i_amp,i_param)= res_end(i_amp,i_param) + ...
                sparseness_end(i_amp,i_param)*e.M/e.L;
            kurt(i_amp,i_param)=(sum((S(:)-( sum(S(:))/e.M/e.batch_size )).^4)/e.M/e.batch_size)/...
                (sum((S(:)-( sum(S(:))/e.M/e.batch_size )).^2)/e.M/e.batch_size)^2 -3; % Kurtosis
        end
    end
    save('-v7',nom_exp,'res_end','sparseness_end','sparseness_default','L1_end','cost','v_amp','kurt') %
    add_log(e,'Made coding with experiment_stability_cgf');
    
    
end;
if switch_lock(e,nom_exp)==-1,
    load(nom_exp)
    
    v_amp=logspace(-1,1,9); % (9 columns) amplification factor for the parameters
    for i_param=1:n_param,
        for i_amp=1:length(v_amp),
            if ~exist([e.where '/fig_stability_cgf_' num2str(i_param) '_' num2str(i_amp) '.png'],'file'),
                nom_exp_=[e.where '/stability_cgf_' num2str(i_param) '_' num2str(i_amp) '.mat'];
                load(nom_exp_)
                imwrite((tile(n.A)+1)/2,[e.where '/fig_stability_cgf_' num2str(i_param) '_' num2str(i_amp) '.png'])
            end
        end
    end
    %%%%% PLOTS
    for  i_param=1:n_param,
        e=default(e.where);
        switch i_param,
            case 1, v_=v_amp*e.beta;   nom='beta';
            case 2, v_=v_amp*e.sigma;  nom='sigma';
            case 3, v_=v_amp*e.tol;  nom='tol';
            case 4, v_=v_amp*e.alpha; nom='alpha';
        end
        
        if ~exist([e.where '/fig_stability_cgf_param_' nom '_res.eps'],'file'),
            figure(i_param), set(gcf, 'visible', 'off'), subplot(111),
            loglog(v_,res_end(:,i_param)),
            axis tight,%
            xlabel(nom),ylabel('Energy (L2-norm)'),
            fig2pdf([e.where '/fig_stability_cgf_param_' nom '_res'],10,10)
        end
        if ~exist([e.where '/fig_stability_cgf_param_' nom '_sparseness_default.eps'],'file'),
            figure(i_param), set(gcf, 'visible', 'off'), subplot(111),
            loglog(v_,sparseness_default(:,i_param)),
            axis tight,%
            xlabel(nom),ylabel('Olshausen''s sparseness'),
            fig2pdf([e.where '/fig_stability_cgf_param_' nom '_sparseness_default'],10,10)
        end
        if ~exist([e.where '/fig_stability_cgf_param_' nom '_sparseness.eps'],'file'),
            figure(i_param), set(gcf, 'visible', 'off'), subplot(111),
            loglog(v_,sparseness_end(:,i_param)),
            axis tight,%
            xlabel(nom),ylabel('Olshausen''s sparseness'),
            fig2pdf([e.where '/fig_stability_cgf_param_' nom '_sparseness'],10,10)
        end
        if ~exist([e.where '/fig_stability_cgf_param_' nom '.eps'],'file'),
            figure(i_param), set(gcf, 'visible', 'off'), subplot(111),
            loglog(v_,cost(:,i_param)),
            axis tight,%
            xlabel(nom),ylabel('Olshausen''s cost'),
            fig2pdf([e.where '/fig_stability_cgf_param_' nom],10,10)
        end
        
    end
end
