function [n,s]=sparsenet(n,e)
% sparsenet.m - simulates the sparse spike coding algorithm
%
% this is the core script for calling the coding algorithms and applying
% the learning schemes. it takes for a network architecture n and its
% corresponding experimental parameters a set of images contained in IMAGES
% and outputs a modified architecture and some (useful) statistics (s)
%
% Inputs : n=network, e=learning and experiment parameters, Images
% Outputs : n=network (modified with new basis functions), s=statistics
%
% The code is inspired from the Sparsenet algorithm from B. Olshausen
% see http://redwood.ucdavis.edu/bruno/sparsenet.html
%
% it should be run as is to get a comparison with the original algorithm
% (same parameters were used) by using higher level
% to run independently, run experiment_simple.m

%## Author : Laurent Perrinet <Laurent.Perrinet@incm.cnrs-mrs.fr>
%## This software is distributed under the terms of the GPL


add_log(e,['Make ' e.Method ' sparsenet with ' num2str(e.num_trials) ' learning steps']);
tic
if (e.display_every>0) && (~e.video)
    display_network(n,e);
end

s=[];
% --------------------------------------------
% main loop
% --------------------------------------------

for t=1:e.num_trials
    % gets a batch of imagelets
    X=get_patch(e);
    S=zeros(e.M,e.batch_size); % initialize coeffs for LGM
    dA=zeros(size(n.A)); % initalize weight's gradient
    
    % --------------------------------------------
    % SPARSIFICATION
    if strcmp(e.Method,'ssc') | strcmp(e.Method,'amp'),% sparsify by Matching Pursuit
        [S, dA, Pz_j_]=mp_fitS(n.A,X,e.noise_var_ssc,e.frac,e.switch_choice,n.Mod,0,0,e.switch_sym);%
    else
        % sparsify by Conjugate Gradient
        % calculate coefficients for these data via conjugate gradient routine
        S=cgf_fitS(n.A,X,e.noise_var_cgf,e.beta,e.sigma,e.tol);
    end
    
    % residual
    E=X-n.A*S;
    
    % --------------------------------------------
    % STATISTICS
    
    if e.switch_stats,
        % compute some statistics. In general we try to keep them dimension
        % less (e.g. the energy is per pixel, the L1-norm per coefficient)
        % to avoid further scaling problems and keep useful ranges in mind
        % for different sets of parameters, and compare different choices
        % of parameters
        
        %%  how much energy did we grab? that is, how much information
        %%  (-logP) did we take in the coding sweep? note the normalization
        %%  by the norm of A (see Perrinet04tauc)
        
        s.res(t)=sum(E(:).^2)/e.L/e.batch_size;%
        s.L0(t)=sum(S(:)~=0)/e.M/e.batch_size; % i.e. ~1 in CG (most are *close* to zero but not zero) and the mean proportion of active neurons in MP
        normA=sqrt(sum(n.A.*n.A));
        coeff=S.*(normA'*ones(1,e.batch_size)); % these correspond to the rectified coeffs (independently of the norm of the filter)
        % it is not done apparently in original sparsenet.m and cgf.c
        % s.L1(t)=sum(abs(coeff(:)))/e.M/e.batch_size; % accounts for the norm of the filters
        % the s.ol sparseness term comes from the cgf.c file (line 183) and corresponds to
        % the cost function used in (the original) SparseNet. It corresponds to a parametric model of
        % the distribution of coefficients. I use the corrected coeff since
        % they don't show the pathological behavior of the cost at the very
        % beginning (see Figure 13.2 of Prob. Models of the Brain from Rao et al)
        s.ol(t)=e.beta*sum(log(1+coeff(:).^2/e.sigma^2))/e.M/e.batch_size;
        s.kurt(t)=(sum((coeff(:)-( sum(coeff(:))/e.M/e.batch_size )).^4)/e.M/e.batch_size)/...
            (sum((coeff(:)-( sum(coeff(:))/e.M/e.batch_size )).^2)/e.M/e.batch_size)^2 -3; % Kurtosis
        
        
        if e.switch_verbosity>=3,%
            add_log(e,['t= ' num2str(t,'%0.6d') '/' num2str(e.num_trials) '|res=' num2str(s.res(t)) '|L0=' num2str(s.L0(t)) '|kurt=' num2str(s.kurt(t)) '|ol=' num2str(s.ol(t))  '| elapsed time = ' num2str(toc)] );
        end;
        
    end
    
    % display information and basis functions (to display / disk)
    if (mod(t,e.display_every)==1) && (e.display_every>0)
        disp([' t= ' num2str(t) '/' num2str(e.num_trials), ' elapsed time = ',num2str(toc)])
        if e.video,
            nom=[e.where '/tmp/' e.Method '_' num2str(t,'%0.6d') '.png'];
            imwrite((tile(n.A)+1)/2,nom);%
        end
        if e.switch_verbosity>3,     if e.switch_stats,
                display_network(n,e);
                figure(4), set(gcf,'MenuBar','none','ToolBar','none')%
                array=tile(E*S',0);%
                imagesc(array,[-1 1])%
                axis image off
                ax = gca();
                set (ax, 'position', [0., 0., 1., 1.]);
                colormap gray
                
                figure(3), set(gcf,'MenuBar','none','ToolBar','none')%
                
                subplot(131),    plot(s.res),
                hold on, plot(0,0,''), hold off
                axis tight, %title('Learning : Energy of residual'),
                xlabel('Learning Step'), ylabel('Residual Energy (L2 norm)'),
                if strcmp(e.Method,'cgf'), %
                    subplot(132),  plot(s.ol),
                    hold on, plot(0,0,''), hold off
                    axis tight, %title('Learning: L1-sparseness of representation'),
                    xlabel('Learning Step'),ylabel('Olshausen''s sparseness'),
                    subplot(133),  plot(1/e.noise_var_cgf*s.res+s.ol*e.M/e.L),
                    hold on, plot(0,0,''), hold off
                    axis tight,
                    xlabel('Learning Step'),ylabel('Olshausen''s cost'),
                else
                    subplot(132),  plot(s.L0),
                    hold on, plot(0,0,''), hold off
                    axis tight, %title('Learning: L0-sparseness of representation'),
                    xlabel('Learning Step'),ylabel('L0 sparseness'),
                    
                    subplot(133),  plot(1/e.noise_var_cgf*s.res+s.L0*log2(e.M))%
                    hold on, plot(0,0,''), hold off
                    axis tight,
                    xlabel('Learning Step'),ylabel('Coding cost'),
                end
                figure(2),  set(gcf,'MenuBar','none','ToolBar','none')%
                subplot(211),  plot(n.Mod),
                ylabel('z'), axis tight,%
                subplot(212), semilogy(n.Pz_j.*(n.Pz_j>1e-6)*e.M + 1e-6 .*(n.Pz_j<1e-6) ), xlabel('sparse coefficient'),ylabel('Proba*M'), axis tight, drawnow
            end
        end;
    end
    
    %% --------------------------------------------
    %% apply the learning gradient (dA) on the network (n) and modify
    %% homeostatic variables
    
    if strcmp(e.Method,'ssc') | strcmp(e.Method,'amp'),
        eta = e.eta_ssc;% sparsify by Matching Pursuit
    else
        eta = e.eta_cgf;% or by CGF
        % TODO: remove dA in mp_fitS if strcmp(e.Method,'cgf'), % eq. 17 in [Olshausen1998]
        %    dA = E*S'; % Hebb rule See Eq. 17 in Olshausen et al. 1998
            % it's the same for the other method, but we did compute it
            % directly during the coding
        dA=zeros(e.L,e.M);
        for i=1:e.batch_size
            dA = dA + E(:,i)*S(:,i)';
        end
    end
    %% LEARNING : it is the same for both methods
    if (eta>0), %        
        % 1) updates basis functions
        
        %% if you increase the batch size, the gradient increases proportionnally
        %% (with ergodicity...), so by Knuth programming law ("Thou shall
        %% make your program scale invariant")
        dA = dA/e.batch_size;
        
        %% applies the gradient descent
        n.A = n.A + eta * dA;%
        
    end% end learning loop


    %% HOMEOSTASIS : it's different for both methods
    % 2) update the norm and average use of every neuron (homeostatic rules)
    normA=sqrt(sum(n.A.*n.A));
    % normalization
    for i_M=1:e.M %over basis functions
        n.A(:,i_M)=n.A(:,i_M)/normA(i_M);
    end

    %% adaptive gain for the competition between filters
    if strcmp(e.Method,'cgf'), % eq. 17 in [Olshausen1998]
        for i_batch=1:e.batch_size,
            n.S_var = (1-e.var_eta_cgf)*n.S_var + e.var_eta_cgf*S(:,i_batch).*S(:,i_batch);
        end
        n.gain = n.gain .* ((n.S_var/e.VAR_GOAL).^e.alpha);
        %             disp(std(n.S_var)/mean(n.S_var))
        % normalization
        for i_M=1:e.M % over basis functions
            n.A(:,i_M)=n.gain(i_M)*n.A(:,i_M);
        end
        
    else % egalitarian homeostasis in SSC
        if e.var_eta_ssc >0,
            %  adaptive rule for homeostasis
            t_homeo = 1/e.var_eta_ssc; % TODO: remove? min(t,1/e.var_eta_ssc); %
            n.Pz_j=(1-1/t_homeo)*n.Pz_j+1/t_homeo*Pz_j_;% update statistics
            n.Mod=cumsum(n.Pz_j); %
        end        
    end
    % end iteration loop
end
% --------------------------------------------
% end main loop
% --------------------------------------------
ttoc=toc;
if e.switch_verbosity>1, add_log(e,[' Done in  ' num2str(ttoc) ' seconds']); end;
