function experiment_learn(e)
% experiment_learn.m : does the learning with both methods
% --------------------------------------------------------

% performs (a lot of) experiments for comparing the learning schemes

% /!\ long and pretends to be complete, for a simpler experiment, see
% experiment_simple.m

%## Author : Laurent Perrinet <Laurent.Perrinet@incm.cnrs-mrs.fr>
%## This software is distributed under the terms of the GPL

% stored default parameters
e=default(e.where);
load(e.image_base); % a stacked 3-d matrix of natural images

%%  Learning
%% ----------
% generates 2 learning with all stats and video
nom_exp_CG=[e.where '/CG.mat'];
nom_exp_MP=[e.where '/MP.mat'];
if ~(switch_lock(e,nom_exp_MP)==-1 && switch_lock(e,nom_exp_CG)==-1),
    add_log(e,'Make learning with experiment_learn');
    
    A_rand = rand(e.L,e.M)-0.5; A_rand = A_rand*diag(1./sqrt(sum(A_rand.*A_rand)));
    gain_rand=sqrt(sum(A_rand.*A_rand))';
    if ~exist([e.where '/fig_map_rand.png'],'file'),
        imwrite((tile(A_rand)+1)/2,[e.where '/fig_map_rand.png'])
    end
    % MP
    if ~switch_lock(e,nom_exp_MP),
        switch_lock(e,nom_exp_MP,1) % lock
        e=default(e.where); %loads default parameters
        e.Method='ssc'; e.video = 1;
        n.A = A_rand;
        n.Pz_j=1/e.n_quant*ones(e.n_quant,e.M); n.Mod=cumsum(n.Pz_j);
        [n,s]=sparsenet(n,e);
        save('-v7',nom_exp_MP,'n','s')
    end
    % CG
    if ~switch_lock(e,nom_exp_CG),
        switch_lock(e,nom_exp_CG,1) % lock
        e=default(e.where);%loads default parameters
        e.Method='cgf'; e.video = 1;
        n.A = A_rand; n.gain=gain_rand;
        n.S_var=e.VAR_GOAL*ones(e.M,1);
        [n,s]=sparsenet(n,e);
        save('-v7',nom_exp_CG,'n','s')
    end
    cmd= ['rm -f ' e.where '/fig_learn_* ' e.where '/fig_map_cgf.png ' e.where '/fig_map_ssc.png ' ];
    unix(cmd)
    add_log(e,'Made learning with experiment_learn');
    
end;
if switch_lock(e,nom_exp_MP)==-1 && switch_lock(e,nom_exp_CG)==-1,
    
    load(nom_exp_CG), s_cg=s; n_cg =n;
    load(nom_exp_MP), s_ssc=s; n_ssc =n;
    if ~exist([e.where '/fig_map_cgf.png'],'file'),
        add_log(e,'Generate image of the basis functions');
        imwrite((tile(n_cg.A)+1)/2,[e.where '/fig_map_cgf.png'])
    end
    if ~exist([e.where '/fig_map_ssc.png'],'file'),
        imwrite((tile(n_ssc.A)+1)/2,[e.where '/fig_map_ssc.png'])
    end
    
    if ~exist([e.where '/fig_learn_energy.eps'],'file') | ...
            ~exist([e.where '/fig_learn_ol_cost.eps'],'file') | ...
            ~exist([e.where '/fig_learn_ol.eps'],'file') | ...
            ~exist([e.where '/fig_learn_ol_phase.eps'],'file') | ...
            ~exist([e.where '/fig_learn_kurtosis.eps'],'file'),
        
        % Smoothing the data accross learning steps
        v_smooth=1:e.n_mean:(e.num_trials-e.n_mean); % where we smooth
        res_cg_=smooth(s_cg.res,e.n_mean);  res_ssc_=smooth(s_ssc.res,e.n_mean);
        ol_cg_=smooth(s_cg.ol,e.n_mean);    ol_ssc_=smooth(s_ssc.ol,e.n_mean);
        L0_cg_=smooth(s_cg.L0,e.n_mean);    L0_ssc_=smooth(s_ssc.L0,e.n_mean);
        kurt_cg_=smooth(s_cg.kurt,e.n_mean);    kurt_ssc_=smooth(s_ssc.kurt,e.n_mean);
    end
    
    if ~exist([e.where '/fig_learn_energy.eps'],'file'),
        figure(5),set(gcf, 'visible', 'off'),
        plot(v_smooth,res_cg_,'r',v_smooth,res_ssc_,'b'),
        hold on, plot(0,0,''), hold off
        axis tight,
        xlabel('Learning Step'), ylabel('Residual Energy (L2 norm)'),legend('cgf','ssc','Location','NorthEast')
        fig2pdf([e.where '/fig_learn_energy'],12,10)
        
    end
    
    if ~exist([e.where '/fig_learn_ol.eps'],'file'),
        figure(6),set(gcf, 'visible', 'off'),
        plot(v_smooth,ol_cg_,'r',v_smooth,ol_ssc_,'b'),
        hold on, plot(0,0,''), hold off
        axis tight,
        xlabel('Learning Step'),ylabel('Olshausen''s sparseness'),legend('cgf','ssc','Location','NorthEast'),hold off
        fig2pdf([e.where '/fig_learn_ol'],12,10)
        
    end
    if ~exist([e.where '/fig_learn_ol_cost.eps'],'file'),
        figure(7),set(gcf, 'visible', 'off'),
        plot(v_smooth,1/e.noise_var_cgf*res_cg_+ol_cg_*e.M/e.L,'r',...
            v_smooth,1/e.noise_var_cgf*res_ssc_+ol_ssc_*e.M/e.L,'b'),
        hold on, plot(0,0,''), hold off
        axis tight,
        xlabel('Learning Step'),ylabel('Olshausen''s cost'),legend('cgf','ssc','Location','NorthEast'),hold off
        fig2pdf([e.where '/fig_learn_ol_cost'],12,10)
        
    end
    if ~exist([e.where '/fig_learn_ol_phase.eps'],'file'),
        figure(8),set(gcf, 'visible', 'off'),
        ol_max=full(max(max(ol_ssc_),max(ol_cg_)));
        E_max=full(max(max(res_cg_),max(res_ssc_)));
        [Y,X] = meshgrid(0:ol_max/100:ol_max, 0:E_max/100:E_max);
        axis ij, hold on
        plot(ol_cg_/ol_max*100,res_cg_/E_max*100,'r',ol_ssc_/ol_max*100,res_ssc_/E_max*100,'b'),
        legend('cgf','ssc','location','northwest')
        axis xy, grid on,
        hold off
        xlabel('Olshausen''s Sparseness'),ylabel('Residual Energy (L2 norm)'),
        fig2pdf([e.where '/fig_learn_ol_phase'],10,10)
        
    end
    if ~exist([e.where '/fig_learn_L0.eps'],'file'),
        figure(9),set(gcf, 'visible', 'off'),
        plot(v_smooth,L0_cg_,'r',v_smooth,L0_ssc_,'b'),
        hold on, plot(0,0,''), hold off
        axis tight,%
        xlabel('Learning Step'),ylabel('Sparseness (L0 norm)'),legend('cgf','ssc','Location','NorthEast'),hold off
        fig2pdf([e.where '/fig_learn_L0'],12,10)
        
    end
    if ~exist([e.where '/fig_learn_kurtosis.eps'],'file'),
        % shows that distibutions get more kurtotic ( s.kurt )
        figure(11),set(gcf, 'visible', 'off'),
        plot(v_smooth,kurt_cg_,'r',v_smooth,kurt_ssc_,'b'),
        axis tight,%
        legend('cgf','ssc','Location','East'),hold off
        xlabel('Learning Step'),ylabel('Kurtosis'),hold off
        fig2pdf([e.where '/fig_learn_kurtosis'],12,10)
    end
    
    nom_movie=[e.where '/fig_parallel.avi'];
    if 0% ~exist(nom_movie,'file') ,
        % Makes a video of the learning
        % -----------------------------
        % now OBSOLETE
        %
        % generates a video of the learning... for presentations
        % you should have done the learning before... (see experiment_learn.m)
        % PRO : useful to show for talks && presentations
        % CONs : long to compute! lots of ppm files! it is simpler to use ImageMagick's convert function...
        
        
        add_log(e,'Make movie for experiment_make_movie');
        
        if 1, %montage
            figure(1, 'visible', 'off')
            mov = avifile(nom_movie)
            %mov.Quality = 100;
            mov.Fps = 15;
            % makes the montage
            for t=1:e.display_every:e.num_trials,
                disp(t)
                tile_cg=double(imread([e.where '/tmp/cgf_' num2str(t,'%0.6d') '.png']))/256;
                tile_ssc=double(imread([e.where '/tmp/ssc_' num2str(t,'%0.6d') '.png']))/256;
                A_big=[tile_cg ones(size(tile_cg,1),20) tile_ssc];
                h=imagesc(A_big); colormap(gray); axis off,
                M(t) = getframe;
                set(h,'EraseMode','xor');
                axis equal
                set(gcf,'Position',[1 1 size(A_big,2) size(A_big,1)])
                subplot('position',[0 0 1 1])
                drawnow
                F = getframe(gca);%,[s*frac s*frac s s]);
                mov = addframe(mov,F);
            end
            % MAKES VIDEO
            mov = close(mov)
            
        else, % 2 separate movies
            mov = avifile([e.where '/fig_movie_cgf.avi'])
            %mov.Quality = 100;
            mov.Fps = 15;
            % makes the montage
            for t=1:e.display_every:e.num_trials,
                disp(t)
                tile_cg=double(imread([e.where '/tmp/cgf_' num2str(t,'%0.6d') '.png']))/256;
                h=imagesc(tile_cg); colormap(gray); axis off,
                M(t) = getframe;
                set(h,'EraseMode','xor');
                axis equal
                set(gcf,'Position',[1 1 size(tile_cg,2) size(tile_cg,1)])
                subplot('position',[0 0 1 1])
                drawnow
                F = getframe(gca);%,[s*frac s*frac s s]);
                mov = addframe(mov,F);
            end
            % MAKES VIDEO
            
            mov = close(mov)
            mov = avifile([e.where '/fig_movie_ssc.avi'])
            %mov.Quality = 100;
            mov.Fps = 15;
            % makes the montage
            for t=1:e.display_every:e.num_trials,
                disp(t)
                tile_ssc=double(imread([e.where '/tmp/ssc_' num2str(t,'%0.6d') '.png']))/256;
                h=imagesc(tile_ssc); colormap(gray); axis off,
                M(t) = getframe;
                set(h,'EraseMode','xor');
                axis equal
                set(gcf,'Position',[1 1 size(tile_ssc,2) size(tile_ssc,1)])
                subplot('position',[0 0 1 1])
                drawnow
                F = getframe(gca);%,[s*frac s*frac s s]);
                mov = addframe(mov,F);
            end
            % MAKES VIDEO
            mov = close(mov)
        end
        
    end
end
