function experiment_stats_images(e)
% experiment_stats_images : gets a raw estimate of the noise in the images
% from the base


%## Author : Laurent Perrinet <Laurent.Perrinet@incm.cnrs-mrs.fr>
%## This software is distributed under the terms of the GPL


% stored default parameters
e=default(e.where);


%%  Conputations
%% ----------
% generates 2 learning with all stats and video
nom_exp=[e.where '/stats.mat'];
if ~exist(nom_exp,'file'),
    
    add_log(e,'Make learning with experiment_stats_images');
    e.batch_size=5000; % many for the stats
    
    % compute covariance matrix
    % we may represent it as images of images the covariance of the other
    % pixels being computed knowing one pixel
    
    X=get_patch(e);
    % to get the threshold saying what is good to learn :
    [Energy_hist,xout]=hist((sum(X.^2,1)/e.L),100);
    
    Xcov=zeros(e.L);
    for i_x=1:e.L,
        for j_x=1:i_x, % e.L, divide computation times : matric is symmetric
            Xcov(i_x,j_x)=(sum(X(i_x,:).*X(j_x,:))/e.L/e.batch_size)-...
                (sum(X(i_x,:))/e.L/e.batch_size)*(sum(X(j_x,:))/e.L/e.batch_size);
            Xcov(j_x,i_x)=Xcov(i_x,j_x);
        end
    end
    % mean energy of images= diagonal of Xcov
    K_wiener=pinv(Xcov);
    save('-v7',nom_exp,'Xcov','K_wiener','Energy_hist','xout')
    clear X dump
    
    add_log(e,'Made learning with experiment_stats_images');
    
end;


if exist(nom_exp,'file'),
    load(nom_exp)
    % makes imagelets
    if ~exist([e.where '/fig_stats_imagelets.png'],'file'),
        add_log(e,'Make imagelets');
        e.batch_size=14^2;
        X=get_patch(e);
        for i_batch_size=1:e.batch_size,
            kurt(i_batch_size)=(sum( (X(:,i_batch_size)-(sum(X(:,i_batch_size))/e.L) ).^4)/e.L)/(sum((X(:,i_batch_size)-(sum(X(:,i_batch_size))/e.L)).^2)/e.L)^2 -3; % Kurtosis
        end
        [dump, ind] = sort(kurt,'descend');
        disp(['Kurtosis: Min =  ', min(kurt), ' max=' , max(kurt)])
        imagesc(tile(X(:,ind))), colormap gray
        imwrite((tile(X,0)+1)/2,[e.where '/fig_stats_imagelets.png'])
    end
    
    if ~exist([e.where '/fig_stats_images_energy_hist.eps'],'file'),
        figure(1),set(gcf, 'visible', 'off'),
        plot(xout,Energy_hist/sum(Energy_hist))
        axis tight,
        grid on,xlabel('energy'),ylabel('Probability'),
        fig2pdf([e.where '/fig_stats_images_energy_hist'],10,10)
        
    end
    
    if ~exist([e.where '/fig_stats_images_cov.png'],'file'),
        imagesc(tile(Xcov)), colormap default
        axis off,title('Covariance for all pixels')
        imwrite((tile(Xcov)+1)/2,[e.where '/fig_stats_images_cov.png'])
    end
    
end;
