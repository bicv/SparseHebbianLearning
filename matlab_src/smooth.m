function v_=smooth(v,n_mean)

% function to smooth the statistics and down_sample
%
% tipically, this leads to a problem at borders... so we do the mean solely on the valid part of the vector

%## Author : Laurent Perrinet <Laurent.Perrinet@incm.cnrs-mrs.fr>
%## This software is distributed under the terms of the GPL


if n_mean==1,
    v_=v(2:length(v));
else
    for i_v=1:length(v),%-1,
        v_(i_v)=mean( v(max(1,i_v-n_mean):min(length(v),i_v+n_mean-1)) );
    end
    % downsample
    v_=v_(n_mean-1:n_mean:length(v_));
end