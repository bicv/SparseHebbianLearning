function display_network(n,e,norm)
%
%  display_network -- displays the state of the network (weights and
%                     output variances)
%
%  Usage:
%
%    h=display_network(A,S_var,h);
%
%    A = basis function matrix
%    S_var = vector of coefficient variances
%    h = display handle (optional)

% this script is modified from B. Olshausen's Sparsenet :
% see http://redwood.ucdavis.edu/bruno/sparsenet.html
%
% you may have to change some parameters to match your screensize

%## Author : Laurent Perrinet <Laurent.Perrinet@incm.cnrs-mrs.fr>
%## This software is distributed under the terms of the GPL


% basis functions
figure(1)
set(gcf,'MenuBar','none','ToolBar','none')

if nargin < 3, norm = 1; end

array=tile(n.A,norm);%

imagesc(array,[-1 1])%
axis image off

ax = gca ();
set (ax, 'position', [0., 0., 1., 1.]);
colormap gray
