function fig2pdf(nom,height,width)

%## Author : Laurent Perrinet <Laurent.Perrinet@incm.cnrs-mrs.fr>
%## This software is distributed under the terms of the GPL


if strcmp(version,'3.2.4')
 disp('running octave')
else,% generate PDF figures
set(gcf,'PaperUnits','centimeters')%
opts = struct('height',height,'width',width,'FontMode','fixed','FontSize',10,'Color' ,'cmyk','Bounds','tight','LockAxes',0);%,'SeparateText',1)%
exportfig(gcf, [nom '.eps'], opts);
% http://www.mathworks.com/matlabcentral/fileexchange/loadFile.do?objectId=7401&objectType=FILE
% plot2svg([nom '.svg'],gcf)

% I hope you have a good TeX distribution with the eps2pdf perl script!
% 
% see http://www.wlug.org.nz/epstopdf(1) and LiveTeX, MikTeX, etc...
cmd=['epstopdf --nocompress ' nom '.eps --outfile=' nom '.pdf']
unix(cmd);
end
