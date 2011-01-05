function add_log(e,string)
% writes all output to a log file. the philosophy is to use
% matlab's GUI (yuk) as less as possible (especially for non-windows users)

%## Author : Laurent Perrinet <Laurent.Perrinet@incm.cnrs-mrs.fr>
%## This software is distributed under the terms of the GPL

disp(string)

name_log=[e.where '/log.txt'];
fid=fopen(name_log,'a+');
fprintf(fid,'%s\t',e.version);
fprintf(fid,'%s\t',datestr(now));
%fprintf(fid,' in %s\t',e.where);
fprintf(fid,[string ' \n']);
fclose(fid);
