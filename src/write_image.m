function write_image(name, data)
% this function is obosolete with the new version of Octave > 3.0

%## Author : Laurent Perrinet <Laurent.Perrinet@incm.cnrs-mrs.fr>
%## This software is distributed under the terms of the GPL


% This is a dirty hack for some older versions of octave that don't have a imwrite function 
if strcmp(version('-release'),'2007b') | strcmp(version,'3.2.3'), % matlab or latest octave
    imwrite(data,name)
else,i % use the one from octave-forge, but parameters order's inverted :-(
    imwrite(name,data)
end
