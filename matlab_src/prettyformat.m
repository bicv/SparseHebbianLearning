function prettyformat(e)

% pretty format struc e to LaTeX format
% oputput compatible with the tabular env in LaTeX (see
% ../code/results.tex)

name_log=[e.where '/default.txt'];
fid=fopen(name_log,'w');
list = sort(fieldnames(e));
for i_item=1:size(list,1),    
    fprintf(fid,'\\verb+%s+ & ',list{i_item});
    fprintf(fid,' \\verb+%s+ \\\\ \n',num2str(getfield(e,list{i_item}))); 
end
fclose(fid);
