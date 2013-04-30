function array=tile(A,norm)
% Converts the matrix of filters into a displayable image matrix
% returns values between -1 and 1

% the norm switch gives a normalisation per filter if 1 (default) or for
% all filters if 0
if nargin < 2, norm = 1; end


[L M]=size(A);
sz=sqrt(L);
buf=1;
if floor(sqrt(M))^2 ~= M
    n=sqrt(M/2);
    m=M/n;
else
    n=sqrt(M);
    m=n;
end

% black = -1 (that is -max(|A_i|))... 

array=-ones(buf+m*(sz+buf),buf+n*(sz+buf));

if ~norm,
            clim=max(abs(A(:)));
end

k=1;

for i=1:m
    for j=1:n
        if norm,
            clim=max(abs(A(:,k)));
        end
        
        array(buf+(j-1)*(sz+buf)+[1:sz],buf+(i-1)*(sz+buf)+[1:sz])=...
            reshape(A(:,k),sz,sz)/clim;
        
        k=k+1;
    end
end
