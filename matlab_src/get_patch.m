function X=get_patch(e,norm)
% get_patch : gets a batch of imagelets into stripes...
% inputs : - e the experiments's parameter
%          - IMAGES the image matrix (n_x,n_y,n_im)
% output : - X= a matrix of e.batch_size vectorized images

% to test :
%  e.L=12^2; e.batch_size=20^2; imagesc(tile(get_patch(e))) , colormap gray


%## Author : Laurent Perrinet <Laurent.Perrinet@incm.cnrs-mrs.fr>
%## This software is distributed under the terms of the GPL


if nargin < 2, norm = 0; end % by default, do not normalize patches

load(e.image_base)

num_images=size(IMAGES,3); % num_images of square images
image_size=size(IMAGES,1); % of size image_size
image_size_v=size(IMAGES,2); %for non square images??

sz=sqrt(e.L); %sM=sqrt(e.M); % so you've been warned!
BUFF=ceil(sz/2)+1; % to avoid taking *out* of the border
X=zeros(sz*sz,e.batch_size); % intitialize array

for i_batch=1:(e.batch_size),
    
    i_im=ceil(rand(1)*num_images);
    r=BUFF+ceil((image_size-sz-2*BUFF)*rand(1));
    c=BUFF+ceil((image_size-sz-2*BUFF)*rand(1));
    dump=IMAGES(r:r+sz-1,c:c+sz-1,i_im);
    dump_=reshape(dump,sz^2,1);
    X(:,i_batch)=dump_-sum(dump_)/(sz*sz);
    if norm == 1,
        X(:,i_batch)=X(:,i_batch)/sqrt(sum(X(:,i_batch).^2)/e.L);
    end
end

