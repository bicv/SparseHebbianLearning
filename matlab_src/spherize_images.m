% spherize_images
clear all
% takes a bunch of images (8-bit PNG format) named 1.png etc...
% it then decorrelates the image and put them in a huge matrix IMAGES.mat

% one problem is that we take only patches of limited sizes, at only one
% scale. A solution is therefore to take only a narrow band and remove
% information from lower spatial frequencies (this is just approached by
% the decorrelation). The experiment experiment_stats_images shows then
% the shape of the covariance matrix which sould be isotropic for all
% pixels.


unix('ls /ih/lup/Documents/Mes_images/photos_2004/2005-04-10_Yelmo/*.jpg > image_list.txt')
%% reads data
string= textread('image_list.txt','%s');

N=size(string,1);
I=double(imread([string{1}]));
[image_size_h,image_size_v,dump]=size(I);
down =2; % downsampling
image_size_h=image_size_h/2;image_size_v=image_size_v/down;
image_size=image_size_h;
IMAGES=zeros(image_size_h^2,N);

% the filtering done for decorrelating images in Olshausen and following
[fx fy]=meshgrid(-image_size/2:image_size/2-1,-image_size/2:image_size/2-1);
rho=sqrt(fx.*fx+fy.*fy);
f_0=0.4*image_size_h;
filt=rho.*exp(-(rho/f_0).^4);

for i_im=1:N,
    disp(i_im)
    %loads image
    I=double(imread([string{i_im}]));
    [image_size_h,image_size_v,dump]=size(I);
    image_size_h=image_size_h/2;image_size_v=image_size_v/down;
    I=sum(I,3); % color -> grayscale
    I=conv2(I,ones(down),'same');
    I=I(1:down:down*image_size_h,1:down:down*image_size_v);

    % extracts central part
    offset=ceil((size(I,2)-size(I,1))/2);
    if offset>0,
        I=I(:,offset+1:offset+size(I,1));
    else
        I=I(-offset+1:-offset+size(I,2),:);
    end

    %%%%%%%%%%%%%%%
    % Decorrelation
    if 1,
        If=fft2(I);
        I_=real(ifft2(If.*fftshift(filt)));
        imagesc(I_), colormap gray, axis equal off, drawnow,% pause
    else
        %load blanc % decorrelation kernel 
        K=fspecial('log',[7 7],1.5);
        this_image=conv2(I,K,'valid');
    end
    IMAGES(:,i_im)=reshape(I_,image_size^2,1);
end

IMAGES=sqrt(0.1)*IMAGES/sqrt(mean(var(IMAGES)));
save('../data/IMAGES_yelmo','IMAGES')
