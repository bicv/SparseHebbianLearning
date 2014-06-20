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


load('../data/IMAGES_icabench');
IMAGES_=IMAGES;
[image_size_h,image_size_v,N]=size(IMAGES_);
image_size=image_size_h;
IMAGES=zeros(image_size,image_size,N);

% the filtering done for decorrelating images in Olshausen and following
[fx fy]=meshgrid(-image_size/2:image_size/2-1,-image_size/2:image_size/2-1);
rho=sqrt(0.01^2+fx.^2+fy.^2);
f_0=0.5*image_size_h;
filt=rho.*exp(-(rho/f_0).^1.4);

for i_im=1:N,
    disp(i_im)
    %loads image
    I_=IMAGES_(:,:,i_im);

    % extracts central part
    offset=ceil((image_size_v-image_size_h)/2);
    if offset>0,
        I_=I_(:,offset+1:offset+image_size_h);
    else
        I_=I_(-offset+1:-offset+image_size_v,:);
    end

    %%%%%%%%%%%%%%%
    % Decorrelation
    If=fft2(I_);
    IMAGES(:,:,i_im)=real(ifft2(If.*fftshift(filt)));
    imagesc(IMAGES(:,:,i_im)), colormap gray, axis equal off, drawnow, pause
end

IMAGES=sqrt(0.1)*IMAGES/sqrt(mean(var(IMAGES(:))));
save('../data/IMAGES_icabench_decorr','IMAGES')
