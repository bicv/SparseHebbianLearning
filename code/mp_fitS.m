function [S, dA, Pz_j]=mp_fitS(A,X,threshold,frac,switch_choice,Mod,switch_Mod,lut,switch_sym)%
%% mp_fitS -- fit internal vars S to the data X using matching pursuit
% Usage :
%     [S, Pz_j]=mp_fitS(A,X,threshold,frac,Mod,lut,switch_sym)
%
%   Inputs ===============================================================
%      A             basis functions (size: L dimensions x M atoms)
%      X             data vector (size: L dimensions x n_batch)
%      threshold     the threshold for which the MP is stopped (in percent
%      of the energy per pixel) corresponding to the estimate of the energy
%      of background noise (in default.m, it is noise_var_ssc).
if nargin < 3, threshold = 0; end
%      frac         the maximal fraction of coefficients used (normalized
%      by the number of filters)
if nargin < 4, frac = 1; end
%       switch_choice     tunes the choice of the most probable component
%       according to previous statistics of cosinus (spike gain control):
if nargin < 5, switch_choice = 0;  end % normal matching pursuit
%       Mod(:,j) corresponds to the value of |S_j| knowing the rank
if nargin < 6 ,
    Mod = 1;
end
%       switch_Mod  do we use quantization by Mod?
if nargin < 7,  switch_Mod = 0; end % no by default
if nargin < 8,  lut = 1; end % no lut by default
%       switch_sym  do we use the ON/OFF symmetry?
if nargin < 9,  switch_sym = 1; end % yes by default

% retrieves all dimensions from the input
L = size(X,1); M = size(A,2);
n_batch = size(X,2); n_quant = size(Mod,1);
max_iter = ceil(M*frac); % max number of MP iterations : a fraction of the
% overcomplete dimension

%   Outputs =============================================================
%      S           the estimated coefficients (size: M atoms x n_batch col)
%      Pz_j    the empirical probability for this batch (size: n_quant x M)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%## Author : Laurent Perrinet <Laurent.Perrinet@incm.cnrs-mrs.fr>
%## This software is distributed under the terms of the GPL

switch_learn = 0;
if nargout > 1, % for the learning
    switch_learn = 1;
    %      Pjr corresponds to the probability distribution of j knowing the rank
    Pz_j = zeros(n_quant,M);
    dA = zeros(L,M);
end
if switch_Mod,
    mMod = mean(Mod,2);
end

S=zeros(M,n_batch); % initialized at zero
RK=A'*A; % correlation matrix
%%%%%%%%%%% matching pursuit %%%%%%%%%%%%%%%
% initial activity  < X, A/|A|> of data over the basis functions
C=A'*X; % (normA is 1)
% X_energy=sum(X(:).^2)/n_batch/L; % mean energy per pixel of the whole batch

for i_batch=1:n_batch,
    %looping over images in the batch
    i_iter=1;
    %  E     =   residual image
    E=X(:,i_batch); % initialization
    E_energy=sum(E.^2)/L; % initialize energy per pixel of this residual image
    E_energy_0=sum(E.^2)/L; % initialize energy per pixel of this residual image

    % cosinus = abs(C(:,i_batch)) /sqrt(E_energy_0*L) ; % ArgMax abs(cos(E,phi_i))

    while (E_energy>threshold) && (i_iter<=max_iter), %(sum(S(:,i_batch)~=0)<=max_iter)&& 
        % MATCHING
        if switch_sym,
            % We normalize by energy to have a real cosinus. has no effect
            % on the choice of the best match.
            % Take either the biggest absolute value (so may be largest
            % negative correlation)
             cosinus = 1 - exp(- abs(C(:,i_batch)) ); % ; % ArgMax abs(cos(E,phi_i))
        else,
            % or simply the biggest value (it's a rectification, since
            %negative correlations can not be chosen and the alg. would be
            %the same by zeroing out negative values)
             cosinus = 1 - exp(- C(:,i_batch).*(C(:,i_batch)>0) ); % ; % ArgMax abs(cos(E,phi_i))
        end
        
        if ~switch_choice,
            % ArgMax C : simplest case : just chosing the one with maximal correlation
            [dump i_win]=max( cosinus  );
        else
            % exact : choose the one corresponding to ArgMax (f_i (C) )
            index = floor(cosinus' * n_quant) + 1 + (0:n_quant:((M-1)*n_quant)) ;
            z = Mod(index); % we use a trick here of matlab array representation as vectors
            [dump i_win]=max( z .* (S(:,i_batch)==0)' ); % TODO: what in case of ex aequo? add perturbation?
        end

        % PURSUIT
        if switch_Mod, % quantization using Mod
            % we use the LUT
            proj = sign(C(i_win,i_batch)) * lut(i_iter);% * sqrt(E_energy_0*L);%
        else,
            proj = C(i_win,i_batch);
        end
        
        E=E-proj*A(:,i_win); % new residual image % MP : substracts the projection also from the input
        C(:,i_batch) = C(:,i_batch) - proj*RK(:,i_win); %updates activities according to pursuit
        S(i_win,i_batch) = S(i_win,i_batch) + proj; % add in case a basis function is selected more than once
        E_energy=sum(E.^2)/L;

        if switch_learn, 
            % computes the gradient \rho_j* ( X - s_{j^*} A_{j^*})
            dA(:,i_win) = dA(:,i_win) + proj * E ;
        end
        
        i_iter=i_iter+1;
    end;% coding done :)
end %  batch done

if switch_learn,%
    rank = 1/2/n_quant:1/(n_quant):1-1/2/n_quant; % define the bins' centers
    %  probability defined versus coefficients which are in [0, 1]
    for i_M=1:M,
        Pz_j(:,i_M)=hist(1 - exp(-abs(S(i_M,:))),rank)/n_batch;
    end    
end
