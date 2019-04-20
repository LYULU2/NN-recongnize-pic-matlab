% Initialization
clear ; close all; clc

% Load all the pic name path in fileparh
fprintf('Loading Data ...\n')
folder='/Users/luerlyu/Downloads/animals/';
filepaths=dir(fullfile(folder,'*.jpg')); % put the path names in filepaths

%initiliaze the X set and y set
X=zeros(40000,3000);

y=zeros(1,3000);

for i=1: length(filepaths)
    image=imread(fullfile(folder,filepaths(i).name)); %read in image
    image=rgb2gray(image); %convert it to gray
    
    %flaten the data
    image=imresize(image,[200,200]); %resize the pic
    X(:,i)=reshape(image,[40000,1]); % convert to 1D array
    
    
    if(~isempty(strfind(filepaths(i).name,'cats')))
        y(1,i)=1;
    elseif(~isempty(strfind(filepaths(i).name,'dogs')))
        y(1,i)=2;
    elseif(~isempty(strfind(filepaths(i).name,'panda')))
        y(1,i)=3;
    end
end

%X=X/255;

%% using PCA to process data
% decentralize 
X=X-repmat(mean(X,2),1,3000);
X=X';
%[coeff,score,latent]=pca(X);

% Calculate the eigenvalues and eigenvectors
covarianceMatrix = X*X'/(size(X,2)-1);
[V, D] = eig(covarianceMatrix);

% Sort the eigenvalues
[d_out,order] = sort(diag(D),'descend');
V = V(:,order);

%select the first 400 vectors
V_proj = V(:,1:400);

%compute the projection of the original data
X = X'*V_proj;

%% export X y
X=X';
y=full(ind2vec(y));
save('data.mat','X','y');
