pkg unload control;
addpath(genpath('~/Sys/opt/matlab/ident'));
addpath(genpath('~/Sys/opt/matlab/control'));

randn("seed",5489)

% SISO ARX model
A=zeros(1,1,2);
B=zeros(1,1,2);
A(:,:,1) = eye(1);
A(:,:,2) = [-1.5];
B(:,:,2) = [1];
m0 = idarx(A, B, 1);
u = iddata([],idinput(300));
e = iddata([],randn(300,1));
%y = sim(m0,u);
y = sim(m0,[u e]);
