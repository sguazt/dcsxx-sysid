%pkg unload control;
%addpath(genpath('~/Sys/opt/matlab/ident'));
%addpath(genpath('~/Sys/opt/matlab/control'));

%randn("seed",5489)
%nobs = 50;
%u = sign(randn(nobs,1)); % input
u = [1; 1; -1; 1; 1; -1; -1; 1; 1; 1; -1; 1; 1; -1; 1; -1; -1; 1; 1; 1; 1; -1; 1; 1; 1; 1; 1; -1; 1; -1; 1; -1; -1; -1; -1; 1; 1; -1; 1; -1; -1; -1; 1; 1; -1; -1; -1; 1; 1; 1]; % input
nobs = length(u);
th0 = idpoly([1 -1.5 0.7],[0.2 1 0.5]); % a low order idpoly model
y = sim(th0,u);
z = iddata(y,u);
plot(z) % analysis data object

ff = 0.98; % forgetting factor
nn = [2 3 1]; % model order
yh=[];
[th,yh(1),p,phi] = rarx(z(1,:),nn,'ff',ff);
plot(1,th(1),'*',1,th(2),'+',1,th(3),'o',1,th(4),'*'),
axis([1 nobs -2 2]),title('Estimated Parameters'),drawnow
hold on;
for kkk = 2:nobs
    [th,yh(kkk),p,phi] = rarx(z(kkk,:),nn,'ff',ff,th',p,phi);
    plot(kkk,th(1),'*',kkk,th(2),'+',kkk,th(3),'o',kkk,th(4),'*')
end
hold off
