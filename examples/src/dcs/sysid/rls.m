pkg unload control;
addpath(genpath('~/Sys/opt/matlab/ident'));
addpath(genpath('~/Sys/opt/matlab/control'));

randn("seed",5489)
u = sign(randn(50,2)); % input
e = 0.2*randn(50,1);   % noise
%th0 = idpoly([1 -1.5 0.7],[0 1 0.5],[1 -1 0.2]); % a low order idpoly model
th0 = idpoly([1 -1.5 0.7],[0 1 0.5]); % a low order idpoly model
y = sim(th0,[u e]);
z = iddata(y,u);
plot(z) % analysis data object

[th,yh,p,phi] = rarx(z(1,:),[2 2 1],'ff',0.98);
plot(1,th(1),'*',1,th(2),'+',1,th(3),'o',1,th(4),'*'),
axis([1 50 -2 2]),title('Estimated Parameters'),drawnow
hold on;
for kkk = 2:50
	[th,yh,p,phi] = rarx(z(kkk,:),[2 2 1],'ff',0.98,th',p,phi);
	plot(kkk,th(1),'*',kkk,th(2),'+',kkk,th(3),'o',kkk,th(4),'*')
end
hold off


