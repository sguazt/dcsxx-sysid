
U=[ -1.0 -1.0;
    -1.0 -1.0;
     1.0 -1.0;
    -1.0 -1.0;
     1.0 -1.0;
     1.0  1.0;
     1.0 -1.0;
    -1.0  1.0;
    -1.0  1.0;
    -1.0  1.0;
     1.0  1.0;
     1.0  1.0;
    -1.0 -1.0;
    -1.0  1.0;
    -1.0 -1.0;
    -1.0 -1.0;
     1.0 -1.0;
     1.0  1.0;
     1.0 -1.0;
    -1.0  1.0;
     1.0 -1.0;
    -1.0 -1.0;
     1.0  1.0;
     1.0  1.0;
    -1.0 -1.0];

y=[ -5.00000000e-01;
	-5.00000000e-01;
	 1.11000000e+00;
	-6.31000000e+00;
	 1.29560000e+01;
	-3.65970000e+01;
	 8.59596500e+01;
	-2.16322175e+02;
	 5.14889120e+02;
	-1.25448088e+03;
	 3.03143890e+03;
	-7.33945830e+03;
	 1.77600940e+04;
	-4.29877638e+04;
	 1.04033071e+05;
	-2.51779966e+05;
	 6.09341983e+05;
	-1.47470321e+06;
	 3.56899718e+06;
	-8.63750985e+06;
	 2.09040482e+07;
	-5.05908975e+07;
	 1.22437451e+08;
	-2.96316758e+08;
	 7.17130398e+08];

na=2;
nb=2;
nk=1;
ny=1;
nu=2;
nobs=size(U,1);
ff = 0.98;

y_hat = [];
z = iddata(y, U);
[Theta_hat,y_hat(1),P,phi] = rarx(z(1,:),[na [nb nb] [nk nk]],'ff',ff);
for k = 2:nobs
	[Theta_hat,y_hat(k),P,phi] = rarx(z(k,:),[na [nb nb] [nk nk]],'ff',ff,Theta_hat',P,phi);

	disp(['Observation #: ', num2str(k)]);
	disp(['U: ', mat2str(U(k,:))]);
	disp(['y: ', num2str(y(k))]);
	disp(['Theta_hat: ', mat2str(Theta_hat)]);
	disp(['P: ', mat2str(P)]);
	disp(['phi: ', mat2str(phi)]);
	disp(['y_hat: ', mat2str(y_hat)]);
	disp(['r.e.: ', mat2str(abs(y_hat/y(k)-1))]);
	disp(['--------------------------------: ', num2str(k)]);
end
y_hat = y_hat';


%n = na*ny+(nb+1)*nu+nk;
%Theta_hat = eps*ones(ny,n,1);
%I = eye(n);
%P=10000*I;
%phi=zeros(n,1);
%Y_hat = zeros(nobs,2);
%L = zeros(n,1);
%
%for k=1:nobs
%	L = (P*phi)/(ff+phi'*P*phi);
%	P = (I-L*phi')*P/ff;
%	Y_hat(k,:) = [Theta_hat*phi]';
%	Theta_hat = Theta_hat+(Y(k,:)'-Y_hat(k,:)')*L';
%	phi = [-Y(k,:) phi(1:(na-1)*ny)' U(k,:) phi((na*ny+1):(na*ny+nb*nu))']';
%
%	disp(['Observation #: ', num2str(k)]);
%	disp(['U: ', mat2str(U(k,:))]);
%	disp(['Y: ', mat2str(Y(k,:))]);
%	disp(['Theta_hat: ', mat2str(Theta_hat)]);
%	disp(['P: ', mat2str(P)]);
%	disp(['phi: ', mat2str(phi)]);
%	disp(['Y_hat: ', mat2str(Y_hat(k,:))]);
%	disp(['--------------------------------: ', num2str(k)]);
%end

