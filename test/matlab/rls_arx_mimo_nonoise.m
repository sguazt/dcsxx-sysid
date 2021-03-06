
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

Y=[ -5.00000000e-01  2.00000000e-01;
	-5.00000000e-01  8.50000000e-01;
	 1.11000000e+00  2.49000000e+00;
	-6.31000000e+00 -2.37750000e+00;
	 1.29560000e+01  1.28237500e+01;
	-3.65970000e+01 -2.52615000e+01;
	 8.59596500e+01  6.85872625e+01;
	-2.16322175e+02 -1.61890181e+02;
	 5.14889120e+02  3.95251008e+02;
	-1.25448088e+03 -9.53798566e+02;
	 3.03143890e+03  2.31236360e+03;
	-7.33945830e+03 -5.59327618e+03;
	 1.77600940e+04  1.35388854e+04;
	-4.29877638e+04 -3.27650970e+04;
	 1.04033071e+05  7.92989952e+04;
	-2.51779966e+05 -1.91911412e+05;
	 6.09341983e+05  4.64461202e+05;
	-1.47470321e+06 -1.12405886e+06;
	 3.56899718e+06  2.72039643e+06;
	-8.63750985e+06 -6.58375763e+06;
	 2.09040482e+07  1.59336759e+07;
	-5.05908975e+07 -3.85618464e+07;
	 1.22437451e+08  9.33253832e+07;
	-2.96316758e+08 -2.25861231e+08;
	 7.17130398e+08  5.46617611e+08];

na=2;
nb=2;
nk=0;
ny=2;
nu=2;
nobs=size(U,1);
ff = 0.98;

%n1 = na*ny+(nb+1)*nu;
%n2 = na*ny+(nb+1)*nu+nk;
n = na*ny+(nb+1)*nu+nk;
Theta_hat = eps*ones(ny,n,1);
I = eye(n);
P=10000*I;
phi=zeros(n,1);
Y_hat = zeros(nobs,2);
L = zeros(n,1);

for k=1:nobs
	L = (P*phi)/(ff+phi'*P*phi);
	P = (I-L*phi')*P/ff;
	Y_hat(k,:) = [Theta_hat*phi]';
	Theta_hat = Theta_hat+(Y(k,:)'-Y_hat(k,:)')*L';
	phi = [-Y(k,:) phi(1:(na-1)*ny)' U(k,:) phi((na*ny+1):(na*ny+nb*nu))']';

	disp(['Observation #: ', num2str(k)]);
	disp(['U: ', mat2str(U(k,:))]);
	disp(['Y: ', mat2str(Y(k,:))]);
	disp(['Theta_hat: ', mat2str(Theta_hat)]);
	disp(['P: ', mat2str(P)]);
	disp(['phi: ', mat2str(phi)]);
	disp(['Y_hat: ', mat2str(Y_hat(k,:))]);
	disp(['--------------------------------: ', num2str(k)]);
end

