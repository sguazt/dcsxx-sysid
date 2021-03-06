
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

E = [3.08172011e-01, -2.45693088e-01;
	 2.85335326e-01, -3.08004951e-01;
	-1.95837736e-02,  3.53297234e-01;
	 3.70122695e-01, -2.16048598e-01;
	-9.97247159e-02, -1.24976540e-01;
	 1.51356721e-01, -1.32192925e-02;
	 1.27577162e-01,  1.74148053e-02;
	 3.43638492e-01, -3.24963713e-01;
	-1.51691115e-01,  1.19461012e-01;
	-1.08410776e-01, -2.90062964e-02;
	-3.28313303e-01, -6.14573765e-01;
	-4.85119390e-01,  3.77101946e-01;
	-1.35164607e-01, -1.34291744e-01;
	 1.32145774e-01, -2.54598641e-01;
	-5.36407195e-03,  9.46612239e-02;
	 1.19110858e-01,  1.70817089e-01;
	-1.63276923e-01,  1.21445715e-01;
	 2.15859342e-01,  3.47538376e-01;
	-1.16053388e-02, -1.63092351e-01;
	 1.51414967e-01,  8.45085859e-02;
	 1.95624602e-01, -1.37456787e-01;
	-9.67659712e-02,  4.04050559e-02;
	-1.47895670e-01, -1.21097851e-01;
	-8.54656473e-03,  2.07195950e-01;
	 1.89103320e-02,  8.60167891e-03];


Y=[ -3.45913994e-01  1.59413142e-02;
	-1.43300971e-01  4.59962449e-01;
	 1.54498732e+00  2.50084935e+00;
	-6.53928706e+00 -3.17576279e+00;
	 1.46862817e+01  1.33995987e+01;
	-3.92207194e+01 -2.80764173e+01;
	 9.39887947e+01  7.38579240e+01;
	-2.33895468e+02 -1.76553533e+02;
	 5.59856224e+02  4.28366755e+02;
	-1.36104369e+03 -1.03624595e+03;
	 3.29161246e+03  2.50884712e+03;
	-7.96569571e+03 -6.07204769e+03;
	 1.92784928e+04  1.46946331e+04;
	-4.66592139e+04 -3.55655468e+04;
	 1.12922429e+05  8.60727815e+04;
	-2.73289602e+05 -2.08308551e+05;
	 6.61402124e+05  5.04141156e+05;
	-1.60069276e+06 -1.22009332e+06;
	 3.87391364e+06  2.95281082e+06;
	-9.37544916e+06 -7.14623820e+06;
	 2.26899742e+07  1.72949585e+07;
	-5.49130987e+07 -4.18563557e+07;
	 1.32897822e+08  1.01298579e+08;
	-3.21632398e+08 -2.45157549e+08;
	 7.78397998e+08  5.93317552e+08];

na=2;
nb=2;
nk=0;
ny=2;
nu=2;
nobs=rows(U);
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

	%disp(['Observation #: ', num2str(k)]);
	%disp(['U: ', mat2str(U(k,:))]);
	%disp(['Y: ', mat2str(Y(k,:))]);
	%disp(['Theta_hat: ', mat2str(Theta_hat)]);
	%disp(['P: ', mat2str(P)]);
	%disp(['phi: ', mat2str(phi)]);
	%disp(['Y_hat: ', mat2str(Y_hat(k,:))]);
	%disp(['--------------------------------: ', num2str(k)]);
end

