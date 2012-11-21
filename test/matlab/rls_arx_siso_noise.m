pkg unload control;
addpath(genpath('~/Sys/opt/matlab/ident'));
addpath(genpath('~/Sys/opt/matlab/control'));

%randn("seed",5489)
%nobs = 50;
%u = sign(randn(nobs,1)); % input
%e = 0.2*randn(nobs,1);   % noise
u = [ 1;  1; -1;  1;  1; -1; -1;  1; 1;  1;
	 -1;  1;  1; -1;  1; -1; -1;  1; 1;  1;
	  1; -1;  1;  1;  1;  1;  1; -1; 1; -1;
	  1; -1; -1; -1; -1;  1;  1; -1; 1; -1;
	 -1; -1;  1;  1; -1; -1; -1;  1; 1;  1]; % input
e = [-0.172730564397743; 0.015471818226085; -0.242823408723082; -0.222700148297353; -0.001369865620670; 0.306526061656950; -0.153933182750736; 0.074275762552012; -0.045116880454250; 0.223471227762893; -0.217812859010447; 0.006511492832995; 0.110505404222445; 0.220122043576173; 0.308842379100790; 0.017186226635085; -0.298318062127522; -0.148460367451971; -0.212316346663997; 0.470091444800408; -0.123120376293379; 0.149615356740797; -0.038483702117653; 0.177722085084144; -0.152969847313575; -0.280453793867752; -0.284475185018299; 0.097638781971988; -0.035475031323765; -0.039210697561467; 0.283862030128510; 0.058316874796837; 0.039562210692872; 0.317539817994812; -0.160893191269909; 0.139324883169921; 0.167017633014536; -0.048743028075590; 0.043134017280749; -0.233168786296410; -0.229590555779719; 0.020974943203299; 0.144450806445000; 0.517098250523248; -0.133378134140277; 0.037466204915788; -0.016498885074191; -0.386604583570197; -0.087793230786955; -0.358935768291025]; % noise
th0 = idpoly([1 -1.5 0.7],[0.2 1 0.5],[1 -1 0.2]); % a low order idpoly model
y = sim(th0,[u e]);
%y = [0.027269435602257;1.42910653602721;3.13172985929052;3.42043783778768;3.81121145611023;4.88586659533211;3.50021865618919;0.81973552518544;-0.670729051110678;0.40353481643172;1.92450509756086;2.57330187223511;3.27323057964743;4.51945349682689;4.29874025585624;3.33686119225506;1.04243779634078;-1.9188512000648;-3.03150284917328;-0.851342715740552;1.90936283038388;5.12673816858847;5.84083013749686;6.11865734678332;6.45101625110592;7.00152470437322;7.65196032017242;8.20289639541411;6.45796351870106;4.26070989141841;1.8864680954621;-0.0861820756437078;-2.11178303836729;-4.51770612233904;-7.4688316237776;-8.97712711199028;-7.54199441975046;-3.91689833568865;-0.770670837748398;1.59977116916718;2.25133137414749;0.761088984527941;-1.65674073302557;-1.94103095598901;-1.07341414424014;-0.677135558007186;-2.04495415286072;-4.25604879619889;-3.95709371152744;-1.60486986417004]
z = iddata(y,u);
plot(z) % analysis data object

yh=[];
[th,yh(1),p,phi] = rarx(z(1,:),[2 3 1],'ff',0.98);
plot(1,th(1),'*',1,th(2),'+',1,th(3),'o',1,th(4),'*'),
axis([1 nobs -2 2]),title('Estimated Parameters'),drawnow
hold on;
for kkk = 2:nobs
    [th,yh(kkk),p,phi] = rarx(z(kkk,:),[2 3 1],'ff',0.98,th',p,phi);
    plot(kkk,th(1),'*',kkk,th(2),'+',kkk,th(3),'o',kkk,th(4),'*')
end
hold off