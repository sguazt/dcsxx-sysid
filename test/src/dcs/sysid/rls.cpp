#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <cstddef>
#include <dcs/debug.hpp>
#include <dcs/test.hpp>
//#include <dcs/math/la/container/matrix.hpp>
//#include <dcs/math/la/container/vector.hpp>
//#include <dcs/math/la/container/identity_matrix.hpp>
//#include <dcs/math/la/operation/io.hpp>
//#include <dcs/math/la/operation/matrix_basic_operations.hpp>
//#include <dcs/math/la/operation/row.hpp>
//#include <dcs/math/la/operation/vector_basic_operations.hpp>
#include <dcs/sysid/algorithm/rls.hpp>
#include <dcs/sysid/model/darx_siso.hpp>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <sstream>


namespace ublas = ::boost::numeric::ublas;


static const double tol = 1.0e-5;


DCS_TEST_DEF( rarx_siso_without_noise )
{
	DCS_DEBUG_TRACE("Test Case: SISO system with ARX structure and without noise");

	typedef double real_type;
	typedef ::std::size_t size_type;
	typedef ::std::size_t uint_type;
	typedef ublas::vector<real_type> vector_type;;
	typedef ublas::matrix<real_type> matrix_type;;

	const size_type n_obs(50);
	const size_type n_a(2);
	const size_type n_b(2);
	const size_type d(1);
	const real_type ff(0.98);

	vector_type u(n_obs);
	u( 0) = -1.0;
	u( 1) = -1.0;
	u( 2) = -1.0;
	u( 3) = -1.0;
	u( 4) =  1.0;
	u( 5) = -1.0;
	u( 6) = -1.0;
	u( 7) = -1.0;
	u( 8) =  1.0;
	u( 9) = -1.0;
	u(10) =  1.0;
	u(11) =  1.0;
	u(12) =  1.0;
	u(13) = -1.0;
	u(14) = -1.0;
	u(15) =  1.0;
	u(16) = -1.0;
	u(17) =  1.0;
	u(18) = -1.0;
	u(19) =  1.0;
	u(20) =  1.0;
	u(21) =  1.0;
	u(22) =  1.0;
	u(23) =  1.0;
	u(24) = -1.0;
	u(25) = -1.0;
	u(26) = -1.0;
	u(27) =  1.0;
	u(28) = -1.0;
	u(29) = -1.0;
	u(30) = -1.0;
	u(31) = -1.0;
	u(32) =  1.0;
	u(33) = -1.0;
	u(34) =  1.0;
	u(35) =  1.0;
	u(36) =  1.0;
	u(37) = -1.0;
	u(38) = -1.0;
	u(39) =  1.0;
	u(40) =  1.0;
	u(41) = -1.0;
	u(42) = -1.0;
	u(43) = -1.0;
	u(44) =  1.0;
	u(45) =  1.0;
	u(46) =  1.0;
	u(47) =  1.0;
	u(48) = -1.0;
	u(49) = -1.0;

	vector_type y(n_obs);
	y( 0) =  0.2;
	y( 1) =  1.5;
	y( 2) =  3.41;
	y( 3) =  3.765;
	y( 4) =  3.9605;
	y( 5) =  4.60525;
	y( 6) =  3.435525;
	y( 7) =  0.629612500000002;
	y( 8) = -0.760448749999998;
	y( 9) =  0.118598125000002;
	y(10) =  2.0102113125;
	y(11) =  2.63229828125;
	y(12) =  3.241299503125;
	y(13) =  4.3193404578125;
	y(14) =  3.91010103453125;
	y(15) =  3.14161323132813;
	y(16) =  1.27534912282031;
	y(17) = -1.58610557769922;
	y(18) = -2.57190275252305;
	y(19) = -1.04758022439512;
	y(20) =  1.92896159017346;
	y(21) =  4.92674854233677;
	y(22) =  5.73984970038373;
	y(23) =  5.86105057093986;
	y(24) =  6.47368106614118;
	y(25) =  7.30778619955387;
	y(26) =  8.13010255303197;
	y(27) =  8.37970348986026;
	y(28) =  6.578483447668;
	y(29) =  4.30193272859982;
	y(30) =  1.54796067953213;
	y(31) = -0.389411890721674;
	y(32) = -2.36769031175501;
	y(33) = -4.97894714412734;
	y(34) = -7.5110374979625;
	y(35) = -9.08129324605462;
	y(36) = -7.66421362050817;
	y(37) = -3.83941515852403;
	y(38) = -0.694173203430321;
	y(39) =  1.94633080582134;
	y(40) =  2.70541745113323;
	y(41) =  0.99569461262491;
	y(42) = -1.7002502968559;
	y(43) = -2.54736167412128;
	y(44) = -1.3308673033828;
	y(45) = -0.913147783189296;
	y(46) = -2.13811456241599;
	y(47) = -3.86796839539147;
	y(48) = -3.60527239939602;
	y(49) = -1.00033072232;

	vector_type y_hat_ok(n_obs);
	y_hat_ok( 0) =  0;
	y_hat_ok( 1) = -2.66453525910038e-16;
	y_hat_ok( 2) =  1.87482686694971;
	y_hat_ok( 3) =  5.48583439035849;
	y_hat_ok( 4) =  2.79210396420393;
	y_hat_ok( 5) =  1.06732152822507;
	y_hat_ok( 6) = -0.316102167275933;
	y_hat_ok( 7) =  4.18345416258441;
	y_hat_ok( 8) = -2.73658080208994;
	y_hat_ok( 9) = -1.00344753971502;
	y_hat_ok(10) = -0.268811733617172;
	y_hat_ok(11) =  3.46144226206394;
	y_hat_ok(12) =  1.72772780070264;
	y_hat_ok(13) =  2.90898384392673;
	y_hat_ok(14) =  4.24967287333039;
	y_hat_ok(15) =  2.96818552089304;
	y_hat_ok(16) =  2.55718091996513;
	y_hat_ok(17) = -0.388017828026363;
	y_hat_ok(18) = -3.32851229837221;
	y_hat_ok(19) = -3.18161787664256;
	y_hat_ok(20) =  0.360471248674249;
	y_hat_ok(21) =  3.71426871727841;
	y_hat_ok(22) =  6.34827393739279;
	y_hat_ok(23) =  4.93484259213945;
	y_hat_ok(24) =  4.75006866010663;
	y_hat_ok(25) =  5.51025451819693;
	y_hat_ok(26) =  6.69226409320216;
	y_hat_ok(27) =  7.85918710321379;
	y_hat_ok(28) =  8.04380370779958;
	y_hat_ok(29) =  4.39059230512023;
	y_hat_ok(30) =  1.99896833056786;
	y_hat_ok(31) = -0.881276000841646;
	y_hat_ok(32) = -1.8738830337286;
	y_hat_ok(33) = -3.58844429378258;
	y_hat_ok(34) = -6.47770176344944;
	y_hat_ok(35) = -8.86056382593082;;
	y_hat_ok(36) = -9.47344919342524;
	y_hat_ok(37) = -5.36441524153463;
	y_hat_ok(38) =  0.129373553206052;
	y_hat_ok(39) =  1.79846082706382;
	y_hat_ok(40) =  3.89841418907224;
	y_hat_ok(41) =  3.1532383400713;
	y_hat_ok(42) = -0.462036118076988;
	y_hat_ok(43) = -3.70430063658166;
	y_hat_ok(44) = -2.77959659915778;
	y_hat_ok(45) = -0.103990505692061;
	y_hat_ok(46) = -0.548342239077826;
	y_hat_ok(47) = -2.99255581772226;
	y_hat_ok(48) = -4.92251286502603;
	y_hat_ok(49) = -2.70680183780531;

	vector_type theta_hat;
	matrix_type P;
	vector_type phi;

	DCS_DEBUG_STREAM << ::std::setprecision(16);

	::dcs::sysid::rls_arx_siso_init(n_a, n_b, d, theta_hat, P, phi);

	DCS_DEBUG_TRACE_L( 5, "theta0_hat: " << theta_hat );
	DCS_DEBUG_TRACE_L( 5, "P0: " << P );
	DCS_DEBUG_TRACE_L( 5, "phi0: " << phi );

	vector_type y_hat(n_obs);
	for (size_type i = 0; i < n_obs; ++i)
	{
		real_type yy_hat;

		yy_hat = ::dcs::sysid::rls_ff_arx_siso(
			y(i),
			u(i),
			ff,
			n_a,
			n_b,
			d,
			theta_hat,
			P,
			phi
		);
		y_hat(i) = yy_hat;

		DCS_DEBUG_TRACE_L( 5, ">> Observation #" << i );
		DCS_DEBUG_TRACE_L( 5, ">>" << i << " --> u: " << u(i) );
		DCS_DEBUG_TRACE_L( 5, ">>" << i << " --> y: " << y(i) );
		DCS_DEBUG_TRACE_L( 5, ">>" << i << " --> theta_hat: " << theta_hat );
		DCS_DEBUG_TRACE_L( 5, ">>" << i << " --> P: " << P );
		DCS_DEBUG_TRACE_L( 5, ">>" << i << " --> phi: " << phi );
		DCS_DEBUG_TRACE_L( 5, ">>" << i << " --> y_hat: " << yy_hat );
		DCS_DEBUG_TRACE_L( 5, ">>" << i << " --> y_hat_ok: " << y_hat_ok(i) );
		DCS_DEBUG_TRACE_L( 5, ">>" << i << " --> r.e.: " << (yy_hat/y(i)-real_type(1)) );
		DCS_DEBUG_TRACE_L( 5, "----------------------------------------" << i );

//		DCS_TEST_CHECK_REL_CLOSE(y_hat, y_hat_ok(i), 1.0e-5);
	}

	DCS_DEBUG_TRACE( "u=" << u );
	DCS_DEBUG_TRACE( "y=" << y );
	DCS_DEBUG_TRACE( "y_hat=" << y_hat );
	DCS_DEBUG_TRACE( "y_hat_ok=" << y_hat_ok );
	DCS_TEST_CHECK_VECTOR_CLOSE( y_hat, y_hat_ok, n_obs, tol );
}


DCS_TEST_DEF( rarx_siso_with_noise )
{
	DCS_DEBUG_TRACE("Test Case: SISO system with ARX structure and with noise");

	typedef double real_type;
	typedef ::std::size_t size_type;
	typedef ::std::size_t uint_type;
	typedef ublas::vector<real_type> vector_type;;
	typedef ublas::matrix<real_type> matrix_type;;

	const size_type n_obs(50);
	const size_type n_a(2);
	const size_type n_b(3);
	const size_type d(1);
	const real_type ff(0.98);

	vector_type u(n_obs); // input data
	u( 0) =  1;
	u( 1) =  1;
	u( 2) = -1;
	u( 3) =  1;
	u( 4) =  1;
	u( 5) = -1;
	u( 6) = -1;
	u( 7) =  1;
	u( 8) =  1;
	u( 9) =  1;
	u(10) = -1;
	u(11) =  1;
	u(12) =  1;
	u(13) = -1;
	u(14) =  1;
	u(15) = -1;
	u(16) = -1;
	u(17) =  1;
	u(18) =  1;
	u(19) =  1;
	u(20) =  1;
	u(21) = -1;
	u(22) =  1;
	u(23) =  1;
	u(24) =  1;
	u(25) =  1;
	u(26) =  1;
	u(27) = -1;
	u(28) =  1;
	u(29) = -1;
	u(30) =  1;
	u(31) = -1;
	u(32) = -1;
	u(33) = -1;
	u(34) = -1;
	u(35) =  1;
	u(36) =  1;
	u(37) = -1;
	u(38) =  1;
	u(39) = -1;
	u(40) = -1;
	u(41) = -1;
	u(42) =  1;
	u(43) =  1;
	u(44) = -1;
	u(45) = -1;
	u(46) = -1;
	u(47) =  1;
	u(48) =  1;
	u(49) =  1;

	vector_type y(n_obs); // output data
	y( 0) =  0.027269435602257;
	y( 1) =  1.42910653602721;
	y( 2) =  3.13172985929052;
	y( 3) =  3.42043783778768;
	y( 4) =  3.81121145611023;
	y( 5) =  4.88586659533211;
	y( 6) =  3.50021865618919;
	y( 7) =  0.81973552518544;
	y( 8) = -0.670729051110678;
	y( 9) =  0.40353481643172;
	y(10) =  1.92450509756086;
	y(11) =  2.57330187223511;
	y(12) =  3.27323057964743;
	y(13) =  4.51945349682689;
	y(14) =  4.29874025585624;
	y(15) =  3.33686119225506;
	y(16) =  1.04243779634078;
	y(17) = -1.9188512000648;
	y(18) = -3.03150284917328;
	y(19) = -0.851342715740552;
	y(20) =  1.90936283038388;
	y(21) =  5.12673816858847;
	y(22) =  5.84083013749686;
	y(23) =  6.11865734678332;
	y(24) =  6.45101625110592;
	y(25) =  7.00152470437322;
	y(26) =  7.65196032017242;
	y(27) =  8.20289639541411;
	y(28) =  6.45796351870106;
	y(29) =  4.26070989141841;
	y(30) =  1.8864680954621;
	y(31) = -0.0861820756437078;
	y(32) = -2.11178303836729;
	y(33) = -4.51770612233904;
	y(34) = -7.4688316237776;
	y(35) = -8.97712711199028;
	y(36) = -7.54199441975046;
	y(37) = -3.91689833568865;
	y(38) = -0.770670837748398;
	y(39) =  1.59977116916718;
	y(40) =  2.25133137414749;
	y(41) =  0.761088984527941;
	y(42) = -1.65674073302557;
	y(43) = -1.94103095598901;
	y(44) = -1.07341414424014;
	y(45) = -0.677135558007186;
	y(46) = -2.04495415286072;
	y(47) = -4.25604879619889;
	y(48) = -3.95709371152744;
	y(49) = -1.60486986417004;

	vector_type y_hat_ok(n_obs);
	y_hat_ok( 0) =  0;
	y_hat_ok( 1) =  2.159895738702e-16;
	y_hat_ok( 2) =  1.48355446039972;
	y_hat_ok( 3) =  1.73172474107969;
	y_hat_ok( 4) =  7.16845397508742;
	y_hat_ok( 5) =  1.46405922887714;
	y_hat_ok( 6) =  6.0187628450461;
	y_hat_ok( 7) =  0.149908371859726;
	y_hat_ok( 8) = -0.309401802011093;
	y_hat_ok( 9) = -0.945066134521384;
	y_hat_ok(10) =  2.93765050973589;
	y_hat_ok(11) =  1.80544014856298;
	y_hat_ok(12) =  2.96645154356371;
	y_hat_ok(13) =  4.50834801847696;
	y_hat_ok(14) =  3.96178717286711;
	y_hat_ok(15) =  4.02512322271238;
	y_hat_ok(16) =  1.77342913790201;
	y_hat_ok(17) = -2.0527264246619;
	y_hat_ok(18) = -2.97659171641966;
	y_hat_ok(19) = -1.79968138012433;
	y_hat_ok(20) =  2.36232101170433;
	y_hat_ok(21) =  4.6775748530558;
	y_hat_ok(22) =  5.51833856480925;
	y_hat_ok(23) =  5.54776465192265;
	y_hat_ok(24) =  6.56574806666596;
	y_hat_ok(25) =  7.00902773355202;
	y_hat_ok(26) =  7.58107069492045;
	y_hat_ok(27) =  8.18282528619849;
	y_hat_ok(28) =  6.50829497106674;
	y_hat_ok(29) =  4.72641467336699;
	y_hat_ok(30) =  1.43897511801286;
	y_hat_ok(31) =  0.566107486429317;
	y_hat_ok(32) = -1.78285395551137;
	y_hat_ok(33) = -4.58187374649256;
	y_hat_ok(34) = -6.73314421043879;
	y_hat_ok(35) = -9.5939772640019;
	y_hat_ok(36) = -7.74610227745837;
	y_hat_ok(37) = -3.5377549768095;
	y_hat_ok(38) = -1.22661989339996;
	y_hat_ok(39) =  1.97256193475138;
	y_hat_ok(40) =  2.34745525270751;
	y_hat_ok(41) =  0.723703289462463;
	y_hat_ok(42) = -1.95540045556128;
	y_hat_ok(43) = -2.51231661965855;
	y_hat_ok(44) = -0.245724575888837;
	y_hat_ok(45) = -0.731496900972052;
	y_hat_ok(46) = -1.69346081211635;
	y_hat_ok(47) = -4.01671244038697;
	y_hat_ok(48) = -4.42848639751517;
	y_hat_ok(49) = -1.51830722259658;


	vector_type theta_hat;
	matrix_type P;
	vector_type phi;

	::dcs::sysid::rls_arx_siso_init(n_a, n_b, d, theta_hat, P, phi);

	DCS_DEBUG_TRACE_L( 5, "theta0_hat: " << theta_hat );
	DCS_DEBUG_TRACE_L( 5, "P0: " << P );
	DCS_DEBUG_TRACE_L( 5, "phi0: " << phi );

	vector_type y_hat(n_obs);
	for (size_type i = 0; i < n_obs; ++i)
	{
		real_type yy_hat;

		yy_hat = ::dcs::sysid::rls_ff_arx_siso(
			y(i),
			u(i),
			ff,
			n_a,
			n_b,
			d,
			theta_hat,
			P,
			phi
		);
		y_hat(i) = yy_hat;

		DCS_DEBUG_TRACE_L( 5, ">> Observation #" << i );
		DCS_DEBUG_TRACE_L( 5, ">>" << i << " --> u: " << u(i) );
		DCS_DEBUG_TRACE_L( 5, ">>" << i << " --> y: " << y(i) );
		DCS_DEBUG_TRACE_L( 5, ">>" << i << " --> theta_hat: " << theta_hat );
		DCS_DEBUG_TRACE_L( 5, ">>" << i << " --> P: " << P );
		DCS_DEBUG_TRACE_L( 5, ">>" << i << " --> phi: " << phi );
		DCS_DEBUG_TRACE_L( 5, ">>" << i << " --> y_hat: " << yy_hat );
		DCS_DEBUG_TRACE_L( 5, ">>" << i << " --> y_hat_ok: " << y_hat_ok(i) );
		DCS_DEBUG_TRACE_L( 5, ">>" << i << " --> r.e.: " << (yy_hat/y(i)-real_type(1)) );
		DCS_DEBUG_TRACE_L( 5, "----------------------------------------" << i );

//		DCS_TEST_CHECK_REL_CLOSE(y_hat, y_hat_ok(i), 1.0e-5);
	}

	DCS_DEBUG_TRACE( "u=" << u );
	DCS_DEBUG_TRACE( "y=" << y );
	DCS_DEBUG_TRACE( "y_hat=" << y_hat );
	DCS_DEBUG_TRACE( "y_hat_ok=" << y_hat_ok );
	DCS_TEST_CHECK_VECTOR_CLOSE( y_hat, y_hat_ok, n_obs, tol );
}


DCS_TEST_DEF( rarx_miso_without_noise )
{
	DCS_DEBUG_TRACE("Test Case: MISO system with ARX structure and without noise");

	typedef double real_type;
	typedef ::std::size_t size_type;
	typedef ublas::matrix<real_type> matrix_type;
	typedef ublas::vector<real_type> vector_type;

	const size_type n_obs = 25; // # observations
	const size_type n_a = 2; // the order of the ARX model
	const size_type n_b = 2; // the order of the ARX model
	const size_type n_u = 2; // the size of input vector
	const size_type d = 1; // the input delay
	const real_type ff = 0.98; // the forgetting factor

	matrix_type U(n_obs,n_u);
	U( 0,0) = -1.0; U( 0,1) = -1.0;
	U( 1,0) = -1.0; U( 1,1) = -1.0;
	U( 2,0) =  1.0; U( 2,1) = -1.0;
	U( 3,0) = -1.0; U( 3,1) = -1.0;
	U( 4,0) =  1.0; U( 4,1) = -1.0;
	U( 5,0) =  1.0; U( 5,1) =  1.0;
	U( 6,0) =  1.0; U( 6,1) = -1.0;
	U( 7,0) = -1.0; U( 7,1) =  1.0;
	U( 8,0) = -1.0; U( 8,1) =  1.0;
	U( 9,0) = -1.0; U( 9,1) =  1.0;
	U(10,0) =  1.0; U(10,1) =  1.0;
	U(11,0) =  1.0; U(11,1) =  1.0;
	U(12,0) = -1.0; U(12,1) = -1.0;
	U(13,0) = -1.0; U(13,1) =  1.0;
	U(14,0) = -1.0; U(14,1) = -1.0;
	U(15,0) = -1.0; U(15,1) = -1.0;
	U(16,0) =  1.0; U(16,1) = -1.0;
	U(17,0) =  1.0; U(17,1) =  1.0;
	U(18,0) =  1.0; U(18,1) = -1.0;
	U(19,0) = -1.0; U(19,1) =  1.0;
	U(20,0) =  1.0; U(20,1) = -1.0;
	U(21,0) = -1.0; U(21,1) = -1.0;
	U(22,0) =  1.0; U(22,1) =  1.0;
	U(23,0) =  1.0; U(23,1) =  1.0;
	U(24,0) = -1.0; U(24,1) = -1.0;


	vector_type y(n_obs);
	y( 0) = -5.00000000e-01;
	y( 1) = -5.00000000e-01;
	y( 2) =  1.11000000e+00;
	y( 3) = -6.31000000e+00;
	y( 4) =  1.29560000e+01;
	y( 5) = -3.65970000e+01;
	y( 6) =  8.59596500e+01;
	y( 7) = -2.16322175e+02;
	y( 8) =  5.14889120e+02;
	y( 9) = -1.25448088e+03;
	y(10) =  3.03143890e+03;
	y(11) = -7.33945830e+03;
	y(12) =  1.77600940e+04;
	y(13) = -4.29877638e+04;
	y(14) =  1.04033071e+05;
	y(15) = -2.51779966e+05;
	y(16) =  6.09341983e+05;
	y(17) = -1.47470321e+06;
	y(18) =  3.56899718e+06;
	y(19) = -8.63750985e+06;
	y(20) =  2.09040482e+07;
	y(21) = -5.05908975e+07;
	y(22) =  1.22437451e+08;
	y(23) = -2.96316758e+08;
	y(24) =  7.17130398e+08;


	vector_type y_hat_ok(n_obs);
	y_hat_ok( 0) =          0;
	y_hat_ok( 1) = -        3.33066907387547e-16;
	y_hat_ok( 2) = -        0.49997865868872;
	y_hat_ok( 3) =          1.73315372556005;
	y_hat_ok( 4) =         13.3532910265408;
	y_hat_ok( 5) = -       32.4861683233654;
	y_hat_ok( 6) =         92.9140722657638;
	y_hat_ok( 7) = -      216.150231627515;
	y_hat_ok( 8) =        516.251284887384;
	y_hat_ok( 9) = -     1227.86089663991;
	y_hat_ok(10) =       3002.37409743852;
	y_hat_ok(11) = -     7347.81893929666;
	y_hat_ok(12) =      17757.1789534454;
	y_hat_ok(13) = -    42988.6459864674;
	y_hat_ok(14) =     104036.840655408;
	y_hat_ok(15) = -   251778.240754479;
	y_hat_ok(16) =     609335.942614751;
	y_hat_ok(17) = -  1474706.41235492;
	y_hat_ok(18) =    3568998.13972857;
	y_hat_ok(19) = -  8637513.97657324;
	y_hat_ok(20) =   20904051.5008928;
	y_hat_ok(21) = - 50590903.9921828;
	y_hat_ok(22) =  122437449.324008;
	y_hat_ok(23) = -296316760.059735;
	y_hat_ok(24) =  717130399.72078;

	vector_type theta_hat; //(n_y, n1);
	matrix_type P; //(n2,n2);
	vector_type phi; //(n2);

	::dcs::sysid::rls_arx_miso_init(n_a, n_b, d, n_u, theta_hat, P, phi);

	DCS_DEBUG_TRACE( "theta0_hat: " << theta_hat );
	DCS_DEBUG_TRACE( "P0: " << P );
	DCS_DEBUG_TRACE( "phi0: " << phi );

	vector_type y_hat(n_obs);
	for (size_type i = 0; i < n_obs; ++i)
	{
		real_type yy(y(i));
		vector_type u(ublas::row(U, i));
		real_type yy_hat;

		yy_hat = ::dcs::sysid::rls_ff_arx_miso(
			yy,
			u,
			ff,
			n_a,
			n_b,
			d,
			theta_hat,
			P,
			phi
		);
		y_hat(i) = yy_hat;

		DCS_DEBUG_TRACE( ">> Observation #" << i );
		DCS_DEBUG_TRACE( ">>" << i << " --> u: " << u );
		DCS_DEBUG_TRACE( ">>" << i << " --> y: " << yy );
		DCS_DEBUG_TRACE( ">>" << i << " --> theta_hat: " << theta_hat );
		DCS_DEBUG_TRACE( ">>" << i << " --> P: " << P );
		DCS_DEBUG_TRACE( ">>" << i << " --> phi: " << phi );
		DCS_DEBUG_TRACE( ">>" << i << " --> y_hat: " << yy_hat );
		DCS_DEBUG_TRACE( ">>" << i << " --> y_hat_ok: " << y_hat_ok(i) );
		DCS_DEBUG_TRACE( ">>" << i << " --> r.e.: " << ((yy_hat-yy)/ yy) );
		DCS_DEBUG_TRACE( "----------------------------------------" << i );

//		for (size_type j = 0; j < n_y; ++j)
//		{
//			DCS_TEST_CHECK_REL_CLOSE(y_hat(j), Y_hat_ok(i,j), 1.0e-5);
//		}
	}
	::std::cerr.precision(11);
	DCS_DEBUG_TRACE( "y_hat_ok: " << y_hat_ok );
	DCS_DEBUG_TRACE( "y_hat: " << y_hat );
	DCS_TEST_CHECK_VECTOR_CLOSE(y_hat, y_hat_ok, n_obs, tol);
}


DCS_TEST_DEF( rarx_miso_with_noise )
{
	DCS_DEBUG_TRACE("Test Case: MISO system with ARX structure and with noise");

	typedef double real_type;
	typedef ::std::size_t size_type;
	typedef ublas::matrix<real_type> matrix_type;
	typedef ublas::vector<real_type> vector_type;

	::std::istringstream iss;

	::std::string input_data = "[25,2](\
		(-1.00000000e+00, -1.00000000e+00),\
		(-1.00000000e+00, -1.00000000e+00),\
		( 1.00000000e+00, -1.00000000e+00),\
		(-1.00000000e+00, -1.00000000e+00),\
		( 1.00000000e+00, -1.00000000e+00),\
		( 1.00000000e+00,  1.00000000e+00),\
		( 1.00000000e+00, -1.00000000e+00),\
		(-1.00000000e+00,  1.00000000e+00),\
		(-1.00000000e+00,  1.00000000e+00),\
		(-1.00000000e+00,  1.00000000e+00),\
		( 1.00000000e+00,  1.00000000e+00),\
		( 1.00000000e+00,  1.00000000e+00),\
		(-1.00000000e+00, -1.00000000e+00),\
		(-1.00000000e+00,  1.00000000e+00),\
		(-1.00000000e+00, -1.00000000e+00),\
		(-1.00000000e+00, -1.00000000e+00),\
		( 1.00000000e+00, -1.00000000e+00),\
		( 1.00000000e+00,  1.00000000e+00),\
		( 1.00000000e+00, -1.00000000e+00),\
		(-1.00000000e+00,  1.00000000e+00),\
		( 1.00000000e+00, -1.00000000e+00),\
		(-1.00000000e+00, -1.00000000e+00),\
		( 1.00000000e+00,  1.00000000e+00),\
		( 1.00000000e+00,  1.00000000e+00),\
		(-1.00000000e+00, -1.00000000e+00))";
	matrix_type U;
	iss.str(input_data);
	iss >> U;

	::std::string noise_data = "[25,2](\
		( 3.08172011e-01, -2.45693088e-01),\
		( 2.85335326e-01, -3.08004951e-01),\
		(-1.95837736e-02,  3.53297234e-01),\
		( 3.70122695e-01, -2.16048598e-01),\
		(-9.97247159e-02, -1.24976540e-01),\
		( 1.51356721e-01, -1.32192925e-02),\
		( 1.27577162e-01,  1.74148053e-02),\
		( 3.43638492e-01, -3.24963713e-01),\
		(-1.51691115e-01,  1.19461012e-01),\
		(-1.08410776e-01, -2.90062964e-02),\
		(-3.28313303e-01, -6.14573765e-01),\
		(-4.85119390e-01,  3.77101946e-01),\
		(-1.35164607e-01, -1.34291744e-01),\
		( 1.32145774e-01, -2.54598641e-01),\
		(-5.36407195e-03,  9.46612239e-02),\
		( 1.19110858e-01,  1.70817089e-01),\
		(-1.63276923e-01,  1.21445715e-01),\
		( 2.15859342e-01,  3.47538376e-01),\
		(-1.16053388e-02, -1.63092351e-01),\
		( 1.51414967e-01,  8.45085859e-02),\
		( 1.95624602e-01, -1.37456787e-01),\
		(-9.67659712e-02,  4.04050559e-02),\
		(-1.47895670e-01, -1.21097851e-01),\
		(-8.54656473e-03,  2.07195950e-01),\
		( 1.89103320e-02,  8.60167891e-03))";
	matrix_type E;
	iss.str(noise_data);
	iss >> E;

	::std::string output_data = "[25,1](\
		(-3.45913994e-01),\
		(-1.43300971e-01),\
		( 1.54498732e+00),\
		(-6.53928706e+00),\
		( 1.46862817e+01),\
		(-3.92207194e+01),\
		( 9.39887947e+01),\
		(-2.33895468e+02),\
		( 5.59856224e+02),\
		(-1.36104369e+03),\
		( 3.29161246e+03),\
		(-7.96569571e+03),\
		( 1.92784928e+04),\
		(-4.66592139e+04),\
		( 1.12922429e+05),\
		(-2.73289602e+05),\
		( 6.61402124e+05),\
		(-1.60069276e+06),\
		( 3.87391364e+06),\
		(-9.37544916e+06),\
		( 2.26899742e+07),\
		(-5.49130987e+07),\
		( 1.32897822e+08),\
		(-3.21632398e+08),\
		( 7.78397998e+08))";
	matrix_type Y;
	iss.str(output_data);
	iss >> Y;

	::std::string predicted_data = "[25,1](\
		( 0.00000000e+00),\
		(-3.70820556e-16),\
		(-1.39035452e-01),\
		( 2.51015442e+00),\
		( 6.45949146e+00),\
		(-4.37206630e+01),\
		( 1.01182342e+02),\
		(-2.32915933e+02),\
		( 5.62331447e+02),\
		(-1.33465358e+03),\
		( 3.20705345e+03),\
		(-7.95309039e+03),\
		( 1.92889776e+04),\
		(-4.66707923e+04),\
		( 1.12924474e+05),\
		(-2.73288701e+05),\
		( 6.61405013e+05),\
		(-1.60069125e+06),\
		( 3.87391704e+06),\
		(-9.37545305e+06),\
		( 2.26899713e+07),\
		(-5.49131142e+07),\
		( 1.32897820e+08),\
		(-3.21632396e+08),\
		( 7.78397994e+08))";
	matrix_type Y_hat_ok;
	iss.str(predicted_data);
	iss >> Y_hat_ok;

	size_type n_obs = 25; // # observations
	size_type n_a = 2; // the order of the ARX model
	size_type n_y = 1; // the size of output vector
	size_type n_b = 2; // the order of the ARX model
	size_type n_u = 2; // the size of input vector
	size_type d = 1;

	matrix_type theta_hat; //(n_y, n1);
	matrix_type P; //(n2,n2);
	vector_type phi; //(n2);

	::dcs::sysid::rls_arx_mimo_init(n_a, n_b, d, n_y, n_u, theta_hat, P, phi);

	DCS_DEBUG_TRACE( "theta0_hat: " << theta_hat );
	DCS_DEBUG_TRACE( "P0: " << P );
	DCS_DEBUG_TRACE( "phi0: " << phi );

	for (size_type i = 0; i < n_obs; ++i)
	{
		vector_type y(ublas::row(Y, i));
		vector_type u(ublas::row(U, i));
		vector_type y_hat;

		y_hat = ::dcs::sysid::rls_ff_arx_mimo(
			y,
			u,
			0.98,
			n_a,
			n_b,
			d,
			theta_hat,
			P,
			phi
		);

		DCS_DEBUG_TRACE( ">> Observation #" << i );
		DCS_DEBUG_TRACE( ">>" << i << " --> u: " << u );
		DCS_DEBUG_TRACE( ">>" << i << " --> e: " << ublas::row(E, i) );
		DCS_DEBUG_TRACE( ">>" << i << " --> y: " << y );
		DCS_DEBUG_TRACE( ">>" << i << " --> theta_hat: " << theta_hat );
		DCS_DEBUG_TRACE( ">>" << i << " --> P: " << P );
		DCS_DEBUG_TRACE( ">>" << i << " --> phi: " << phi );
		DCS_DEBUG_TRACE( ">>" << i << " --> y_hat: " << y_hat );
		DCS_DEBUG_TRACE( ">>" << i << " --> y_hat_ok: " << ublas::row(Y_hat_ok, i) );
		DCS_DEBUG_TRACE( ">>" << i << " --> r.e.: " << (ublas::element_div(y_hat-y, y)) );
		DCS_DEBUG_TRACE( "----------------------------------------" << i );

		for (size_type j = 0; j < n_y; ++j)
		{
			DCS_TEST_CHECK_REL_CLOSE(y_hat(j), Y_hat_ok(i,j), 1.0e-5);
		}
	}
}


DCS_TEST_DEF( rarx_mimo_without_noise )
{
	DCS_DEBUG_TRACE("Test Case: MIMO system with ARX structure and without noise");

	typedef double real_type;
	typedef ::std::size_t size_type;
	typedef ublas::matrix<real_type> matrix_type;
	typedef ublas::vector<real_type> vector_type;

	const size_type n_obs = 25; // # observations
	const size_type n_a = 2; // the order of the ARX model
	const size_type n_y = 2; // the size of output vector
	const size_type n_b = 2; // the order of the ARX model
	const size_type n_u = 2; // the size of input vector
	const size_type d = 0; // the input delay

	::std::istringstream iss;

	::std::string input_data = "[25,2](\
		(-1.00000000e+00, -1.00000000e+00),\
		(-1.00000000e+00, -1.00000000e+00),\
		( 1.00000000e+00, -1.00000000e+00),\
		(-1.00000000e+00, -1.00000000e+00),\
		( 1.00000000e+00, -1.00000000e+00),\
		( 1.00000000e+00,  1.00000000e+00),\
		( 1.00000000e+00, -1.00000000e+00),\
		(-1.00000000e+00,  1.00000000e+00),\
		(-1.00000000e+00,  1.00000000e+00),\
		(-1.00000000e+00,  1.00000000e+00),\
		( 1.00000000e+00,  1.00000000e+00),\
		( 1.00000000e+00,  1.00000000e+00),\
		(-1.00000000e+00, -1.00000000e+00),\
		(-1.00000000e+00,  1.00000000e+00),\
		(-1.00000000e+00, -1.00000000e+00),\
		(-1.00000000e+00, -1.00000000e+00),\
		( 1.00000000e+00, -1.00000000e+00),\
		( 1.00000000e+00,  1.00000000e+00),\
		( 1.00000000e+00, -1.00000000e+00),\
		(-1.00000000e+00,  1.00000000e+00),\
		( 1.00000000e+00, -1.00000000e+00),\
		(-1.00000000e+00, -1.00000000e+00),\
		( 1.00000000e+00,  1.00000000e+00),\
		( 1.00000000e+00,  1.00000000e+00),\
		(-1.00000000e+00, -1.00000000e+00))";
	matrix_type U;
	iss.str(input_data);
	iss >> U;

	::std::string output_data = "[25,2](\
		(-5.00000000e-01,  2.00000000e-01),\
		(-5.00000000e-01,  8.50000000e-01),\
		( 1.11000000e+00,  2.49000000e+00),\
		(-6.31000000e+00, -2.37750000e+00),\
		( 1.29560000e+01,  1.28237500e+01),\
		(-3.65970000e+01, -2.52615000e+01),\
		( 8.59596500e+01,  6.85872625e+01),\
		(-2.16322175e+02, -1.61890181e+02),\
		( 5.14889120e+02,  3.95251008e+02),\
		(-1.25448088e+03, -9.53798566e+02),\
		( 3.03143890e+03,  2.31236360e+03),\
		(-7.33945830e+03, -5.59327618e+03),\
		( 1.77600940e+04,  1.35388854e+04),\
		(-4.29877638e+04, -3.27650970e+04),\
		( 1.04033071e+05,  7.92989952e+04),\
		(-2.51779966e+05, -1.91911412e+05),\
		( 6.09341983e+05,  4.64461202e+05),\
		(-1.47470321e+06, -1.12405886e+06),\
		( 3.56899718e+06,  2.72039643e+06),\
		(-8.63750985e+06, -6.58375763e+06),\
		( 2.09040482e+07,  1.59336759e+07),\
		(-5.05908975e+07, -3.85618464e+07),\
		( 1.22437451e+08,  9.33253832e+07),\
		(-2.96316758e+08, -2.25861231e+08),\
		( 7.17130398e+08,  5.46617611e+08))";
	matrix_type Y;
	iss.str(output_data);
	iss >> Y;

	matrix_type Y_hat_ok(n_obs,n_y);
	Y_hat_ok( 0, 0) =  1.0e+8*0.000000000000000000; Y_hat_ok( 0, 1) =  1.0e+8*0.00000000000000000;
	Y_hat_ok( 1, 0) = -1.0e+8*0.000000000000000000; Y_hat_ok( 1, 1) = -1.0e+8*0.00000000000000000;
	Y_hat_ok( 2, 0) = -1.0e+8*0.000000005283621206; Y_hat_ok( 2, 1) =  1.0e+8*0.00000000898215605;
	Y_hat_ok( 3, 0) =  1.0e+8*0.000000024601476220; Y_hat_ok( 3, 1) =  1.0e+8*0.00000002356959581;
	Y_hat_ok( 4, 0) =  1.0e+8*0.000000094977859570; Y_hat_ok( 4, 1) =  1.0e+8*0.00000006927309048;
	Y_hat_ok( 5, 0) = -1.0e+8*0.000000411967834400; Y_hat_ok( 5, 1) = -1.0e+8*0.00000028340996070;
	Y_hat_ok( 6, 0) =  1.0e+8*0.000000897996785700; Y_hat_ok( 6, 1) =  1.0e+8*0.00000071663894800;
	Y_hat_ok( 7, 0) = -1.0e+8*0.000002152582777000; Y_hat_ok( 7, 1) = -1.0e+8*0.00000165415078700;
	Y_hat_ok( 8, 0) =  1.0e+8*0.000005163068273000; Y_hat_ok( 8, 1) =  1.0e+8*0.00000398973057700;
	Y_hat_ok( 9, 0) = -1.0e+8*0.000012278135770000; Y_hat_ok( 9, 1) = -1.0e+8*0.00000932260343600;
	Y_hat_ok(10, 0) =  1.0e+8*0.000029919223520000; Y_hat_ok(10, 1) =  1.0e+8*0.00002288293381000;
	Y_hat_ok(11, 0) = -1.0e+8*0.000073689696700000; Y_hat_ok(11, 1) = -1.0e+8*0.00005608739838000;
	Y_hat_ok(12, 0) =  1.0e+8*0.000177592627100000; Y_hat_ok(12, 1) =  1.0e+8*0.00013536744330000;
	Y_hat_ok(13, 0) = -1.0e+8*0.000429863450400000; Y_hat_ok(13, 1) = -1.0e+8*0.00032765673910000;
	Y_hat_ok(14, 0) =  1.0e+8*0.001040326831825820; Y_hat_ok(14, 1) =  1.0e+8*0.00079298532160000;
	Y_hat_ok(15, 0) = -1.0e+8*0.002517785487456600; Y_hat_ok(15, 1) = -1.0e+8*0.00191913439988734;
	Y_hat_ok(16, 0) =  1.0e+8*0.006093447997852560; Y_hat_ok(16, 1) =  1.0e+8*0.00464462954500000;
	Y_hat_ok(17, 0) = -1.0e+8*0.014747044535083100; Y_hat_ok(17, 1) = -1.0e+8*0.01124059073908370;
	Y_hat_ok(18, 0) =  1.0e+8*0.035689991787426000; Y_hat_ok(18, 1) =  1.0e+8*0.02720399688901190;
	Y_hat_ok(19, 0) = -1.0e+8*0.086375124534214900; Y_hat_ok(19, 1) = -1.0e+8*0.06583759904544560;
	Y_hat_ok(20, 0) =  1.0e+8*0.209040443417509000; Y_hat_ok(20, 1) =  1.0e+8*0.15933672525961700;
	Y_hat_ok(21, 0) = -1.0e+8*0.505909078284890000; Y_hat_ok(21, 1) = -1.0e+8*0.38561856568669400;
	Y_hat_ok(22, 0) =  1.0e+8*1.224374488649260000; Y_hat_ok(22, 1) =  1.0e+8*0.93325380833359700;
	Y_hat_ok(23, 0) = -1.0e+8*2.963167642482150000; Y_hat_ok(23, 1) = -1.0e+8*2.25861234817228000;
	Y_hat_ok(24, 0) =  1.0e+8*7.171304680725230000; Y_hat_ok(24, 1) =  1.0e+8*5.46617649941086000;


	matrix_type theta_hat; //(n_y, n1);
	matrix_type P; //(n2,n2);
	vector_type phi; //(n2);

	::dcs::sysid::rls_arx_mimo_init(n_a, n_b, d, n_y, n_u, theta_hat, P, phi);

	DCS_DEBUG_TRACE( "theta0_hat: " << theta_hat );
	DCS_DEBUG_TRACE( "P0: " << P );
	DCS_DEBUG_TRACE( "phi0: " << phi );

	matrix_type Y_hat(n_obs, n_y);
	for (size_type i = 0; i < n_obs; ++i)
	{
		vector_type y(ublas::row(Y, i));
		vector_type u(ublas::row(U, i));
		vector_type y_hat;

		y_hat = ::dcs::sysid::rls_ff_arx_mimo(
			y,
			u,
			0.98,
			n_a,
			n_b,
			d,
			theta_hat,
			P,
			phi
		);
		ublas::row(Y_hat, i) = y_hat;

		DCS_DEBUG_TRACE( ">> Observation #" << i );
		DCS_DEBUG_TRACE( ">>" << i << " --> u: " << u );
		DCS_DEBUG_TRACE( ">>" << i << " --> y: " << y );
		DCS_DEBUG_TRACE( ">>" << i << " --> theta_hat: " << theta_hat );
		DCS_DEBUG_TRACE( ">>" << i << " --> P: " << P );
		DCS_DEBUG_TRACE( ">>" << i << " --> phi: " << phi );
		DCS_DEBUG_TRACE( ">>" << i << " --> y_hat: " << y_hat );
		DCS_DEBUG_TRACE( ">>" << i << " --> y_hat_ok: " << ublas::row(Y_hat_ok, i) );
		DCS_DEBUG_TRACE( ">>" << i << " --> r.e.: " << (ublas::element_div(y_hat-y, y)) );
		DCS_DEBUG_TRACE( "----------------------------------------" << i );

//		for (size_type j = 0; j < n_y; ++j)
//		{
//			DCS_TEST_CHECK_REL_CLOSE(y_hat(j), Y_hat_ok(i,j), 1.0e-5);
//		}
	}
	(DCS_DEBUG_STREAM).precision(15);
	DCS_DEBUG_TRACE( "Y: " << Y );
	DCS_DEBUG_TRACE( "Y_hat_ok: " << Y_hat_ok );
	DCS_DEBUG_TRACE( "Y_hat: " << Y_hat );
	DCS_TEST_CHECK_MATRIX_CLOSE(Y_hat, Y_hat_ok, n_obs, n_y, tol);
}


DCS_TEST_DEF( rarx_mimo_with_noise )
{
	DCS_DEBUG_TRACE("Test Case: MIMO system with ARX structure and with noise");

	typedef double real_type;
	typedef ::std::size_t size_type;
	typedef ublas::matrix<real_type> matrix_type;
	typedef ublas::vector<real_type> vector_type;

	::std::istringstream iss;

	::std::string input_data = "[25,2](\
		(-1.00000000e+00, -1.00000000e+00),\
		(-1.00000000e+00, -1.00000000e+00),\
		( 1.00000000e+00, -1.00000000e+00),\
		(-1.00000000e+00, -1.00000000e+00),\
		( 1.00000000e+00, -1.00000000e+00),\
		( 1.00000000e+00,  1.00000000e+00),\
		( 1.00000000e+00, -1.00000000e+00),\
		(-1.00000000e+00,  1.00000000e+00),\
		(-1.00000000e+00,  1.00000000e+00),\
		(-1.00000000e+00,  1.00000000e+00),\
		( 1.00000000e+00,  1.00000000e+00),\
		( 1.00000000e+00,  1.00000000e+00),\
		(-1.00000000e+00, -1.00000000e+00),\
		(-1.00000000e+00,  1.00000000e+00),\
		(-1.00000000e+00, -1.00000000e+00),\
		(-1.00000000e+00, -1.00000000e+00),\
		( 1.00000000e+00, -1.00000000e+00),\
		( 1.00000000e+00,  1.00000000e+00),\
		( 1.00000000e+00, -1.00000000e+00),\
		(-1.00000000e+00,  1.00000000e+00),\
		( 1.00000000e+00, -1.00000000e+00),\
		(-1.00000000e+00, -1.00000000e+00),\
		( 1.00000000e+00,  1.00000000e+00),\
		( 1.00000000e+00,  1.00000000e+00),\
		(-1.00000000e+00, -1.00000000e+00))";
	matrix_type U;
	iss.str(input_data);
	iss >> U;

	::std::string noise_data = "[25,2](\
		( 3.08172011e-01, -2.45693088e-01),\
		( 2.85335326e-01, -3.08004951e-01),\
		(-1.95837736e-02,  3.53297234e-01),\
		( 3.70122695e-01, -2.16048598e-01),\
		(-9.97247159e-02, -1.24976540e-01),\
		( 1.51356721e-01, -1.32192925e-02),\
		( 1.27577162e-01,  1.74148053e-02),\
		( 3.43638492e-01, -3.24963713e-01),\
		(-1.51691115e-01,  1.19461012e-01),\
		(-1.08410776e-01, -2.90062964e-02),\
		(-3.28313303e-01, -6.14573765e-01),\
		(-4.85119390e-01,  3.77101946e-01),\
		(-1.35164607e-01, -1.34291744e-01),\
		( 1.32145774e-01, -2.54598641e-01),\
		(-5.36407195e-03,  9.46612239e-02),\
		( 1.19110858e-01,  1.70817089e-01),\
		(-1.63276923e-01,  1.21445715e-01),\
		( 2.15859342e-01,  3.47538376e-01),\
		(-1.16053388e-02, -1.63092351e-01),\
		( 1.51414967e-01,  8.45085859e-02),\
		( 1.95624602e-01, -1.37456787e-01),\
		(-9.67659712e-02,  4.04050559e-02),\
		(-1.47895670e-01, -1.21097851e-01),\
		(-8.54656473e-03,  2.07195950e-01),\
		( 1.89103320e-02,  8.60167891e-03))";
	matrix_type E;
	iss.str(noise_data);
	iss >> E;

	::std::string output_data = "[25,2](\
		(-3.45913994e-01,  1.59413142e-02),\
		(-1.43300971e-01,  4.59962449e-01),\
		( 1.54498732e+00,  2.50084935e+00),\
		(-6.53928706e+00, -3.17576279e+00),\
		( 1.46862817e+01,  1.33995987e+01),\
		(-3.92207194e+01, -2.80764173e+01),\
		( 9.39887947e+01,  7.38579240e+01),\
		(-2.33895468e+02, -1.76553533e+02),\
		( 5.59856224e+02,  4.28366755e+02),\
		(-1.36104369e+03, -1.03624595e+03),\
		( 3.29161246e+03,  2.50884712e+03),\
		(-7.96569571e+03, -6.07204769e+03),\
		( 1.92784928e+04,  1.46946331e+04),\
		(-4.66592139e+04, -3.55655468e+04),\
		( 1.12922429e+05,  8.60727815e+04),\
		(-2.73289602e+05, -2.08308551e+05),\
		( 6.61402124e+05,  5.04141156e+05),\
		(-1.60069276e+06, -1.22009332e+06),\
		( 3.87391364e+06,  2.95281082e+06),\
		(-9.37544916e+06, -7.14623820e+06),\
		( 2.26899742e+07,  1.72949585e+07),\
		(-5.49130987e+07, -4.18563557e+07),\
		( 1.32897822e+08,  1.01298579e+08),\
		(-3.21632398e+08, -2.45157549e+08),\
		( 7.78397998e+08,  5.93317552e+08))";
	matrix_type Y;
	iss.str(output_data);
	iss >> Y;

	::std::string predicted_data = "[25,2](\
		( 0.00000000e+00,  0.00000000e+00),\
		(-3.70820556e-16, -3.70820556e-16),\
		(-1.39035452e-01,  4.46271137e-01),\
		( 2.51015442e+00,  2.91440501e+00),\
		( 6.45949146e+00,  4.23884194e+00),\
		(-4.37206630e+01, -3.28448906e+01),\
		( 1.01182342e+02,  7.91837657e+01),\
		(-2.32915933e+02, -1.78701988e+02),\
		( 5.62331447e+02,  4.29624627e+02),\
		(-1.33465358e+03, -1.01254102e+03),\
		( 3.20705345e+03,  2.43765158e+03),\
		(-7.95309039e+03, -6.04696830e+03),\
		( 1.92889776e+04,  1.46983239e+04),\
		(-4.66707923e+04, -3.55786574e+04),\
		( 1.12924474e+05,  8.60747498e+04),\
		(-2.73288701e+05, -2.08310504e+05),\
		( 6.61405013e+05,  5.04141410e+05),\
		(-1.60069125e+06, -1.22009571e+06),\
		( 3.87391704e+06,  2.95281433e+06),\
		(-9.37545305e+06, -7.14624044e+06),\
		( 2.26899713e+07,  1.72949553e+07),\
		(-5.49131142e+07, -4.18563682e+07),\
		( 1.32897820e+08,  1.01298574e+08),\
		(-3.21632396e+08, -2.45157547e+08),\
		( 7.78397994e+08,  5.93317550e+08))";
	matrix_type Y_hat_ok;
	iss.str(predicted_data);
	iss >> Y_hat_ok;

	size_type n_obs = 25; // # observations
	size_type n_a = 2; // the order of the ARX model
	size_type n_y = 2; // the size of output vector
	size_type n_b = 2; // the order of the ARX model
	size_type n_u = 2; // the size of input vector
	size_type d = 1;

	matrix_type theta_hat; //(n_y, n1);
	matrix_type P; //(n2,n2);
	vector_type phi; //(n2);

	::dcs::sysid::rls_arx_mimo_init(n_a, n_b, d, n_y, n_u, theta_hat, P, phi);

	DCS_DEBUG_TRACE( "theta0_hat: " << theta_hat );
	DCS_DEBUG_TRACE( "P0: " << P );
	DCS_DEBUG_TRACE( "phi0: " << phi );

	for (size_type i = 0; i < n_obs; ++i)
	{
		vector_type y(ublas::row(Y, i));
		vector_type u(ublas::row(U, i));
		vector_type y_hat;

		y_hat = ::dcs::sysid::rls_ff_arx_mimo(
			y,
			u,
			0.98,
			n_a,
			n_b,
			d,
			theta_hat,
			P,
			phi
		);

		DCS_DEBUG_TRACE( ">> Observation #" << i );
		DCS_DEBUG_TRACE( ">>" << i << " --> u: " << u );
		DCS_DEBUG_TRACE( ">>" << i << " --> e: " << ublas::row(E, i) );
		DCS_DEBUG_TRACE( ">>" << i << " --> y: " << y );
		DCS_DEBUG_TRACE( ">>" << i << " --> theta_hat: " << theta_hat );
		DCS_DEBUG_TRACE( ">>" << i << " --> P: " << P );
		DCS_DEBUG_TRACE( ">>" << i << " --> phi: " << phi );
		DCS_DEBUG_TRACE( ">>" << i << " --> y_hat: " << y_hat );
		DCS_DEBUG_TRACE( ">>" << i << " --> y_hat_ok: " << ublas::row(Y_hat_ok, i) );
		DCS_DEBUG_TRACE( ">>" << i << " --> r.e.: " << (ublas::element_div(y_hat-y, y)) );
		DCS_DEBUG_TRACE( "----------------------------------------" << i );

		for (size_type j = 0; j < n_y; ++j)
		{
			DCS_TEST_CHECK_REL_CLOSE(y_hat(j), Y_hat_ok(i,j), 1.0e-5);
		}
	}
}


//DCS_TEST_DEF( simdata_rarx_siso_without_noise )
//{
//	DCS_DEBUG_TRACE("Test Case: SISO system with ARX structure and without noise");
//
//	typedef double real_type;
//	typedef ::std::size_t size_type;
//	typedef ::std::size_t uint_type;
//	typedef ublas::vector<real_type> vector_type;;
//	typedef ublas::matrix<real_type> matrix_type;;
//
//	::std::istringstream iss;
//
//	::std::string input_data = "[50](-1.0, -1.0, -1.0, -1.0,  1.0, -1.0, -1.0, -1.0,  1.0, -1.0, 1.0,  1.0,  1.0, -1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, 1.0,  1.0,  1.0,  1.0, -1.0, -1.0, -1.0,  1.0, -1.0, -1.0,	-1.0, -1.0,  1.0, -1.0,  1.0,  1.0,  1.0, -1.0, -1.0,  1.0, 1.0, -1.0, -1.0, -1.0,  1.0,  1.0,  1.0,  1.0, -1.0, -1.0)";
//	vector_type u;
//	iss.str(input_data);
//	iss >> u;
//
//	::std::string predicted_data = "[50](\
//		 0.00000000e+00, -1.77635684e-16, -1.87482687e+00, -6.76158312e+00, -9.33497429e+00, -6.48490431e+00, -8.10353276e+00, -7.25463600e+00, -7.33128667e+00, -3.70479002e+00, -2.66923525e+00, -5.40730353e-01, 3.17665229e+00,  6.53866828e+00,  6.70921880e+00,  3.73153275e+00, 2.14101469e+00, -6.81560838e-01, -1.50751747e+00, -2.52346682e+00, -2.08885265e+00,  5.68202209e-01,  3.83961013e+00,  7.08037620e+00, 9.59985704e+00,  8.42692318e+00,  4.16941086e+00, -1.18625935e+00, -3.66484518e+00, -5.67374656e+00, -7.78571294e+00, -9.15708810e+00, -1.00445205e+01, -7.61905364e+00, -5.36124738e+00, -1.87366108e+00, 2.85890152e+00,  7.08638603e+00,  7.71842858e+00,  4.88428852e+00, 3.03170983e+00,  2.69156756e+00,  8.53023155e-01, -2.13190427e+00, -5.40581279e+00, -5.75868960e+00, -3.19464651e+00,  8.38939366e-01, 5.27669084e+00,  6.28545520e+00)";
//
//	vector_type y_hat_ok;
//	iss.str(predicted_data);
//	iss >> y_hat_ok;
//
//	size_type n_a = 2;
//	size_type n_b = 2;
//	size_type d = 0;
//
//	vector_type a(n_a);
//	a(0) = -1.5; a(1) = 0.7;
//	vector_type b(n_b+1);
//	b(0) = 0.2; b(1) = 1; b(2) = 0.5;
//	real_type c = 1;
//
//	::dcs::sysid::darx_siso_model<vector_type,real_type,uint_type> mdl(a, b, c);
//
//	vector_type y;
//	y = ::dcs::sysid::simulate(mdl, u);
//
//	vector_type theta_hat;
//	matrix_type P;
//	vector_type phi;
//
//	::dcs::sysid::rls_arx_siso_init(n_a, n_b, d, theta_hat, P, phi);
//
//	DCS_DEBUG_TRACE( "theta0_hat: " );
//	DCS_DEBUG_TRACE( "P0: " << P );
//	DCS_DEBUG_TRACE( "phi0: " << phi );

//	for (size_type i = 0; i < 50; ++i)
//	{
//		real_type y_hat;
//
//		y_hat = ::dcs::sysid::rls_ff_arx_siso(
//			y(i),
//			u(i),
//			0.98,
//			n_a,
//			n_b,
//			d,
//			theta_hat,
//			P,
//			phi
//		);
//
//		DCS_DEBUG_TRACE( ">> Observation #" << i );
//		DCS_DEBUG_TRACE( ">>" << i << " --> u: " << u(i) );
//		DCS_DEBUG_TRACE( ">>" << i << " --> y: " << y(i) );
//		DCS_DEBUG_TRACE( ">>" << i << " --> theta_hat: " << theta_hat );
//		DCS_DEBUG_TRACE( ">>" << i << " --> P: " << P );
//		DCS_DEBUG_TRACE( ">>" << i << " --> phi: " << phi );
//		DCS_DEBUG_TRACE( ">>" << i << " --> y_hat: " << y_hat );
//		DCS_DEBUG_TRACE( ">>" << i << " --> y_hat_ok: " << y_hat_ok(i) );
//		DCS_DEBUG_TRACE( ">>" << i << " --> r.e.: " << (y_hat/y(i)-real_type(1)) );
//		DCS_DEBUG_TRACE( "----------------------------------------" << i );
//
//		DCS_TEST_CHECK_REL_CLOSE(y_hat, y_hat_ok(i), 1.0e-5);
//	}
//}


int main()
{
	DCS_TEST_SUITE("Test suite for Recursive Least-Square algorithms");

	DCS_TEST_BEGIN();

	DCS_TEST_DO( rarx_siso_without_noise );

	DCS_TEST_DO( rarx_siso_with_noise );

	DCS_TEST_DO( rarx_miso_without_noise );

//	DCS_TEST_DO( rarx_miso_with_noise );

	DCS_TEST_DO( rarx_mimo_without_noise );

//	DCS_TEST_DO( rarx_mimo_with_noise );

	DCS_TEST_END();
}
