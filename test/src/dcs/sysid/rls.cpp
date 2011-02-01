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
	const size_type d(0);
	const real_type ff(0.98);

	vector_type u(n_obs);
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

	vector_type y(n_obs);
	y( 0) =  0.000000000000000;
	y( 1) =  1.000000000000000;
	y( 2) =  3.000000000000000;
	y( 3) =  3.300000000000000;
	y( 4) =  3.350000000000000;
	y( 5) =  4.215000000000000;
	y( 6) =  3.477500000000000;
	y( 7) =  0.765750000000001;
	y( 8) = -0.785624999999999;
	y( 9) = -0.214462499999999;
	y(10) =  1.728243750000001;
	y(11) =  2.242489375000000;
	y(12) =  2.653963437500000;
	y(13) =  3.911202593750001;
	y(14) =  3.509029484375001;
	y(15) =  3.025702410937500;
	y(16) =  1.582232977343751;
	y(17) = -1.244642221640624;
	y(18) = -2.474526416601561;
	y(19) = -1.340540069753905;
	y(20) =  1.221358386990235;
	y(21) =  4.270415629313086;
	y(22) =  5.050672573076465;
	y(23) =  5.086717919095538;
	y(24) =  5.594606077489781;
	y(25) =  6.331206572867796;
	y(26) =  7.080585605058848;
	y(27) =  7.689033806580815;
	y(28) =  6.077140786330029;
	y(29) =  4.233387514888474;
	y(30) =  1.596082721901690;
	y(31) = -0.069247177569396;
	y(32) = -1.721128671685277;
	y(33) = -4.033219983229339;
	y(34) = -6.345039904664314;
	y(35) = -8.194305868735935;
	y(36) = -7.349930869838882;
	y(37) = -3.788882196643169;
	y(38) = -1.038371686077538;
	y(39) =  1.594660008533912;
	y(40) =  2.618850193055144;
	y(41) =  1.312013283608978;
	y(42) = -1.365175209725134;
	y(43) = -2.466172113113985;
	y(44) = -1.243635522863384;
	y(45) = -0.639132805115286;
	y(46) = -1.588154341668561;
	y(47) = -3.434838548922141;
	y(48) = -3.540549784215218;
	y(49) = -1.406437692077329;

	vector_type y_hat_ok(n_obs);
	y_hat_ok( 0) =  0.000000000000000;
	y_hat_ok( 1) =  0.000000000000000;
	y_hat_ok( 2) =  0.999903969222796;
	y_hat_ok( 3) =  2.999811784013802;
	y_hat_ok( 4) =  4.030051338753040;
	y_hat_ok( 5) =  4.233020605250848;
	y_hat_ok( 6) =  3.477618665046835;
	y_hat_ok( 7) =  0.765875458183824;
	y_hat_ok( 8) = -0.785507114585483;
	y_hat_ok( 9) = -0.214414983508762;
	y_hat_ok(10) =  1.728218141524053;
	y_hat_ok(11) =  2.242479010891096;
	y_hat_ok(12) =  2.653917608013485;
	y_hat_ok(13) =  3.911187293580382;
	y_hat_ok(14) =  3.509032161517952;
	y_hat_ok(15) =  3.025691722171502;
	y_hat_ok(16) =  1.582260265432459;
	y_hat_ok(17) = -1.244617576999635;
	y_hat_ok(18) = -2.474508541816530;
	y_hat_ok(19) = -1.340533886537183;
	y_hat_ok(20) =  1.221344397879976;
	y_hat_ok(21) =  4.270392607196245;
	y_hat_ok(22) =  5.050662292850067;
	y_hat_ok(23) =  5.086704821579133;
	y_hat_ok(24) =  5.594603320162784;
	y_hat_ok(25) =  6.331202713010411;
	y_hat_ok(26) =  7.080581881953949;
	y_hat_ok(27) =  7.689031148788063;
	y_hat_ok(28) =  6.077147821530055;
	y_hat_ok(29) =  4.233390842678978;
	y_hat_ok(30) =  1.596096924552290;
	y_hat_ok(31) = -0.069242674691179;
	y_hat_ok(32) = -1.721120151924000;
	y_hat_ok(33) = -4.033213675428690;
	y_hat_ok(34) = -6.345032856986659;
	y_hat_ok(35) = -8.194299937681805;
	y_hat_ok(36) = -7.349930679419265;
	y_hat_ok(37) = -3.788886018043527;
	y_hat_ok(38) = -1.038375809938326;
	y_hat_ok(39) =  1.594653181869465;
	y_hat_ok(40) =  2.618848721674821;
	y_hat_ok(41) =  1.312013580287359;
	y_hat_ok(42) = -1.365172201107073;
	y_hat_ok(43) = -2.466170787868957;
	y_hat_ok(44) = -1.243635522151800;
	y_hat_ok(45) = -0.639132268313885;
	y_hat_ok(46) = -1.588153665468250;
	y_hat_ok(47) = -3.434836149800772;
	y_hat_ok(48) = -3.540549416687078;
	y_hat_ok(49) = -1.406438588469492;


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
	const size_type n_b(2);
	const size_type d(0);
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

//	vector_type e(n_obs); // noise data
//	e( 0) = -0.172730564397743;
//	e( 1) =  0.015471818226085;
//	e( 2) = -0.242823408723082;
//	e( 3) = -0.222700148297353;
//	e( 4) = -0.001369865620670;
//	e( 5) =  0.306526061656950;
//	e( 6) = -0.153933182750736;
//	e( 7) =  0.074275762552012;
//	e( 8) = -0.045116880454250;
//	e( 9) =  0.223471227762893;
//	e(10) = -0.217812859010447;
//	e(11) =  0.006511492832995;
//	e(12) =  0.110505404222445;
//	e(13) =  0.220122043576173;
//	e(14) =  0.308842379100790;
//	e(15) =  0.017186226635085;
//	e(16) = -0.298318062127522;
//	e(17) = -0.148460367451971;
//	e(18) = -0.212316346663997;
//	e(19) =  0.470091444800408;
//	e(20) = -0.123120376293379;
//	e(21) =  0.149615356740797;
//	e(22) = -0.038483702117653;
//	e(23) =  0.177722085084144;
//	e(24) = -0.152969847313575;
//	e(25) = -0.280453793867752;
//	e(26) = -0.284475185018299;
//	e(27) =  0.097638781971988;
//	e(28) = -0.035475031323765;
//	e(29) = -0.039210697561467;
//	e(30) =  0.283862030128510;
//	e(31) =  0.058316874796837;
//	e(32) =  0.039562210692872;
//	e(33) =  0.317539817994812;
//	e(34) = -0.160893191269909;
//	e(35) =  0.139324883169921;
//	e(36) =  0.167017633014536;
//	e(37) = -0.048743028075590;
//	e(38) =  0.043134017280749;
//	e(39) = -0.233168786296410;
//	e(40) = -0.229590555779719;
//	e(41) =  0.020974943203299;
//	e(42) =  0.144450806445000;
//	e(43) =  0.517098250523248;
//	e(44) = -0.133378134140277;
//	e(45) =  0.037466204915788;
//	e(46) = -0.016498885074191;
//	e(47) = -0.386604583570197;
//	e(48) = -0.087793230786955;
//	e(49) = -0.358935768291025;

	vector_type y(n_obs); // output data
	y( 0) = -0.172730564397743;
	y( 1) =  0.929106536027214;
	y( 2) =  2.721729859290525;
	y( 3) =  2.955437837787684;
	y( 4) =  3.200711456110224;
	y( 5) =  4.495616595332107;
	y( 6) =  3.542193656189184;
	y( 7) =  0.955873025185438;
	y( 8) = -0.695905301110681;
	y( 9) =  0.070474191431718;
	y(10) =  1.642537535060864;
	y(11) =  2.183492965985113;
	y(12) =  2.685894514022425;
	y(13) =  4.111315632764386;
	y(14) =  3.897668705699988;
	y(15) =  3.220950371864441;
	y(16) =  1.349321650864221;
	y(17) = -1.577387844006209;
	y(18) = -2.934126513251799;
	y(19) = -1.144302561099341;
	y(20) =  1.201759627200661;
	y(21) =  4.470405255564788;
	y(22) =  5.151653010189595;
	y(23) =  5.344324694938996;
	y(24) =  5.571941262454530;
	y(25) =  6.024945077687150;
	y(26) =  6.602443372199293;
	y(27) =  7.512226712134671;
	y(28) =  5.956620857363090;
	y(29) =  4.192164677707061;
	y(30) =  1.934590137831653;
	y(31) =  0.233982637508571;
	y(32) = -1.465221398297563;
	y(33) = -3.571978961441038;
	y(34) = -6.302834030479408;
	y(35) = -8.090139734671594;
	y(36) = -7.227711669081171;
	y(37) = -3.866365373807783;
	y(38) = -1.114869320395609;
	y(39) =  1.248100371879758;
	y(40) =  2.164764116069404;
	y(41) =  1.077407655512012;
	y(42) = -1.321665645894808;
	y(43) = -1.859841394981712;
	y(44) = -0.986182363720728;
	y(45) = -0.403120579933179;
	y(46) = -1.494993932113294;
	y(47) = -3.822918949729564;
	y(48) = -3.892371096346635;
	y(49) = -2.010976833927368;

	vector_type y_hat_ok(n_obs);
	y_hat_ok( 0) =   0.000000000000000;
	y_hat_ok( 1) =   0.000000000000000;
	y_hat_ok( 2) =   0.757330453839821;
	y_hat_ok( 3) =   1.845672693276115;
	y_hat_ok( 4) =   4.980084167629195;
	y_hat_ok( 5) =  10.096635798578069;
	y_hat_ok( 6) =   4.268472326468786;
	y_hat_ok( 7) =   0.807795696189755;
	y_hat_ok( 8) =   0.026884719716669;
	y_hat_ok( 9) = - 0.421013019697662;
	y_hat_ok(10) =   2.074543328364649;
	y_hat_ok(11) =   1.770118238708820;
	y_hat_ok(12) =   2.576633133489035;
	y_hat_ok(13) =   3.946697075527104;
	y_hat_ok(14) =   3.789645036962769;
	y_hat_ok(15) =   3.560734192702355;
	y_hat_ok(16) =   1.752272575324057;
	y_hat_ok(17) = - 1.674660948495208;
	y_hat_ok(18) = - 2.741703757275845;
	y_hat_ok(19) = - 1.844812212757481;
	y_hat_ok(20) =   1.820028659763211;
	y_hat_ok(21) =   3.888482636942519;
	y_hat_ok(22) =   5.236789155809018;
	y_hat_ok(23) =   4.986701553377038;
	y_hat_ok(24) =   5.979773652108474;
	y_hat_ok(25) =   6.057476834806140;
	y_hat_ok(26) =   6.559512122541965;
	y_hat_ok(27) =   7.109994502351562;
	y_hat_ok(28) =   6.119448067520287;
	y_hat_ok(29) =   4.240415815960956;
	y_hat_ok(30) =   1.640802192896264;
	y_hat_ok(31) =   0.569268731882630;
	y_hat_ok(32) = - 1.395254609998421;
	y_hat_ok(33) = - 3.812023673296598;
	y_hat_ok(34) = - 5.733670238452477;
	y_hat_ok(35) = - 8.412430426481615;
	y_hat_ok(36) = - 7.161238243045864;
	y_hat_ok(37) = - 3.679388589173820;
	y_hat_ok(38) = - 1.350077861109394;
	y_hat_ok(39) =   1.420699901298591;
	y_hat_ok(40) =   2.120468248206569;
	y_hat_ok(41) =   0.837684697825752;
	y_hat_ok(42) = - 1.341292120127993;
	y_hat_ok(43) = - 2.217005075071560;
	y_hat_ok(44) = - 0.365000007835218;
	y_hat_ok(45) = - 0.684565503255728;
	y_hat_ok(46) = - 1.360603186668722;
	y_hat_ok(47) = - 3.372417010764925;
	y_hat_ok(48) = - 4.181287736722833;
	y_hat_ok(49) = - 1.699384587067436;


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
	const size_type d = 0; // the input delay
	const real_type ff = 0.98; // the forgetting factor

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
	y_hat_ok( 0) =  1.0e+8*0.000000000000000;
	y_hat_ok( 1) = -1.0e+8*0.000000000000000;
	y_hat_ok( 2) = -1.0e+8*0.000000004999787;
	y_hat_ok( 3) =  1.0e+8*0.000000017331537;
	y_hat_ok( 4) =  1.0e+8*0.000000133532910;
	y_hat_ok( 5) = -1.0e+8*0.000000324861683;
	y_hat_ok( 6) =  1.0e+8*0.000000929140723;
	y_hat_ok( 7) = -1.0e+8*0.000002161502316;
	y_hat_ok( 8) =  1.0e+8*0.000005162512849;
	y_hat_ok( 9) = -1.0e+8*0.000012278608966;
	y_hat_ok(10) =  1.0e+8*0.000030023740974;
	y_hat_ok(11) = -1.0e+8*0.000073478189393;
	y_hat_ok(12) =  1.0e+8*0.000177571789534;
	y_hat_ok(13) = -1.0e+8*0.000429886459865;
	y_hat_ok(14) =  1.0e+8*0.001040368406554;
	y_hat_ok(15) = -1.0e+8*0.002517782407545;
	y_hat_ok(16) =  1.0e+8*0.006093359426147;
	y_hat_ok(17) = -1.0e+8*0.014747064123495;
	y_hat_ok(18) =  1.0e+8*0.035689981396793;
	y_hat_ok(19) = -1.0e+8*0.086375139768011;
	y_hat_ok(20) =  1.0e+8*0.209040515112388;
	y_hat_ok(21) = -1.0e+8*0.505909037339871;
	y_hat_ok(22) =  1.0e+8*1.224374493384276;
	y_hat_ok(23) = -1.0e+8*2.963167600857525;
	y_hat_ok(24) =  1.0e+8*7.171303997198738;


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
	size_type d = 0;

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
	size_type d = 0;

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
