#include <cstddef>
#include <dcs/debug.hpp>
#include <dcs/test.hpp>
#include <dcs/math/la/container/dense_matrix.hpp>
#include <dcs/math/la/container/dense_vector.hpp>
#include <dcs/math/la/container/identity_matrix.hpp>
#include <dcs/math/la/operation/io.hpp>
#include <dcs/math/la/operation/matrix_basic_operations.hpp>
#include <dcs/math/la/operation/row.hpp>
#include <dcs/math/la/operation/vector_basic_operations.hpp>
#include <dcs/sysid/algorithm/rls.hpp>
#include <dcs/sysid/model/darx_siso.hpp>
#include <limits>
#include <string>
#include <sstream>


DCS_TEST_DEF( test_rarx_siso_without_noise )
{
	DCS_DEBUG_TRACE("Test Case: SISO system with ARX structure and without noise");

	typedef double real_type;
	typedef ::std::size_t size_type;
	typedef ::std::size_t uint_type;
	typedef ::dcs::math::la::dense_vector<real_type> vector_type;;
	typedef ::dcs::math::la::dense_matrix<real_type> matrix_type;;

	::std::istringstream iss;

	::std::string input_data = "[50](\
		-1.0, -1.0, -1.0, -1.0,  1.0, -1.0, -1.0, -1.0,  1.0, -1.0,\
		 1.0,  1.0,  1.0, -1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0,\
		 1.0,  1.0,  1.0,  1.0, -1.0, -1.0, -1.0,  1.0, -1.0, -1.0,\
		-1.0, -1.0,  1.0, -1.0,  1.0,  1.0,  1.0, -1.0, -1.0,  1.0,\
		 1.0, -1.0, -1.0, -1.0,  1.0,  1.0,  1.0,  1.0, -1.0, -1.0)";
	vector_type u;
	iss.str(input_data);
	iss >> u;

	::std::string predicted_data = "[50](\
		 0.00000000e+00, -1.77635684e-16, -1.87482687e+00, -6.76158312e+00,\
		-9.33497429e+00, -6.48490431e+00, -8.10353276e+00, -7.25463600e+00,\
		-7.33128667e+00, -3.70479002e+00, -2.66923525e+00, -5.40730353e-01,\
		 3.17665229e+00,  6.53866828e+00,  6.70921880e+00,  3.73153275e+00,\
		 2.14101469e+00, -6.81560838e-01, -1.50751747e+00, -2.52346682e+00,\
		-2.08885265e+00,  5.68202209e-01,  3.83961013e+00,  7.08037620e+00,\
		 9.59985704e+00,  8.42692318e+00,  4.16941086e+00, -1.18625935e+00,\
		-3.66484518e+00, -5.67374656e+00, -7.78571294e+00, -9.15708810e+00,\
		-1.00445205e+01, -7.61905364e+00, -5.36124738e+00, -1.87366108e+00,\
		 2.85890152e+00,  7.08638603e+00,  7.71842858e+00,  4.88428852e+00,\
		 3.03170983e+00,  2.69156756e+00,  8.53023155e-01, -2.13190427e+00,\
		-5.40581279e+00, -5.75868960e+00, -3.19464651e+00,  8.38939366e-01,\
		 5.27669084e+00,  6.28545520e+00)";

	vector_type y_hat_ok;
	iss.str(predicted_data);
	iss >> y_hat_ok;

	size_type n_a = 2;
	size_type n_b = 2;
	size_type d = 0;

	vector_type a(n_a);
	a(0) = -1.5; a(1) = 0.7;
	vector_type b(n_b+1);
	b(0) = 0.2; b(1) = 1; b(2) = 0.5;
	real_type c = 1;

	::dcs::sysid::darx_siso_model<vector_type,real_type,uint_type> mdl(a, b, c);

	vector_type y;
	y = ::dcs::sysid::simulate(mdl, u);

	vector_type theta_hat;
	matrix_type P;
	vector_type phi;

	::dcs::sysid::rls_arx_siso_init(n_a, n_b, d, theta_hat, P, phi);

	DCS_DEBUG_TRACE( "theta0_hat: " );
	DCS_DEBUG_TRACE( "P0: " << P );
	DCS_DEBUG_TRACE( "phi0: " << phi );

	for (size_type i = 0; i < 50; ++i)
	{
		real_type y_hat;

		y_hat = ::dcs::sysid::rls_ff_arx_siso(
			y(i),
			u(i),
			0.98,
			n_a,
			n_b,
			d,
			theta_hat,
			P,
			phi
		);

		DCS_DEBUG_TRACE( ">> Observation #" << i );
		DCS_DEBUG_TRACE( ">>" << i << " --> u: " << u(i) );
		DCS_DEBUG_TRACE( ">>" << i << " --> y: " << y(i) );
		DCS_DEBUG_TRACE( ">>" << i << " --> theta_hat: " << theta_hat );
		DCS_DEBUG_TRACE( ">>" << i << " --> P: " << P );
		DCS_DEBUG_TRACE( ">>" << i << " --> phi: " << phi );
		DCS_DEBUG_TRACE( ">>" << i << " --> y_hat: " << y_hat );
		DCS_DEBUG_TRACE( ">>" << i << " --> y_hat_ok: " << y_hat_ok(i) );
		DCS_DEBUG_TRACE( "----------------------------------------" << i );

		DCS_TEST_CHECK_REL_CLOSE(y_hat, y_hat_ok(i), 1.0e-5);
	}
}


DCS_TEST_DEF( test_rarx_siso_with_noise )
{
	DCS_DEBUG_TRACE("Test Case: SISO system with ARX structure and with noise");

	typedef double real_type;
	typedef ::std::size_t size_type;
	typedef ::std::size_t uint_type;
	typedef ::dcs::math::la::dense_vector<real_type> vector_type;;
	typedef ::dcs::math::la::dense_matrix<real_type> matrix_type;;

	::std::istringstream iss;

	::std::string input_data = "[50](\
		-1.0, -1.0, -1.0, -1.0,  1.0, -1.0, -1.0, -1.0,  1.0, -1.0,\
		 1.0,  1.0,  1.0, -1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0,\
		 1.0,  1.0,  1.0,  1.0, -1.0, -1.0, -1.0,  1.0, -1.0, -1.0,\
		-1.0, -1.0,  1.0, -1.0,  1.0,  1.0,  1.0, -1.0, -1.0,  1.0,\
		 1.0, -1.0, -1.0, -1.0,  1.0,  1.0,  1.0,  1.0, -1.0, -1.0)";
	vector_type u;
	iss.str(input_data);
	iss >> u;

	::std::string noise_data = "[50](\
		 3.08172011e-01, -2.45693088e-01,  2.85335326e-01, -3.08004951e-01,\
		-1.95837736e-02,  3.53297234e-01,  3.70122695e-01, -2.16048598e-01,\
		-9.97247159e-02, -1.24976540e-01,  1.51356721e-01, -1.32192925e-02,\
		 1.27577162e-01,  1.74148053e-02,  3.43638492e-01, -3.24963713e-01,\
		-1.51691115e-01,  1.19461012e-01, -1.08410776e-01, -2.90062964e-02,\
		-3.28313303e-01, -6.14573765e-01, -4.85119390e-01,  3.77101946e-01,\
		-1.35164607e-01, -1.34291744e-01,  1.32145774e-01, -2.54598641e-01,\
		-5.36407195e-03,  9.46612239e-02,  1.19110858e-01,  1.70817089e-01,\
		-1.63276923e-01,  1.21445715e-01,  2.15859342e-01,  3.47538376e-01,\
		-1.16053388e-02, -1.63092351e-01,  1.51414967e-01,  8.45085859e-02,\
		 1.95624602e-01, -1.37456787e-01, -9.67659712e-02,  4.04050559e-02,\
		-1.47895670e-01, -1.21097851e-01, -8.54656473e-03,  2.07195950e-01,\
		 1.89103320e-02,  8.60167891e-03)";
	vector_type e;
	iss.str(noise_data);
	iss >> e;

	::std::string predicted_data = "[50](\
		 0.00000000e+00, -2.46063616e-16, -1.09236664e+00, -5.33186515e+00,\
		-9.76430409e+00, -6.59448614e+00, -7.29959234e+00, -4.80425828e+00,\
		-7.31217088e+00, -3.49939760e+00, -2.79786138e+00, -7.87745060e-01,\
		 2.97551727e+00,  6.58324864e+00,  6.51011395e+00,  4.03449109e+00,\
		 2.03239287e+00, -7.50904217e-01, -1.68217957e+00, -2.65047444e+00,\
		-2.45700520e+00,  3.15429968e-03,  2.31532168e+00,  4.86074138e+00,\
		 8.38738479e+00,  7.73768599e+00,  3.70012536e+00, -9.54089593e-01,\
		-3.65288067e+00, -5.30933326e+00, -7.33248885e+00, -8.58374503e+00,\
		-9.26837641e+00, -7.35827069e+00, -4.96048569e+00, -1.47325505e+00,\
		 3.71974308e+00,  7.77947733e+00,  7.89917162e+00,  4.97600810e+00,\
		 2.97371887e+00,  2.95020885e+00,  8.95178931e-01, -2.35457127e+00,\
		-5.61395129e+00, -6.22343142e+00, -3.77123934e+00,  3.91004280e-01,\
		 5.32798474e+00,  6.55663457e+00)";
	vector_type y_hat_ok;
	iss.str(predicted_data);
	iss >> y_hat_ok;

	size_type n_a = 2;
	size_type n_b = 2;
	size_type d = 0;

	vector_type a(n_a);
	a(0) = -1.5; a(1) = 0.7;
	vector_type b(n_b+1);
	b(0) = 0.2; b(1) = 1; b(2) = 0.5;
	real_type c = 1;

	::dcs::sysid::darx_siso_model<vector_type,real_type,uint_type> mdl(a, b, c);

	vector_type y;
	y = ::dcs::sysid::simulate(mdl, u, e);

	vector_type theta_hat; // (n1);
	matrix_type P; // (n2,n2);
	vector_type phi; // (n2);

	::dcs::sysid::rls_arx_siso_init(n_a, n_b, d, theta_hat, P, phi);

	DCS_DEBUG_TRACE( "theta0_hat: " << theta_hat );
	DCS_DEBUG_TRACE( "P0: " << P );
	DCS_DEBUG_TRACE( "phi0: " << phi );

	for (size_type i = 0; i < 50; ++i)
	{
		real_type y_hat;

		y_hat = ::dcs::sysid::rls_ff_arx_siso(
			y(i),
			u(i),
			0.98,
			n_a,
			n_b,
			d,
			theta_hat,
			P,
			phi
		);

		DCS_DEBUG_TRACE( ">> Observation #" << i );
		DCS_DEBUG_TRACE( ">>" << i << " --> u: " << u(i) );
		DCS_DEBUG_TRACE( ">>" << i << " --> e: " << e(i) );
		DCS_DEBUG_TRACE( ">>" << i << " --> y: " << y(i) );
		DCS_DEBUG_TRACE( ">>" << i << " --> theta_hat: " << theta_hat );
		DCS_DEBUG_TRACE( ">>" << i << " --> P: " << P );
		DCS_DEBUG_TRACE( ">>" << i << " --> phi: " << phi );
		DCS_DEBUG_TRACE( ">>" << i << " --> y_hat: " << y_hat );
		DCS_DEBUG_TRACE( ">>" << i << " --> y_hat_ok: " << y_hat_ok(i) );
		DCS_DEBUG_TRACE( "----------------------------------------" << i );

		DCS_TEST_CHECK_REL_CLOSE(y_hat, y_hat_ok(i), 1.0e-5);
	}
}


DCS_TEST_DEF( test_rarx_mimo_without_noise )
{
	DCS_DEBUG_TRACE("Test Case: MIMO system with ARX structure and without noise");

	typedef double real_type;
	typedef ::std::size_t size_type;
	typedef ::dcs::math::la::dense_matrix<real_type> matrix_type;
	typedef ::dcs::math::la::dense_vector<real_type> vector_type;

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

	::std::string predicted_data = "[25,2](\
		( 0.00000000e+00,  0.00000000e+00),\
		(-3.77475828e-16, -3.77475828e-16),\
		(-5.28362121e-01,  8.98215605e-01),\
		( 2.46014762e+00,  2.35695958e+00),\
		( 4.61538118e+00,  4.29160107e+00),\
		(-4.04340586e+01, -2.81085146e+01),\
		( 9.55155197e+01,  7.37459746e+01),\
		(-2.12667095e+02, -1.63293054e+02),\
		( 5.16161553e+02,  3.99154386e+02),\
		(-1.22692809e+03, -9.29371863e+02),\
		( 2.92020217e+03,  2.24467637e+03),\
		(-7.37542693e+03, -5.61222084e+03),\
		( 1.77602343e+04,  1.35373677e+04),\
		(-4.29869252e+04, -3.27660874e+04),\
		( 1.04032024e+05,  7.92982253e+04),\
		(-2.51779536e+05, -1.91914198e+05),\
		( 6.09344058e+05,  4.64462135e+05),\
		(-1.47470287e+06, -1.12405826e+06),\
		( 3.56900037e+06,  2.72040029e+06),\
		(-8.63751155e+06, -6.58375934e+06),\
		( 2.09040430e+07,  1.59336721e+07),\
		(-5.05909106e+07, -3.85618580e+07),\
		( 1.22437446e+08,  9.33253788e+07),\
		(-2.96316747e+08, -2.25861224e+08),\
		( 7.17130404e+08,  5.46617612e+08))";
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
		vector_type y(::dcs::math::la::row(Y, i));
		vector_type u(::dcs::math::la::row(U, i));
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
		DCS_DEBUG_TRACE( ">>" << i << " --> y: " << y );
		DCS_DEBUG_TRACE( ">>" << i << " --> theta_hat: " << theta_hat );
		DCS_DEBUG_TRACE( ">>" << i << " --> P: " << P );
		DCS_DEBUG_TRACE( ">>" << i << " --> phi: " << phi );
		DCS_DEBUG_TRACE( ">>" << i << " --> y_hat: " << y_hat );
		DCS_DEBUG_TRACE( ">>" << i << " --> y_hat_ok: " << ::dcs::math::la::row(Y_hat_ok, i) );
		DCS_DEBUG_TRACE( "----------------------------------------" << i );

		for (size_type j = 0; j < n_y; ++j)
		{
			DCS_TEST_CHECK_REL_CLOSE(y_hat(j), Y_hat_ok(i,j), 1.0e-5);
		}
	}
}


DCS_TEST_DEF( test_rarx_mimo_with_noise )
{
	DCS_DEBUG_TRACE("Test Case: MIMO system with ARX structure and with noise");

	typedef double real_type;
	typedef ::std::size_t size_type;
	typedef ::dcs::math::la::dense_matrix<real_type> matrix_type;
	typedef ::dcs::math::la::dense_vector<real_type> vector_type;

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
		vector_type y(::dcs::math::la::row(Y, i));
		vector_type u(::dcs::math::la::row(U, i));
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
		DCS_DEBUG_TRACE( ">>" << i << " --> e: " << ::dcs::math::la::row(E, i) );
		DCS_DEBUG_TRACE( ">>" << i << " --> y: " << y );
		DCS_DEBUG_TRACE( ">>" << i << " --> theta_hat: " << theta_hat );
		DCS_DEBUG_TRACE( ">>" << i << " --> P: " << P );
		DCS_DEBUG_TRACE( ">>" << i << " --> phi: " << phi );
		DCS_DEBUG_TRACE( ">>" << i << " --> y_hat: " << y_hat );
		DCS_DEBUG_TRACE( ">>" << i << " --> y_hat_ok: " << ::dcs::math::la::row(Y_hat_ok, i) );
		DCS_DEBUG_TRACE( "----------------------------------------" << i );

		for (size_type j = 0; j < n_y; ++j)
		{
			DCS_TEST_CHECK_REL_CLOSE(y_hat(j), Y_hat_ok(i,j), 1.0e-5);
		}
	}
}


int main()
{
	DCS_TEST_SUITE("Test suite for Recursive Least-Square algorithms");

	DCS_TEST_BEGIN();

	DCS_TEST_DO( test_rarx_siso_without_noise );

	DCS_TEST_DO( test_rarx_siso_with_noise );

	DCS_TEST_DO( test_rarx_mimo_without_noise );

	DCS_TEST_DO( test_rarx_mimo_with_noise );

	DCS_TEST_END();
}
