#include <cstddef>
#include <dcs/debug.hpp>
#include <dcs/test.hpp>
#include <dcs/math/la/container/dense_matrix.hpp>
#include <dcs/math/la/container/dense_vector.hpp>
#include <dcs/math/la/container/identity_matrix.hpp>
#include <dcs/math/la/operation/io.hpp>
#include <dcs/math/la/operation/matrix_basic_operations.hpp>
#include <dcs/math/la/operation/vector_basic_operations.hpp>
#include <dcs/sysid/model/darx_mimo.hpp>
#include <dcs/sysid/model/darx_siso.hpp>
#include <limits>
#include <string>
#include <sstream>


DCS_TEST_DEF( test_sim_siso_without_noise )
{
	DCS_DEBUG_TRACE("Test Case: SISO system with ARX structure and without noise");

	typedef double real_type;
	typedef ::std::size_t size_type;
	typedef ::std::size_t uint_type;
	typedef ::dcs::math::la::dense_vector<real_type> vector_type;;

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

	::std::string output_data = "[50](\
		 0.00000000e+00, -1.00000000e+00, -3.00000000e+00, -5.30000000e+00,\
		-7.35000000e+00, -6.81500000e+00, -5.57750000e+00, -5.09575000e+00,\
		-5.23937500e+00, -3.79203750e+00, -2.52049375e+00, -6.26314375e-01,\
		 2.32487406e+00,  5.42573116e+00,  6.01118489e+00,  3.71876553e+00,\
		 1.87031887e+00, -2.97657569e-01, -1.25570956e+00, -2.17520404e+00,\
		-1.88380937e+00,  1.96928773e-01,  3.11405972e+00,  6.03323944e+00,\
		 8.37001735e+00,  7.83175842e+00,  4.38862549e+00, -3.99292665e-01,\
		-3.17097684e+00, -4.97696039e+00, -6.74575680e+00, -8.13476293e+00,\
		-8.98011463e+00, -7.27583790e+00, -5.12767660e+00, -2.09842838e+00,\
		1.94173106e+00, 5.88149645e+00, 6.96303293e+00, 4.82750189e+00,\
		2.86712978e+00, 2.42144334e+00, 1.12517417e+00, -1.50724908e+00,\
		-4.54849554e+00, -5.26766896e+00, -3.21755656e+00, 3.61033436e-01,\
		4.29383974e+00, 5.68803621e+00)";
	vector_type y_ok;
	iss.str(output_data);
	iss >> y_ok;

	size_type n_a = 2;
	size_type n_b = 2;
	size_type d = 0;
	size_type n_obs = 50;

	vector_type a(n_a);
	a(0) = -1.5; a(1) = 0.7;
	vector_type b(n_b+1);
	b(0) = 0; b(1) = 1; b(2) = 0.5;
	real_type c = 1;

	::dcs::sysid::darx_siso_model<vector_type,real_type,uint_type> mdl(a, b, c);

	vector_type y;
	y = ::dcs::sysid::simulate(mdl, u);

	for (size_type i = 0; i < n_obs; ++i)
	{
		std::cout << ">> Observation #" << i << std::endl;
		std::cout << ">>" << i << " --> u: " << u(i) << std::endl;
		std::cout << ">>" << i << " --> y: " << y(i) << std::endl;
		std::cout << ">>" << i << " --> y_ok: " << y_ok(i) << std::endl;
		std::cout << "----------------------------------------" << i << std::endl;

		DCS_TEST_CHECK_REL_PRECISION(y(i), y_ok(i), 1.0e-5);
	}
}


DCS_TEST_DEF( test_sim_siso_with_noise )
{
	DCS_DEBUG_TRACE("Test Case: SISO system with ARX structure and with noise");

	typedef double real_type;
	typedef ::std::size_t size_type;
	typedef ::std::size_t uint_type;
	typedef ::dcs::math::la::dense_vector<real_type> vector_type;;

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

	::std::string output_data = "[50](\
		3.08172011e-01, -7.83435071e-01, -2.60553769e+00, -5.16790693e+00,\
		-7.44756779e+00, -6.70051960e+00, -4.96735925e+00, -4.47672376e+00,\
		-4.83765887e+00, -3.74775822e+00, -2.58391940e+00, -7.65667636e-01,\
		2.28781929e+00, 5.48511108e+00, 6.46983161e+00, 4.04020595e+0,\
		1.87973568e+00, -3.89079633e-01, -1.50784520e+00, -2.51841835e+00,\
		-2.55044919e+00, -1.17735471e+00, 1.03416298e+00, 4.25249472e+00,\
		7.01966338e+00, 6.91845702e+00, 4.09606694e+00, -4.53418142e-01,\
		-3.05273815e+00, -4.66705330e+00, -6.24455238e+00, -7.42907418e+00,\
		-8.43570152e+00, -6.83175464e+00, -4.62678156e+00, -1.31040571e+00,\
		2.76153319e+00, 6.39649143e+00, 7.31307887e+00, 5.07658290e+00,\
		3.19134374e+00, 2.59595079e+00, 1.06321960e+00, -1.68193110e+00,\
		-4.91504604e+00, -5.81631514e+00, -3.79248705e+00, 8.98859772e-02,\
		4.30848023e+00, 5.90840184e+00)";
	vector_type y_ok;
	iss.str(output_data);
	iss >> y_ok;

	size_type n_a = 2;
	size_type n_b = 2;
	size_type d = 0;
	size_type n_obs = 50;

	vector_type a(n_a);
	a(0) = -1.5; a(1) = 0.7;
	vector_type b(n_b+1);
	b(0) = 0; b(1) = 1; b(2) = 0.5;
	real_type c = 1;

	::dcs::sysid::darx_siso_model<vector_type,real_type,uint_type> mdl(a, b, c);

	vector_type y;
	y = ::dcs::sysid::simulate(mdl, u, e);

	for (size_type i = 0; i < n_obs; ++i)
	{
		std::cout << ">> Observation #" << i << std::endl;
		std::cout << ">>" << i << " --> u: " << u(i) << std::endl;
		std::cout << ">>" << i << " --> y: " << y(i) << std::endl;
		std::cout << ">>" << i << " --> y_ok: " << y_ok(i) << std::endl;
		std::cout << "----------------------------------------" << i << std::endl;

		DCS_TEST_CHECK_REL_PRECISION(y(i), y_ok(i), 1.0e-5);
	}
}


DCS_TEST_DEF( test_sim_mimo_without_noise )
{
	DCS_DEBUG_TRACE("Test Case: MIMO system with ARX structure and without noise");

	typedef double real_type;
	typedef ::std::size_t size_type;
	typedef ::std::size_t uint_type;
	typedef ::dcs::math::la::dense_matrix<real_type> matrix_type;

	::std::istringstream iss;

	::std::string input_data = "[25,2](\
		(-1.0, -1.0),\
		(-1.0, -1.0),\
		( 1.0, -1.0),\
		(-1.0, -1.0),\
		( 1.0, -1.0),\
		( 1.0,  1.0),\
		( 1.0, -1.0),\
		(-1.0,  1.0),\
		(-1.0,  1.0),\
		(-1.0,  1.0),\
		( 1.0,  1.0),\
		( 1.0,  1.0),\
		(-1.0, -1.0),\
		(-1.0,  1.0),\
		(-1.0, -1.0),\
		(-1.0, -1.0),\
		( 1.0, -1.0),\
		( 1.0,  1.0),\
		( 1.0, -1.0),\
		(-1.0,  1.0),\
		( 1.0, -1.0),\
		(-1.0, -1.0),\
		( 1.0,  1.0),\
		( 1.0,  1.0),\
		(-1.0, -1.0))";
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
	matrix_type Y_ok;
	iss.str(output_data);
	iss >> Y_ok;

	size_type n_a = 2; // the order of the ARX model
	size_type n_y = 2; // the size of output vector
	size_type n_b = 2; // the order of the ARX model
	size_type n_u = 2; // the size of input vector
	size_type d = 0;
	size_type n_obs = 25;

	// Create A1,A2 matrices
	::std::vector<matrix_type> As(n_a);
	matrix_type A(n_y, n_y);
	// Create A1 matrix
	A(0,0) = 1.0; A(0,1) = 2.0;
	A(1,0) = 1.5; A(1,1) = 0.5;
	As[0] = A;
	A.clear();
	// Create A2 matrix
	A(0,0) = 0.1; A(0,1) = 0.2;
	A(1,0) = 0.05; A(1,1) = 0.05;
	As[1] = A;
	A.clear();

	// Create B0,B1,B2 matrices
	::std::vector<matrix_type> Bs(n_b+1);
	matrix_type B(n_y, n_u);
	// Create B0 matrix
	B(0,0) = 1.5; B(0,1) = -1.0;
	B(1,0) = 1.0; B(1,1) = -1.2;
	Bs[0] = B;
	B.clear();
	// Create B1 matrix
	B(0,0) = 0.3; B(0,1) = -0.2;
	B(1,0) = 0.2; B(1,1) = -0.2;
	Bs[1] = B;
	B.clear();
	// Create B2 matrix
	B(0,0) = 0.1; B(0,1) = 0.0;
	B(1,0) = 0.1; B(1,1) = -0.05;
	Bs[2] = B;
	B.clear();

	matrix_type C(::dcs::math::la::identity_matrix<real_type>(n_y, n_y));

	::dcs::sysid::darx_mimo_model<matrix_type,real_type,uint_type> mdl(As.begin(), As.end(), Bs.begin(), Bs.end(), C);

	matrix_type Y;
	Y = ::dcs::sysid::simulate(mdl, U);

	for (size_type i = 0; i < n_obs; ++i)
	{
		std::cout << ">> Observation #" << i << std::endl;
		std::cout << ">>" << i << " --> U: " << ::dcs::math::la::row(U, i) << std::endl;
		std::cout << ">>" << i << " --> Y: " << ::dcs::math::la::row(Y, i) << std::endl;
		std::cout << ">>" << i << " --> Y_ok: " << ::dcs::math::la::row(Y_ok, i) << std::endl;
		std::cout << "----------------------------------------" << i << std::endl;

		for (size_type j = 0; j < n_y; ++j)
		{
			DCS_TEST_CHECK_REL_PRECISION(Y(i,j), Y_ok(i,j), 1.0e-5);
		}
	}
}


DCS_TEST_DEF( test_sim_mimo_with_noise )
{
	DCS_DEBUG_TRACE("Test Case: MIMO system with ARX structure and with noise");

	typedef double real_type;
	typedef ::std::size_t size_type;
	typedef ::std::size_t uint_type;
	typedef ::dcs::math::la::dense_matrix<real_type> matrix_type;

	::std::istringstream iss;

	::std::string input_data = "[25,2](\
		(-1.0, -1.0),\
		(-1.0, -1.0),\
		( 1.0, -1.0),\
		(-1.0, -1.0),\
		( 1.0, -1.0),\
		( 1.0,  1.0),\
		( 1.0, -1.0),\
		(-1.0,  1.0),\
		(-1.0,  1.0),\
		(-1.0,  1.0),\
		( 1.0,  1.0),\
		( 1.0,  1.0),\
		(-1.0, -1.0),\
		(-1.0,  1.0),\
		(-1.0, -1.0),\
		(-1.0, -1.0),\
		( 1.0, -1.0),\
		( 1.0,  1.0),\
		( 1.0, -1.0),\
		(-1.0,  1.0),\
		( 1.0, -1.0),\
		(-1.0, -1.0),\
		( 1.0,  1.0),\
		( 1.0,  1.0),\
		(-1.0, -1.0))";
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
	matrix_type Y_ok;
	iss.str(output_data);
	iss >> Y_ok;

	size_type n_a = 2; // the order of the ARX model
	size_type n_y = 2; // the size of output vector
	size_type n_b = 2; // the order of the ARX model
	size_type n_u = 2; // the size of input vector
	size_type d = 0;
	size_type n_obs = 25;

	// Create A1,A2 matrices
	::std::vector<matrix_type> As(n_a);
	matrix_type A(n_y, n_y);
	// Create A1 matrix
	A(0,0) = 1.0; A(0,1) = 2.0;
	A(1,0) = 1.5; A(1,1) = 0.5;
	As[0] = A;
	A.clear();
	// Create A2 matrix
	A(0,0) = 0.1; A(0,1) = 0.2;
	A(1,0) = 0.05; A(1,1) = 0.05;
	As[1] = A;
	A.clear();

	// Create B0,B1,B2 matrices
	::std::vector<matrix_type> Bs(n_b+1);
	matrix_type B(n_y, n_u);
	// Create B0 matrix
	B(0,0) = 1.5; B(0,1) = -1.0;
	B(1,0) = 1.0; B(1,1) = -1.2;
	Bs[0] = B;
	B.clear();
	// Create B1 matrix
	B(0,0) = 0.3; B(0,1) = -0.2;
	B(1,0) = 0.2; B(1,1) = -0.2;
	Bs[1] = B;
	B.clear();
	// Create B2 matrix
	B(0,0) = 0.1; B(0,1) = 0.0;
	B(1,0) = 0.1; B(1,1) = -0.05;
	Bs[2] = B;
	B.clear();

	// Create the covariance noise matrix
	matrix_type C(n_y, n_y);
	C(0,0) = 0.5; C(0,1) = 0;
	C(1,0) = 0.2; C(1,1) = 1.0;

	::dcs::sysid::darx_mimo_model<matrix_type,real_type,uint_type> mdl(As.begin(), As.end(), Bs.begin(), Bs.end(), C);

	matrix_type Y;
	Y = ::dcs::sysid::simulate(mdl, U, E);

	for (size_type i = 0; i < n_obs; ++i)
	{
		std::cout << ">> Observation #" << i << std::endl;
		std::cout << ">>" << i << " --> U: " << ::dcs::math::la::row(U, i) << std::endl;
		std::cout << ">>" << i << " --> Y: " << ::dcs::math::la::row(Y, i) << std::endl;
		std::cout << ">>" << i << " --> Y_ok: " << ::dcs::math::la::row(Y_ok, i) << std::endl;
		std::cout << "----------------------------------------" << i << std::endl;

		for (size_type j = 0; j < n_y; ++j)
		{
			DCS_TEST_CHECK_REL_PRECISION(Y(i,j), Y_ok(i,j), 1.0e-5);
		}
	}
}


int main()
{
	DCS_TEST_SUITE("Test suite for Recursive Least-Square algorithms");

	DCS_TEST_BEGIN();

	DCS_TEST_DO( test_sim_siso_without_noise );

	DCS_TEST_DO( test_sim_siso_with_noise );

	DCS_TEST_DO( test_sim_mimo_without_noise );

	DCS_TEST_DO( test_sim_mimo_with_noise );

	DCS_TEST_END();
}
