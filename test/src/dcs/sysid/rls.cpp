#include <cstddef>
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
	typedef double real_type;
	typedef ::std::size_t size_type;
	typedef ::std::size_t uint_type;
	typedef ::dcs::math::la::dense_vector<real_type> vector_type;;

	::std::istringstream iss;

	::std::string input_data = "[50](\
		-1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0,\
		 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0,\
		 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0,\
		-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0,\
		 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0)";
	::dcs::math::la::dense_vector<real_type> u;
	iss.str(input_data);
	iss >> u;

//	::std::string output_data = "[50](
//		3.08172011e-01, -7.83435071e-01, -2.60553769e+00, -5.16790693e+00,
//		-7.44756779e+00, -6.70051960e+00, -4.96735925e+00, -4.47672376e+00,
//		-4.83765887e+00, -3.74775822e+00, -2.58391940e+00, -7.65667636e-01,
//		2.28781929e+00, 5.48511108e+00, 6.46983161e+00, 4.04020595e+0,
//		1.87973568e+00, -3.89079633e-01, -1.50784520e+00, -2.51841835e+00,
//		-2.55044919e+00, -1.17735471e+00, 1.03416298e+00, 4.25249472e+00,
//		7.01966338e+00, 6.91845702e+00, 4.09606694e+00, -4.53418142e-01,
//		-3.05273815e+00, -4.66705330e+00, -6.24455238e+00, -7.42907418e+00,
//		-8.43570152e+00, -6.83175464e+00, -4.62678156e+00, -1.31040571e+00,
//		2.76153319e+00, 6.39649143e+00, 7.31307887e+00, 5.07658290e+00,
//		3.19134374e+00, 2.59595079e+00, 1.06321960e+00, -1.68193110e+00,
//		-4.91504604e+00, -5.81631514e+00, -3.79248705e+00, 8.98859772e-02,
//		4.30848023e+00, 5.90840184e+00)";
//	::dcs::math::la::dense_vector<real_type> y;
//	iss.str(output_data);
//	iss >> y;
	::std::string noise_data = "[50](\
		 3.08172011e-01, -2.45693088e-01, 2.85335326e-01, -3.08004951e-01,\
		-1.95837736e-02, 3.53297234e-01, 3.70122695e-01, -2.16048598e-01,\
		-9.97247159e-02, -1.24976540e-01, 1.51356721e-01, -1.32192925e-02,\
		 1.27577162e-01, 1.74148053e-02, 3.43638492e-01, -3.24963713e-01,\
		-1.51691115e-01, 1.19461012e-01, -1.08410776e-01, -2.90062964e-02,\
		-3.28313303e-01, -6.14573765e-01, -4.85119390e-01, 3.77101946e-01,\
		-1.35164607e-01, -1.34291744e-01, 1.32145774e-01, -2.54598641e-01,\
		-5.36407195e-03, 9.46612239e-02, 1.19110858e-01, 1.70817089e-01,\
		-1.63276923e-01, 1.21445715e-01, 2.15859342e-01, 3.47538376e-01,\
		-1.16053388e-02, -1.63092351e-01, 1.51414967e-01, 8.45085859e-02,\
		 1.95624602e-01, -1.37456787e-01, -9.67659712e-02, 4.04050559e-02,\
		-1.47895670e-01, -1.21097851e-01, -8.54656473e-03, 2.07195950e-01,\
		 1.89103320e-02, 8.60167891e-03)";
	::dcs::math::la::dense_vector<real_type> e;
	iss.str(noise_data);
	iss >> e;


	::std::string predicted_data = "[50](\
		0.00000000e+00, -2.90472537e-16, -5.42696205e-01, -3.60209268e+00,\
		-8.86909885e+00, -1.17353736e+00, -6.07629254e+00, -2.62868748e+00,\
		-4.85250223e+00, -3.49510757e+00, -2.53824298e+00, -8.94170741e-01,\
		2.32248948e+00, 5.46467883e+00, 5.91018880e+00, 4.09196821e+00,\
		1.83410521e+00, -3.76704228e-01, -1.50217590e+00, -2.31096875e+00,\
		-2.28206055e+00, -3.75772995e-01, 1.55840973e+00, 3.83479213e+00,\
		7.18146568e+00, 7.15898425e+00, 3.91500990e+00, -1.92748283e-01,\
		-3.17999756e+00, -4.66475634e+00, -6.33950700e+00, -7.55068694e+00,\
		-8.19132085e+00, -6.99584215e+00, -4.70939600e+00, -1.68680954e+00,\
		2.84007950e+00, 6.58058178e+00, 7.17798223e+00, 4.95481254e+00,\
		2.85477284e+00, 2.65344416e+00, 1.15960784e+00, -1.70927254e+00,\
		-4.75317878e+00, -5.73629036e+00, -3.80742853e+00, -9.47086855e-02,\
		4.33997661e+00, 5.97313069e+00)";
	::dcs::math::la::dense_vector<real_type> y_hat_ok;
	iss.str(predicted_data);
	iss >> y_hat_ok;

	size_type n_a = 2;
	size_type n_b = 2;
	size_type d = 0;

	::dcs::math::la::dense_vector<real_type> a(n_a);
	a(0) = -1.5; a(1) = 0.7;
	::dcs::math::la::dense_vector<real_type> b(n_b+1);
	b(0) = 0; b(1) = 1; b(2) = 0.5;
	real_type c = 1;

DCS_DEBUG_TRACE("HERE1");//XXX
	::dcs::sysid::darx_siso_model<vector_type,real_type,uint_type> mdl(a, b, c);
DCS_DEBUG_TRACE("HERE2");//XXX

	::dcs::math::la::dense_vector<real_type> y;
	y = ::dcs::sysid::simulate(mdl, u, e);
DCS_DEBUG_TRACE("HERE3");//XXX

	::dcs::math::la::dense_vector<real_type> theta_hat;
	::dcs::math::la::dense_matrix<real_type> P;
	::dcs::math::la::dense_vector<real_type> phi;

	::dcs::sysid::rls_arx_siso_init(n_a, n_b, d, theta_hat, P, phi);

	std::cout << "\\theta0_hat: " << theta_hat << std::endl;
	std::cout << "P0: " << P << std::endl;
	std::cout << "\\phi0: " << phi << std::endl;

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

		std::cout << ">> Oberservation #" << i << std::endl;
		std::cout << ">>" << i << " --> u: " << u(i) << std::endl;
		std::cout << ">>" << i << " --> y: " << y(i) << std::endl;
		std::cout << ">>" << i << " --> \\theta_hat: " << theta_hat << std::endl;
		std::cout << ">>" << i << " --> P: " << P << std::endl;
		std::cout << ">>" << i << " --> \\phi: " << phi << std::endl;
		std::cout << ">>" << i << " --> \\y_hat: " << y_hat << std::endl;
		std::cout << ">>" << i << " --> \\y_hat_ok: " << y_hat_ok(i) << std::endl;
		std::cout << "----------------------------------------" << i << std::endl;

		DCS_TEST_CHECK_CLOSE(y_hat, y_hat_ok(i), 1.0e-5);
	}
}


DCS_TEST_DEF( test_rarx_siso_with_noise )
{
	typedef double real_type;
	typedef ::std::size_t size_type;

	::std::istringstream iss;

	::std::string input_data = "[50](\
		-1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0,\
		 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0,\
		 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0,\
		-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0,\
		 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0)";
	::dcs::math::la::dense_vector<real_type> u;
	iss.str(input_data);
	iss >> u;

	::std::string noise_data = "[50](\
		 3.08172011e-01, -2.45693088e-01, 2.85335326e-01, -3.08004951e-01,\
		-1.95837736e-02, 3.53297234e-01, 3.70122695e-01, -2.16048598e-01,\
		-9.97247159e-02, -1.24976540e-01, 1.51356721e-01, -1.32192925e-02,\
		 1.27577162e-01, 1.74148053e-02, 3.43638492e-01, -3.24963713e-01,\
		-1.51691115e-01, 1.19461012e-01, -1.08410776e-01, -2.90062964e-02,\
		-3.28313303e-01, -6.14573765e-01, -4.85119390e-01, 3.77101946e-01,\
		-1.35164607e-01, -1.34291744e-01, 1.32145774e-01, -2.54598641e-01,\
		-5.36407195e-03, 9.46612239e-02, 1.19110858e-01, 1.70817089e-01,\
		-1.63276923e-01, 1.21445715e-01, 2.15859342e-01, 3.47538376e-01,\
		-1.16053388e-02, -1.63092351e-01, 1.51414967e-01, 8.45085859e-02,\
		 1.95624602e-01, -1.37456787e-01, -9.67659712e-02, 4.04050559e-02,\
		-1.47895670e-01, -1.21097851e-01, -8.54656473e-03, 2.07195950e-01,\
		 1.89103320e-02, 8.60167891e-03)";
	::dcs::math::la::dense_vector<real_type> e;
	iss.str(noise_data);
	iss >> e;

	::std::string output_data = "[50](\
		3.08172011e-01, -1.09160708e+00, -2.76046821e+00, -5.51905626e+00,\
		-7.50076840e+00, -6.57653320e+00, -5.10135321e+00, -5.06396842e+00,\
		-5.33465697e+00, -4.02566910e+00, -2.54785545e+00, -6.93386131e-01,\
		2.41448742e+00, 5.49429520e+00, 6.40304073e+00, 3.59343521e+00,\
		1.65002460e+00, -3.34208362e-01, -1.41453977e+00, -2.28456713e+00,\
		-2.25766201e+00, -5.79357747e-01, 2.27511850e+00, 6.05753476e+00,\
		8.38442875e+00, 7.91266205e+00, 4.73929755e+00, -3.43519878e-01,\
		-3.05712438e+00, -4.79611708e+00, -6.53081174e+00, -7.86829718e+00,\
		-9.04114939e+00, -7.23503000e+00, -4.96198219e+00, -1.72248411e+00,\
		2.07368953e+00, 5.73429383e+00, 6.96204432e+00, 4.82953595e+00,\
		3.01227191e+00, 2.32155303e+00, 9.53554944e-01, -1.58507504e+00,\
		-4.75275494e+00, -5.48470105e+00, -3.31715096e+00, 5.55087231e-01,\
		4.46464159e+00, 5.83953186e+00)";
	::dcs::math::la::dense_vector<real_type> y;
	iss.str(output_data);
	iss >> y;

	// [th,yh,p,phi] = rarx(z(1,:),[2 2 1],'ff',0.98);

	size_type n_a = 2;
	size_type n_b = 2;
	size_type d = 1;
	//size_type n1 = n_a+n_b+d-1;
	//size_type n2 = n_a+n_b;

	::dcs::math::la::dense_vector<real_type> theta_hat; // (n1);
	::dcs::math::la::dense_matrix<real_type> P; // (n2,n2);
	::dcs::math::la::dense_vector<real_type> phi; // (n2);

	::dcs::sysid::rls_arx_siso_init(n_a, n_b, d, theta_hat, P, phi);

	std::cout << "\\theta0_hat: " << theta_hat << std::endl;
	std::cout << "P0: " << P << std::endl;
	std::cout << "\\phi0: " << phi << std::endl;

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

		std::cout << ">>" << i << " --> \\theta_hat: " << theta_hat << std::endl;
		std::cout << ">>" << i << " --> P: " << P << std::endl;
		std::cout << ">>" << i << " --> \\phi: " << phi << std::endl;
		std::cout << ">>" << i << " --> \\y_hat: " << y_hat << std::endl;
//		std::cout << ">>" << i << " --> \\y_hat_ok: " << y_hat_ok(i) << std::endl;
//
//		DCS_TEST_CHECK_CLOSE(y_hat, y_hat_ok(i), 1.0e-5);
	}

	// plot(1,th(1),'*',1,th(2),'+',1,th(3),'o',1,th(4),'*'),
	// axis([1 50 -2 2]),title('Estimated Parameters'),drawnow
	// hold on;
	// for kkk = 2:50
	//   [th,yh,p,phi] = rarx(z(kkk,:),[2 2 1],'ff',0.98,th',p,phi);
	//   plot(kkk,th(1),'*',kkk,th(2),'+',kkk,th(3),'o',kkk,th(4),'*')
	// end
	// hold off

}


DCS_TEST_DEF( test_rarx_mimo_with_noise )
{
	typedef double real_type;
	typedef ::std::size_t size_type;

	::std::istringstream iss;

	// randn("seed",5489)
	// u = sign(randn(5,10,1)); % input
	::std::string input_data = "[5,10](\
		(-1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0),\
		( 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0),\
		( 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0),\
		(-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0),\
		( 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0))";
	::dcs::math::la::dense_matrix<real_type> U;
	iss.str(input_data);
	iss >> U;

	// e = 0.2*randn(5,10,1);   % noise
	::std::string noise_data = "[5,10](\
		( 3.08172011e-01, -2.45693088e-01, 2.85335326e-01, -3.08004951e-01, -1.95837736e-02, 3.53297234e-01, 3.70122695e-01, -2.16048598e-01, -9.97247159e-02, -1.24976540e-01),\
		( 1.51356721e-01, -1.32192925e-02, 1.27577162e-01, 1.74148053e-02, 3.43638492e-01, -3.24963713e-01, -1.51691115e-01, 1.19461012e-01, -1.08410776e-01, -2.90062964e-02),\
		(-3.28313303e-01, -6.14573765e-01, -4.85119390e-01, 3.77101946e-01, -1.35164607e-01, -1.34291744e-01, 1.32145774e-01, -2.54598641e-01, -5.36407195e-03, 9.46612239e-02),\
		( 1.19110858e-01, 1.70817089e-01, -1.63276923e-01, 1.21445715e-01, 2.15859342e-01, 3.47538376e-01, -1.16053388e-02, -1.63092351e-01, 1.51414967e-01, 8.45085859e-02),\
		( 1.95624602e-01, -1.37456787e-01, -9.67659712e-02, 4.04050559e-02, -1.47895670e-01, -1.21097851e-01, -8.54656473e-03, 2.07195950e-01, 1.89103320e-02, 8.60167891e-03))";
	::dcs::math::la::dense_matrix<real_type> E;
	iss.str(noise_data);
	iss >> E;

	// th0 = idpoly([1 -1.5 0.7],[0 1 0.5],[1 -1 0.2]); % a low order idpoly model
	// y = sim(th0,[u e]);
	// z = iddata(y,u);
	// plot(z) % analysis data object
	//
	::std::string output_data = "[5,10](\
		( 3.08172011e-01, -1.09160708e+00, -2.76046821e+00, -5.51905626e+00, -7.50076840e+00, -6.57653320e+00, -5.10135321e+00, -5.06396842e+00, -5.33465697e+00, -4.02566910e+00),\
		(-2.54785545e+00, -6.93386131e-01, 2.41448742e+00, 5.49429520e+00, 6.40304073e+00, 3.59343521e+00, 1.65002460e+00, -3.34208362e-01, -1.41453977e+00, -2.28456713e+00),\
		(-2.25766201e+00, -5.79357747e-01, 2.27511850e+00, 6.05753476e+00, 8.38442875e+00, 7.91266205e+00, 4.73929755e+00, -3.43519878e-01, -3.05712438e+00, -4.79611708e+00),\
		(-6.53081174e+00, -7.86829718e+00, -9.04114939e+00, -7.23503000e+00, -4.96198219e+00, -1.72248411e+00, 2.07368953e+00, 5.73429383e+00, 6.96204432e+00, 4.82953595e+00),\
		( 3.01227191e+00, 2.32155303e+00, 9.53554944e-01, -1.58507504e+00, -4.75275494e+00, -5.48470105e+00, -3.31715096e+00, 5.55087231e-01, 4.46464159e+00, 5.83953186e+00))";
	::dcs::math::la::dense_matrix<real_type> Y;
	iss.str(output_data);
	iss >> Y;

	// [th,yh,p,phi] = rarx(z(1,:),[2 2 1],'ff',0.98);

	size_type n_a = 2; // the order of the ARX model
	size_type n_y = 10; // the size of output vector
	size_type n_b = 2; // the order of the ARX model
	size_type n_u = 10; // the size of input vector
	size_type d = 1;
	size_type n1 = n_a*n_y+n_b*n_u+d-1;
	size_type n2 = n_a*n_y+n_b*n_u;

	::dcs::math::la::dense_matrix<real_type> theta; //(n_y, n1);
	::dcs::math::la::dense_matrix<real_type> P; //(n2,n2);
	::dcs::math::la::dense_vector<real_type> phi; //(n2);

	::dcs::sysid::rls_arx_mimo_init(n_a, n_b, d, n_y, n_u, theta, P, phi);

	std::cout << "\\theta0: " << theta << std::endl;
	std::cout << "P0: " << P << std::endl;
	std::cout << "\\phi0: " << phi << std::endl;

	for (size_type i = 0; i < 5; ++i)
	{
		::dcs::math::la::dense_vector<real_type> y(::dcs::math::la::row(Y, i));
		::dcs::math::la::dense_vector<real_type> u(::dcs::math::la::row(U, i));
		//::dcs::math::la::dense_vector<real_type> y(n_y);
		//::dcs::math::la::dense_vector<real_type> u(n_u);
		//for (size_type j = 0; j < n_y; ++j)
		//{
		//	y(j) = Y(i,j);
		//}
		//for (size_type j = 0; j < n_u; ++j)
		//{
		//	u(j) = U(i,j);
		//}

		::dcs::sysid::rls_ff_arx_mimo(
			//::dcs::math::la::row(Y, i),
			y,
			//::dcs::math::la::row(U, i),
			u,
			0.98,
			n_a,
			n_b,
			d,
			theta,
			P,
			phi
		);

		std::cout << ">>" << i << " --> \\theta_hat: " << theta << std::endl;
		std::cout << ">>" << i << " --> P: " << P << std::endl;
		std::cout << ">>" << i << " --> \\phi: " << phi << std::endl;
		std::cout << ">>" << i << " --> \\y_hat: " << ::dcs::math::la::prod(theta, phi) << std::endl;
	}

	// plot(1,th(1),'*',1,th(2),'+',1,th(3),'o',1,th(4),'*'),
	// axis([1 50 -2 2]),title('Estimated Parameters'),drawnow
	// hold on;
	// for kkk = 2:50
	//   [th,yh,p,phi] = rarx(z(kkk,:),[2 2 1],'ff',0.98,th',p,phi);
	//   plot(kkk,th(1),'*',kkk,th(2),'+',kkk,th(3),'o',kkk,th(4),'*')
	// end
	// hold off
}


int main()
{
	DCS_TEST_SUITE("Test suite for Recursive Least-Square algorithms");

	DCS_TEST_BEGIN();

	DCS_TEST_DO( test_rarx_siso_without_noise );

//	DCS_TEST_DO( test_rarx_siso_with_noise );

//	DCS_TEST_DO( test_rarx_mimo_with_noise );

	DCS_TEST_END();
}
