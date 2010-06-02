#include <cmath>
#include <cstddef>
#include <dcs/math/function/sign.hpp>
#include <dcs/math/la/container/dense_matrix.hpp>
#include <dcs/math/la/container/dense_vector.hpp>
#include <dcs/math/la/container/identity_matrix.hpp>
#include <dcs/math/la/operation/io.hpp>
#include <dcs/math/la/operation/size.hpp>
#include <dcs/math/random/mersenne_twister.hpp>
#include <dcs/math/stats/distribution/normal.hpp>
#include <dcs/math/stats/function/rand.hpp>
#include <dcs/sysid/algorithm/rls.hpp>
#include <dcs/sysid/model/darx_siso.hpp>
#include <iostream>


int main()
{
	typedef double real_type;
	typedef unsigned long uint_type;
	typedef ::std::size_t size_type;
	typedef ::dcs::math::la::dense_matrix<real_type> matrix_type;
	typedef ::dcs::math::la::dense_vector<real_type> vector_type;

	uint_type seed(5489);
	uint_type num_obs(50);
	real_type ff(0.98); // forgetting factor
	uint_type n_a(2);
	uint_type n_b(2);
	uint_type delay(0);

	::dcs::math::random::mt19937 urng(seed);
	::dcs::math::stats::normal_distribution<real_type> dist;

	// Create a SISO model with ARX structure

	vector_type a(n_a);
	a(0) = -1.5; a(1) = 1.0;
	vector_type b(n_b+1);
	b(0) = 0; b(1) = 1.0; b(2) = 0.5;
	real_type c = 1.0; // noise variance

	::dcs::sysid::darx_siso_model<vector_type,real_type,uint_type> siso_model(a, b, c);

	::std::cout << siso_model << ::std::endl;

	// Generate random input data
	::dcs::math::la::dense_vector<real_type> u(num_obs);
	for (size_type i = 0; i < num_obs; ++i)
	{
		u(i) = ::dcs::math::sign(::dcs::math::stats::rand(dist, urng));
	}

	// Generate random noise
	::dcs::math::la::dense_vector<real_type> e(num_obs);
	for (size_type i = 0; i < num_obs; ++i)
	{
		e(i) = 0.2*::dcs::math::stats::rand(dist, urng);
	}


	::std::cout << "RLS with forgetting factor for SISO models:" << ::std::endl;

	vector_type theta_hat;
	matrix_type P;
	vector_type phi;

	::std::cout << "N_A: " << n_a << ::std::endl;
	::std::cout << "N_B: " << n_b << ::std::endl;
	::std::cout << "D: " << delay << ::std::endl;
	::dcs::sysid::rls_arx_siso_init(n_a, n_b, delay, theta_hat, P, phi);

	::std::cout << "\tInput Data: " << u << ::std::endl;
	::std::cout << "\tNoise Data: " << e << ::std::endl;
	::std::cout << "\tInitial Estimated Parameters: " << theta_hat << ::std::endl;
	::std::cout << "\tInitial Covariance Matrix: " << P << ::std::endl;
	::std::cout << "\tInitial Regressor: " << phi << ::std::endl;

	vector_type y;
	y = ::dcs::sysid::simulate(siso_model, u, e);

	for (size_type i = 0; i < num_obs; ++i)
	{
		::std::cout << "\n\tObservation #" << i << ::std::endl;
		::dcs::sysid::rls_ff_arx_siso(y(i), u(i), ff, n_a, n_b, delay, theta_hat, P, phi);
		//real_type y_hat = ::dcs::math::la::inner_prod(theta_hat, phi);

		::std::cout << "\t\tInput Data: " << u(i) << ::std::endl;
		::std::cout << "\t\tOutput Data: " << y(i) << ::std::endl;
		::std::cout << "\t\tEstimated Parameters: " << theta_hat << ::std::endl;
		::std::cout << "\t\tCovariance Matrix: " << P << ::std::endl;
		::std::cout << "\t\tRegressor: " << phi << ::std::endl;
		//::std::cout << "\t\tEstimated Output Data: " << y_hat << ::std::endl;
		//::std::cout << "\t\tRelative Error: " << ::std::abs((y(i)-y_hat)/y(i)) << ::std::endl;
	}
//	::std::cout << "Simulation without noise: " << ::std::endl;
//	::std::cout << "\tInput Data: " << u << ::std::endl;
//	::std::cout << "\tOutput Data: " << y << ::std::endl;
//
//	y = ::dcs::sysid::simulate(siso_model, u, e);
//	::std::cout << "Simulation with noise: " << ::std::endl;
//	::std::cout << "\tInput Data: " << u << ::std::endl;
//	::std::cout << "\tNoise Data: " << e << ::std::endl;
//	::std::cout << "\tOutput Data: " << y << ::std::endl;
}
