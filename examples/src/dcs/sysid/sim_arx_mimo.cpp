#include <cstddef>
#include <dcs/math/function/sign.hpp>
#include <dcs/math/la/container/dense_matrix.hpp>
#include <dcs/math/la/operation/io.hpp>
#include <dcs/math/la/operation/matrix_basic_operations.hpp>
#include <dcs/math/random/mersenne_twister.hpp>
#include <dcs/math/stats/distribution/normal.hpp>
#include <dcs/math/stats/function/rand.hpp>
#include <dcs/sysid/model/darx_mimo.hpp>
#include <iostream>


int main()
{
	typedef double real_type;
	typedef unsigned long uint_type;
	typedef ::std::size_t size_type;
	typedef ::dcs::math::la::dense_matrix<real_type> matrix_type;

	const uint_type seed(5489);
	const uint_type num_obs(50);

	::dcs::math::random::mt19937 urng(seed);
	::dcs::math::stats::normal_distribution<real_type> dist;

	// Create a MIMO ARX model

	::std::vector<matrix_type> As(1);
	As[0] = matrix_type(2,2);
	As[0](0,0) = 0.7; As[0](0,1) = 0.1; As[0](1,0) = 0.2; As[0](1,1) = 0.5;
	//As[1] = matrix_type(2,2);
	//As[1](0,0) = 0.7; As[1](0,1) = -0.3; As[1](1,0) = 0.1; As[1](1,1) = 0.7;

	::std::vector<matrix_type> Bs(2);
	Bs[0] = matrix_type(2,2);
	Bs[0](0,0) = 1.0; Bs[0](0,1) = 1.0; Bs[0](1,0) = 1.0; Bs[0](1,1) = -1.0;
	Bs[1] = matrix_type(2,2);
	Bs[1](0,0) = 0.5; Bs[1](0,1) = 1.2; Bs[1](1,0) = -0.5; Bs[1](1,1) = 1.2;

	// Covariance Matrix
	matrix_type C(2,2);
	C(0,0) = 0.3; C(0,1) = 0.02; C(1,0) = 0.4; C(1,1) = 0.05;

	::dcs::sysid::darx_mimo_model<matrix_type,real_type> mimo_model(
			As.begin(),
			As.end(),
			Bs.begin(),
			Bs.end(),
			C
	);
	::std::cout << mimo_model << ::std::endl;

	// Simulate a MIMO model
	::dcs::math::la::dense_matrix<real_type> U(num_obs, 2);
	for (size_type i = 0; i < num_obs; ++i)
	{
		for (size_type j = 0; j < 2; ++j)
		{
			U(i,j) = ::dcs::math::sign(::dcs::math::stats::rand(dist, urng));
		}
	}

	::dcs::math::la::dense_matrix<real_type> E(num_obs, 2);
	for (size_type i = 0; i < num_obs; ++i)
	{
		for (size_type j = 0; j < 2; ++j)
		{
			E(i,j) = 0.2*::dcs::math::stats::rand(dist, urng);
		}
	}

	matrix_type Y;
	Y = ::dcs::sysid::simulate(mimo_model, U);
	::std::cout << "Simulation without noise: " << ::std::endl;
	::std::cout << "\tInput Data: " << U << ::std::endl;
	::std::cout << "\tOutput Data: " << Y << ::std::endl;

	Y = ::dcs::sysid::simulate(mimo_model, U, E);
	::std::cout << "Simulation with noise: " << ::std::endl;
	::std::cout << "\tInput Data: " << U << ::std::endl;
	::std::cout << "\tNoise Data: " << E << ::std::endl;
	::std::cout << "\tOutput Data: " << Y << ::std::endl;
}
