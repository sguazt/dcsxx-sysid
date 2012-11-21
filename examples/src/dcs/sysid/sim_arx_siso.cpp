#include <cstddef>
#include <dcs/math/function/sign.hpp>
#include <dcs/math/la/container/dense_vector.hpp>
#include <dcs/math/la/operation/io.hpp>
#include <dcs/math/la/operation/vector_basic_operations.hpp>
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
	typedef ::dcs::math::la::dense_vector<real_type> vector_type;

	const uint_type seed(5489);
	const uint_type num_obs(50);

	::dcs::math::random::mt19937 urng(seed);
	::dcs::math::stats::normal_distribution<real_type> dist;

	// Create a SISO model with ARX structure

	vector_type a(2);
	a(0) = -1.5; a(1) = 0.7;
	vector_type b(3);
	b(0) = 0; b(1) = 1.0; b(2) = 0.5;
	real_type c = 0.3; // noise variance

	::dcs::sysid::darx_siso_model<vector_type,real_type,uint_type> siso_model(a, b, c);

	::std::cout << siso_model << ::std::endl;

	// Simulate a SISO model
	::dcs::math::la::dense_vector<real_type> u(num_obs);
	for (size_type i = 0; i < num_obs; ++i)
	{
		u(i) = ::dcs::math::sign(::dcs::math::stats::rand(dist, urng));
	}

	::dcs::math::la::dense_vector<real_type> e(num_obs);
	for (size_type i = 0; i < num_obs; ++i)
	{
		e(i) = 0.2*::dcs::math::stats::rand(dist, urng);
	}
	vector_type y;
	y = ::dcs::sysid::simulate(siso_model, u);
	::std::cout << "Simulation without noise: " << ::std::endl;
	::std::cout << "\tInput Data: " << u << ::std::endl;
	::std::cout << "\tOutput Data: " << y << ::std::endl;

	y = ::dcs::sysid::simulate(siso_model, u, e);
	::std::cout << "Simulation with noise: " << ::std::endl;
	::std::cout << "\tInput Data: " << u << ::std::endl;
	::std::cout << "\tNoise Data: " << e << ::std::endl;
	::std::cout << "\tOutput Data: " << y << ::std::endl;
}
