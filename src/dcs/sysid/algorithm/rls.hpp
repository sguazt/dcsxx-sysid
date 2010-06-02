/**
 * \file dcs/sysid/algorithm/rls.hpp
 *
 * \brief Recursive Least-Square algorithm.
 *
 * Copyright (C) 2009-2010  Distributed Computing System (DCS) Group, Computer
 * Science Department - University of Piemonte Orientale, Alessandria (Italy).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * \author Marco Guazzone, &lt;marco.guazzone@mfn.unipmn.it&gt;
 */
#ifndef DCS_SYSID_ALGORITHM_RLS_HPP
#define DCS_SYSID_ALGORITHM_RLS_HPP


#include <dcs/assert.hpp>
#include <dcs/debug.hpp>
#include <dcs/math/la/container/identity_matrix.hpp>
#include <dcs/math/la/operation/matrix_basic_operations.hpp>
#include <dcs/math/la/operation/num_columns.hpp>
#include <dcs/math/la/operation/num_rows.hpp>
#include <dcs/math/la/operation/vector_basic_operations.hpp>
#include <dcs/math/la/operation/size.hpp>
#include <dcs/math/la/operation/subrange.hpp>
#include <dcs/math/la/traits/matrix.hpp>
#include <dcs/math/la/traits/vector.hpp>
#include <limits>
#include <stdexcept>


namespace dcs { namespace sysid {

/**
 * \brief Provide initial values for executing the Recursive Least-Square
 *  algorithm for SISO system models.
 *
 * \tparam RealT The type for real numbers.
 * \tparam VectorT The type for vectors.
 * \tparam MatrixT The type for matrices.
 *
 * \param theta0_hat The initial parameter vector that will be populated by this
 *  function.
 * \param P0 The initial covariance matrix that will be populated by this
 *  function.
 * \param phi0 The initial regressor vector that will be populated by this
 *  function.
 * \return Nothing. However vector \a theta0_hat, matrix \a P' and vector
 *  \a phi0 are initialized.
 */
template <
	//typename RealT,
	typename UIntT,
	typename VectorT,
	typename MatrixT
>
void rls_arx_siso_init(UIntT n_a, UIntT n_b, UIntT d, VectorT& theta0_hat, MatrixT& P0, VectorT& phi0)
{
	//typedef RealT real_type;
	typedef typename ::dcs::math::la::matrix_traits<MatrixT>::value_type value_type;
	typedef typename ::dcs::math::la::vector_traits<VectorT>::size_type size_type;

	UIntT n1 = n_a+n_b+1;
	UIntT n2 = n_a+n_b+1+d;

	theta0_hat = VectorT(
						//::dcs::math::la::size(theta0_hat),
						n1,
						//real_type(2.2204e-16)
						value_type(2.2204e-16)
	);
	//P0 = 10000*::dcs::math::la::identity_matrix<value_type>(::dcs::math::la::num_rows(P0));
	P0 = 10000*::dcs::math::la::identity_matrix<value_type>(n2);
	//phi0 = ::dcs::math::la::dense_vector<real_type>(::dcs::math::la::size(phi0), ::std::numeric_limits<real_type>::epsilon());
	phi0 = VectorT(n2, 0);
}


/**
 * \brief Execute one step of the Recursive Least-Square with forgetting factor
 *  algorithm for SISO system models with ARX structure.
 *
 * \tparam RealT The type for real numbers.
 * \tparam UIntT The type for unsigned integral numbers.
 * \tparam VectorT The type for vectors.
 * \tparam MatrixT The type for matrices.
 *
 * \param y The current measurement (output) value.
 * \param u The current regressor (input) value.
 * \param lambda The forgetting factor.
 * \param n_a The memory of the ARX model with respect to the output variables.
 * \param n_b The memory of the ARX model with respect to the input variables.
 * \param d The delay of the ARX model.
 * \param theta_hat The current parameter estimate vector.
 * \param P Covariance matrix.
 * \param theta_hat The current regressor vector.
 * \return The predicted value of the outputs.
 *  Furthermore vector \a \theta_hat, matrix \a P and vector \a phi
 *  are changed in order to reflect the current RLS update step.
 */
template <
	typename RealT,
	typename UIntT,
	typename VectorT,
	typename MatrixT
>
RealT rls_ff_arx_siso(RealT y, RealT u, RealT lambda, UIntT n_a, UIntT n_b, UIntT d, VectorT& theta_hat, MatrixT& P, VectorT& phi)
{
	typedef RealT real_type;
	typedef typename ::dcs::math::la::vector_traits<VectorT>::size_type size_type;

	size_type n = ::dcs::math::la::size(phi);

	DCS_ASSERT(
		n == (n_a+n_b+1+d),
		throw ::std::logic_error("The size of the regression vector must be equal to the sum of the orders of the ARX model.")
	);

	//VectorT phi_t(::dcs::math::la::trans(phi));

	// Compute the Gain
	// L(k+1) = \frac{P(k)\phi(k+1)}{\lambda(k)+\phi^T(k+1)P(k)\phi(k+1)}
	VectorT L(n);
	L = ::dcs::math::la::prod(P, phi)
		/ (
			lambda
			+ ::dcs::math::la::inner_prod(
				//::dcs::math::la::prod(phi_t, P),
				::dcs::math::la::prod(phi, P),
				phi
			)
	);

	// Update the covariance matrix
	// P(k+1) = \frac{1}{\lambda(k)}\left[I-L(k+1)\Phi^T(k+1)\right]P(k)
	P = ::dcs::math::la::prod(
			::dcs::math::la::identity_matrix<real_type>(n)
			-
			//::dcs::math::la::outer_prod(L, phi_t),
			::dcs::math::la::outer_prod(L, phi),
			P
		)
		/ lambda;

	// Compute output estimate
	real_type y_hat = ::dcs::math::la::inner_prod(phi, theta_hat);

	// Update parameters estimate
	// \hat{\theta}(k+1) = \hat{\theta}(k)+L(k+1)(y(k+1)-\Phi^T(k+1)\hat{\theta}(k))
	theta_hat = theta_hat + (y-y_hat)*L;

	// Update the Regression vector
	// \phi(t) &= [-y(t-1) -y(t-2) ... -y(t-n_a) u(t-1) u(t-2) ... u(t-n_b)]^T
	// \phi(t+1) &= [-y(t) -y(t-1) ... -y(t-n_a+1) u(t) u(t-1) ... u(t-n_b+1)]^T
	//          &= [-y(t) \phi(t,1) ... \phi(t,n_a-1) u(t) \phi(t,n_a+1) ... \phi(t,n_a+n_b-1)]
	// \phi(k+2) &= [-y(k+1), -y(k), y(k-1), \ldots, -y(k-n_a+2), u(k+1), \ldots, u(k-n_b+2)]
	//           &= [-y(k+1), \phi_1(k+1), \phi_2(k+1), \ldots, \phi_{n_a-1}(k+1), u(k+1), \ldots, \phi_(k+1)]
	VectorT phi_new(n, 0);
	phi_new(0) = -y;
	::dcs::math::la::subrange(phi_new, 1, n_a) = ::dcs::math::la::subrange(phi, 0, n_a-1);
	phi_new(n_a) = u;
	::dcs::math::la::subrange(phi_new, n_a+1, n_a+n_b) = ::dcs::math::la::subrange(phi, n_a, n_a+n_b-1);
	phi = phi_new;

	return y_hat;
}


/**
 * \brief Provide initial values for executing the Recursive Least-Square
 *  algorithms for MIMO system models.
 *
 * \tparam RealT The type for real numbers.
 * \tparam VectorT The type for vectors.
 * \tparam MatrixT The type for matrices.
 *
 * \param theta0_hat The initial parameter matrix that will be populated by this
 *  function.
 * \param P0 The initial covariance matrix that will be populated by this
 *  function.
 * \param phi0 The initial regressor vector that will be populated by this
 *  function.
 * \return Nothing. However matrix \a theta0_hat, matrix \a P' and vector
 *  \a phi0 are initialized.
 */
template <
	typename UIntT,
	typename VectorT,
	typename MatrixT
>
void rls_arx_mimo_init(UIntT n_a, UIntT n_b, UIntT d, UIntT n_y, UIntT n_u, MatrixT& theta0_hat, MatrixT& P0, VectorT& phi0)
{
	typedef typename ::dcs::math::la::matrix_traits<MatrixT>::value_type value_type;
	typedef typename ::dcs::math::la::vector_traits<VectorT>::size_type size_type;

	//UIntT n1 = n_a+n_b+1-d;
	//UIntT n2 = n_a+n_b;
	size_type n1 = n_a*n_y+(n_b+1)*n_u;
	size_type n2 = n_a*n_y+(n_b+1)*n_u+d;

	theta0_hat = MatrixT(
						//::dcs::math::la::num_rows(theta0_hat),
						n_y,
						//::dcs::math::la::num_columns(theta0_hat),
						n1,
						value_type(2.2204e-16)
	);
	//P0 = 10000*::dcs::math::la::identity_matrix<value_type>(::dcs::math::la::num_rows(P0));
	P0 = 10000*::dcs::math::la::identity_matrix<value_type>(n1);
	////phi0 = ::dcs::math::la::dense_vector<real_type>(::dcs::math::la::size(phi0), ::std::numeric_limits<real_type>::epsilon());
	//phi0 = VectorT(::dcs::math::la::size(phi0), 0);
	phi0 = VectorT(n2, 0);
}


/**
 * \brief Execute one step of the Recursive Least-Square with forgetting factor
 *  algorithm for MIMO system models with ARX structure.
 *
 * \tparam RealT The type for real numbers.
 * \tparam UIntT The type for unsigned integral numbers.
 * \tparam VectorT The type for vectors.
 * \tparam MatrixT The type for matrices.
 *
 * \param y The current measurement (output) vector.
 * \param u The current regressor (input) vector.
 * \param lambda The forgetting factor.
 * \param n_a The memory of the ARX model with respect to the output variables.
 * \param n_b The memory of the ARX model with respect to the input variables.
 * \param d The delay of the ARX model.
 * \param theta_hat The current parameter estimate matrix.
 * \param P Covariance matrix.
 * \param phi The current regression vector.
 * \return Nothing. However matrix \a theta_hat, matrix \a P and vector \a phi
 *  are changed in order to reflect the current RLS update step.
 */
template <
	typename RealT,
	typename UIntT,
	typename VectorT,
	typename MatrixT
>
void rls_ff_arx_mimo(VectorT y, VectorT u, RealT lambda, UIntT n_a, UIntT n_b, UIntT d, MatrixT& theta_hat, MatrixT& P, VectorT& phi)
{
	typedef RealT real_type;
	typedef typename ::dcs::math::la::vector_traits<VectorT>::size_type size_type;

	size_type n = ::dcs::math::la::size(phi);
	size_type n_y = ::dcs::math::la::size(y);
	size_type n_u = ::dcs::math::la::size(u);

	// preconditions
	DCS_ASSERT(
		n == (n_a*n_y+n_b*n_u),
		throw ::std::logic_error("The size of the regression vector must be equal to the sum of the orders of the ARX model.")
	);
	DCS_ASSERT(
		::dcs::math::la::num_rows(P) == n && ::dcs::math::la::num_columns(P) == n,
		throw ::std::logic_error("The size of the covariance matrix must be equal to the sum of the orders of the ARX model.")
	);
	DCS_ASSERT(
		::dcs::math::la::num_rows(theta_hat) == n_y && ::dcs::math::la::num_columns(theta_hat) == n,
		throw ::std::logic_error("The size of the parameter matrix is not compatible with the ARX model.")
	);

	// Compute the Gain
	// L(k+1) = \frac{P(k)\phi(k+1)}{\lambda(k)+\phi^T(k+1)P(k)\phi(k+1)}
	VectorT L(n);
	L = ::dcs::math::la::prod(P, phi)
		/ (
			lambda
			+ ::dcs::math::la::inner_prod(
				::dcs::math::la::prod(phi, P),
				phi
			)
	);

	// Update the covariance matrix
	// P(k+1) = \frac{1}{\lambda(k)}\left[I-L(k+1)\Phi^T(k+1)\right]P(k)
	P = ::dcs::math::la::prod(
			::dcs::math::la::identity_matrix<real_type>(n)
			-
			::dcs::math::la::outer_prod(L, phi),
			P
		)
		/ lambda;

	// Update parameters estimate
	// \hat{\theta}(k+1) = \hat{\theta}(k)+(y(k+1)-\Phi^T(k+1)\hat{\theta}(k))L^T(k+1)
	theta_hat = theta_hat
				+ 
					::dcs::math::la::outer_prod(
						y
						- ::dcs::math::la::prod(
							theta_hat,
							phi
						),
						L
	);

	// Update the Regression vector
	// \phi(t) &= [-y(t-1) -y(t-2) ... -y(t-n_a) u(t-1) u(t-2) ... u(t-n_b)]^T
	// \phi(t+1) &= [-y(t) -y(t-1) ... -y(t-n_a+1) u(t) u(t-1) ... u(t-n_b+1)]^T
	//          &= [-y(t) \phi(t,1) ... \phi(t,n_a-1) u(t) \phi(t,n_a+1) ... \phi(t,n_a+n_b-1)]
	// \phi(k+2) &= [-y(k+1), -y(k), y(k-1), \ldots, -y(k-n_a+2), u(k+1), \ldots, u(k-n_b+2)]
	//           &= [-y(k+1), \phi_1(k+1), \phi_2(k+1), \ldots, \phi_{n_a-1}(k+1), u(k+1), \ldots, \phi_(k+1)]
	VectorT phi_new(n, 0);
	//::dcs::math::la::subrange(phi_new, 0, n_y-1) = -y;
	::dcs::math::la::subrange(phi_new, 0, n_y-1) = ::dcs::math::la::subrange(-y, 0, n_y-1);
	::dcs::math::la::subrange(phi_new, n_y, n_y*n_a-1) = ::dcs::math::la::subrange(phi, 0, n_y*(n_a-1)-1);
	//::dcs::math::la::subrange(phi_new, n_y*(n_a-1), n_y*n_a+n_u) = u;
	::dcs::math::la::subrange(phi_new, n_y*n_a, n_y*n_a+n_u-1) = ::dcs::math::la::subrange(u, 0, n_u-1);
	::dcs::math::la::subrange(phi_new, n_y*n_a+n_u, n_y*n_a+n_u*n_b-1) = ::dcs::math::la::subrange(phi, n_y*n_a, n_y*n_a+n_u*(n_b-1)-1);
	phi = phi_new;
}


}} // Namespace dcs::sysid


#endif // DCS_SYSID_ALGORITHM_RLS_HPP
