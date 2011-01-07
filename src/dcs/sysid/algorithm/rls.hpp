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


#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <boost/numeric/ublasx/operation/size.hpp>
#include <dcs/assert.hpp>
#include <dcs/debug.hpp>
//#include <dcs/math/la/container/identity_matrix.hpp>
//#include <dcs/math/la/operation/matrix_basic_operations.hpp>
//#include <dcs/math/la/operation/num_columns.hpp>
//#include <dcs/math/la/operation/num_rows.hpp>
//#include <dcs/math/la/operation/vector_basic_operations.hpp>
//#include <dcs/math/la/operation/size.hpp>
//#include <dcs/math/la/operation/subrange.hpp>
//#include <dcs/math/la/traits/matrix.hpp>
//#include <dcs/math/la/traits/vector.hpp>
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
	namespace ublas = ::boost::numeric::ublas;

	//typedef RealT real_type;
	typedef typename ublas::matrix_traits<MatrixT>::value_type value_type;
	typedef typename ublas::vector_traits<VectorT>::size_type size_type;

	//UIntT n1 = n_a+n_b+1;
	//UIntT n2 = n_a+n_b+1+d;
	UIntT n = n_a+n_b+1+d;

	theta0_hat = VectorT(
						//n1,
						n,
						//real_type(2.2204e-16)
						value_type(2.22045e-16)
						//::std::numeric_limits<value_type>::min() // alternative initialization
	);
	//P0 = 10000*ublas::identity_matrix<value_type>(n2);
	P0 = 10000*ublas::identity_matrix<value_type>(n);
	//phi0 = VectorT(n2, 0);
	phi0 = VectorT(n, 0);
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
	namespace ublas = ::boost::numeric::ublas;
	namespace ublasx = ::boost::numeric::ublasx;

	typedef RealT real_type;
	typedef typename ublas::vector_traits<VectorT>::size_type size_type;

	size_type n = ublasx::size(phi);

	DCS_ASSERT(
		n == (n_a+n_b+1+d),
		throw ::std::logic_error("The size of the regression vector must be equal to the sum of the orders of the ARX model.")
	);

	//VectorT phi_t(::dcs::math::la::trans(phi));

	// Compute the Gain
	// L(k+1) = \frac{P(k)\phi(k+1)}{\lambda(k)+\phi^T(k+1)P(k)\phi(k+1)}
	VectorT L(n);
	L = ublas::prod(P, phi)
		/ (
			lambda
			+ ublas::inner_prod(
				//ublas::prod(phi_t, P),
				ublas::prod(phi, P),
				phi
			)
	);

	// Update the covariance matrix
	// P(k+1) = \frac{1}{\lambda(k)}\left[I-L(k+1)\Phi^T(k+1)\right]P(k)
	P = ublas::prod(
			ublas::identity_matrix<real_type>(n)
			-
			//ublas::outer_prod(L, phi_t),
			ublas::outer_prod(L, phi),
			P
		)
		/ lambda;

	// Compute output estimate
	real_type y_hat = ublas::inner_prod(phi, theta_hat);

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
	//ublas::subrange(phi_new, 1, n_a) = ublas::subrange(phi, 0, n_a-1);
	ublas::subrange(phi_new, 1, n_a) = ublas::subrange(phi, 0, n_a-1);
	phi_new(n_a) = u;
	//ublas::subrange(phi_new, n_a+1, n_a+n_b) = ublas::subrange(phi, n_a, n_a+n_b-1);
	ublas::subrange(phi_new, n_a+1, n_a+n_b+1) = ublas::subrange(phi, n_a, n_a+n_b);
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
	namespace ublas = ::boost::numeric::ublas;

	typedef typename ublas::matrix_traits<MatrixT>::value_type value_type;
	typedef typename ublas::vector_traits<VectorT>::size_type size_type;

	//size_type n1 = n_a*n_y+(n_b+1)*n_u;
	//size_type n2 = n_a*n_y+(n_b+1)*n_u+d;
	size_type n = n_a*n_y+(n_b+1)*n_u+d;

//	theta0_hat = ublas::zero_matrix<value_type>(n_y, n);
	theta0_hat = ublas::scalar_matrix<value_type>(n_y, n, 2.22045e-16);
//	theta0_hat = ublas::scalar_matrix<value_type>(n_y, n, ::std::numeric_limits<value_type>::min()); // alternative initialization
	//P0 = 10000*::dcs::math::la::identity_matrix<value_type>(n2);
	P0 = 1e+4*ublas::identity_matrix<value_type>(n);
	//phi0 = VectorT(n2, 0);
	//phi0 = VectorT(n, 0);
	phi0 = ublas::zero_vector<value_type>(n);
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
 * \return The output estimate \f$\hat{y}\f$. Furthermore, matrix \a theta_hat,
 *  matrix \a P and vector \a phi are changed in order to reflect the current
 *  RLS update step.
 */
template <
	typename RealT,
	typename UIntT,
	typename VectorT,
	typename MatrixT
>
VectorT rls_ff_arx_mimo(VectorT y, VectorT u, RealT lambda, UIntT n_a, UIntT n_b, UIntT d, MatrixT& theta_hat, MatrixT& P, VectorT& phi)
{
	namespace ublas = ::boost::numeric::ublas;
	namespace ublasx = ::boost::numeric::ublasx;

	typedef RealT real_type;
	typedef VectorT vector_type;
	typedef typename ublas::vector_traits<VectorT>::size_type size_type;

	size_type n = ublasx::size(phi);
	size_type n_y = ublasx::size(y);
	size_type n_u = ublasx::size(u);

	// preconditions
	DCS_ASSERT(
		n == (n_a*n_y+(n_b+1)*n_u+d),
		throw ::std::logic_error("The size of the regression vector must be equal to the sum of the orders of the ARX model.")
	);
	DCS_ASSERT(
		ublasx::num_rows(P) == n && ublasx::num_columns(P) == n,
		throw ::std::logic_error("The size of the covariance matrix must be equal to the sum of the orders of the ARX model.")
	);
	DCS_ASSERT(
		ublasx::num_rows(theta_hat) == n_y && ublasx::num_columns(theta_hat) == n,
		throw ::std::logic_error("The size of the parameter matrix is not compatible with the ARX model.")
	);

	// Compute the Gain
	// L(k+1) = \frac{P(k)\phi(k+1)}{\lambda(k)+\phi^T(k+1)P(k)\phi(k+1)}
	VectorT L(n);
	L = ublas::prod(P, phi)
		/ (
			lambda
			+ ublas::inner_prod(
				ublas::prod(phi, P),
				phi
			)
	);

	// Update the covariance matrix
	// P(k+1) = \frac{1}{\lambda(k)}\left[I-L(k+1)\Phi^T(k+1)\right]P(k)
	P = ublas::prod(
			ublas::identity_matrix<real_type>(n)
			-
			ublas::outer_prod(L, phi),
			P
		)
		/ lambda;

	// Compute output estimate
	vector_type y_hat = ublas::prod(theta_hat, phi);

	// Update parameters estimate
	// \hat{\theta}(k+1) = \hat{\theta}(k)+(y(k+1)-\Phi^T(k+1)\hat{\theta}(k))L^T(k+1)
	theta_hat = theta_hat + ublas::outer_prod(y - y_hat, L);

	// Update the Regression vector
	VectorT phi_new(n, 0);
	ublas::subrange(phi_new, 0, n_y) = ublas::subrange(-y, 0, n_y);
	ublas::subrange(phi_new, n_y, n_y*n_a) = ublas::subrange(phi, 0, n_y*(n_a-1));
	ublas::subrange(phi_new, n_y*n_a, n_y*n_a+n_u) = ublas::subrange(u, 0, n_u);
	ublas::subrange(phi_new, n_y*n_a+n_u, n_y*n_a+n_u*(n_b+1)) = ublas::subrange(phi, n_y*n_a, n_y*n_a+n_u*n_b);
	phi = phi_new;

	return y_hat;
}


}} // Namespace dcs::sysid


#endif // DCS_SYSID_ALGORITHM_RLS_HPP
