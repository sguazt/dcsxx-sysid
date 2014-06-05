/**
 * \file dcs/sysid/algorithm/rls.hpp
 *
 * \brief Recursive Least-Square algorithm.
 *
 * Recursive Least-Square (RLS) algorithm for different system models:
 * - ARX model
 *   \f[
 *     y(k)+A_1y(k-1)+\cdots+A_{n_a}y(k-n_a) = B_1u(k-1-d)+\cdots+B_{n_b}u(k-n_b-d)
 *   \f]
 *   where \f$n_a\f$ and \f$n_b\f$ are the system orders, and \f$d\f$ is the
 *   input delay (dead time).
 * -
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 *
 * <hr/>
 *
 * Copyright 2009 Marco Guazzone (marco.guazzone@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef DCS_SYSID_ALGORITHM_RLS_HPP
#define DCS_SYSID_ALGORITHM_RLS_HPP


#include <boost/numeric/ublas/expression_types.hpp>
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
#include <limits>
#include <stdexcept>


namespace dcs { namespace sysid {

/**
 * \brief Provide initial values for executing the Recursive Least-Square
 *  algorithms for MISO system models.
 *
 * \tparam RealT The type for real numbers.
 * \tparam ThetaVectorT The type of the initial parameters vector.
 * \tparam PMatrixT The type of the scaled covariance matrix of the parameters.
 * \tparam PhiVectorT The type of the initial data vector.
 *
 * \param theta0_hat The initial parameter vector that will be populated by this
 *  function.
 * \param P0 The initial covariance matrix that will be populated by this
 *  function.
 * \param phi0 The initial regressor vector that will be populated by this
 *  function.
 * \return Nothing. However vector \a theta0_hat, matrix \a P and vector
 *  \a phi0 are initialized.
 */
template <
	typename UIntT,
	typename ThetaVectorT,
	typename PMatrixT,
	typename PhiVectorT
>
void rls_arx_miso_init(UIntT n_a,
					   UIntT n_b,
					   UIntT d,
					   UIntT n_u,
					   ::boost::numeric::ublas::vector_container<ThetaVectorT>& theta0_hat,
					   ::boost::numeric::ublas::matrix_container<PMatrixT>& P0,
					   ::boost::numeric::ublas::vector_container<PhiVectorT>& phi0)
{
	namespace ublas = ::boost::numeric::ublas;

	typedef typename ublas::promote_traits<
					typename ublas::promote_traits<
							typename ublas::vector_traits<ThetaVectorT>::value_type,
							typename ublas::matrix_traits<PMatrixT>::value_type
						>::promote_type,
					typename ublas::vector_traits<PhiVectorT>::value_type
				>::promote_type value_type;
	typedef typename ublas::promote_traits<
					typename ublas::promote_traits<
							typename ublas::vector_traits<ThetaVectorT>::size_type,
							typename ublas::matrix_traits<PMatrixT>::size_type
						>::promote_type,
					typename ublas::vector_traits<PhiVectorT>::size_type
				>::promote_type size_type;

//	// pre: d > 0
//	DCS_ASSERT(
//		d > 0,
//		throw ::std::invalid_argument("[dcs::sysid::rls_arx_miso_init] Input delay must be greater than zero.")
//	);
	// pre: d > 0
	DCS_ASSERT(
		d > 0,
		throw ::std::invalid_argument("[dcs::sysid::rls_ff_arx_miso] The input delay cannot be less than 1.")
	);

	--d;

////	size_type n = n_a*n_y+(n_b+1)*n_u+d;
////	size_type n = n_a+(n_b+d+1)*n_u;
	const size_type n(n_a+n_b*n_u);
	const size_type n_phi(n_a+(n_b+d)*n_u);

	theta0_hat() = ublas::scalar_vector<value_type>(n, ::std::numeric_limits<value_type>::epsilon());
	P0() = 1.0e+4*ublas::identity_matrix<value_type>(n,n);
	phi0() = ublas::zero_vector<value_type>(n_phi);
}


/**
 * \brief Execute one step of the Recursive Least-Square with Forgetting Factor
 *  algorithm for MISO system models with ARX structure.
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
	typename UVectorExprT,
	typename UIntT,
	typename ThetaVectorExprT,
	typename PMatrixExprT,
	typename PhiVectorExprT
>
RealT rls_ff_arx_miso(RealT y,
					  ::boost::numeric::ublas::vector_expression<UVectorExprT> const& u,
					  RealT lambda,
					  UIntT n_a,
					  UIntT n_b,
					  UIntT d,
					  ::boost::numeric::ublas::vector_expression<ThetaVectorExprT>& theta_hat,
					  ::boost::numeric::ublas::matrix_expression<PMatrixExprT>& P,
					  ::boost::numeric::ublas::vector_expression<PhiVectorExprT>& phi)
{
	namespace ublas = ::boost::numeric::ublas;
	namespace ublasx = ::boost::numeric::ublasx;

	typedef typename ublas::promote_traits<
					typename ublas::promote_traits<
							typename ublas::promote_traits<
									typename ublas::promote_traits<
											RealT,
											typename ublas::vector_traits<UVectorExprT>::value_type
										>::promote_type,
									typename ublas::vector_traits<ThetaVectorExprT>::value_type
								>::promote_type,
							typename ublas::matrix_traits<PMatrixExprT>::value_type
						>::promote_type,
					typename ublas::vector_traits<PhiVectorExprT>::value_type
				>::promote_type value_type;
	typedef typename ublas::promote_traits<
					typename ublas::promote_traits<
							typename ublas::promote_traits<
									typename ublas::promote_traits<
											UIntT,
											typename ublas::vector_traits<UVectorExprT>::size_type
										>::promote_type,
									typename ublas::vector_traits<ThetaVectorExprT>::size_type
								>::promote_type,
							typename ublas::matrix_traits<PMatrixExprT>::size_type
						>::promote_type,
					typename ublas::vector_traits<PhiVectorExprT>::size_type
				>::promote_type size_type;
	typedef ublas::vector<value_type> work_vector_type;

	const size_type n(ublasx::size(theta_hat));
	const size_type n_phi(ublasx::size(phi));
	const size_type n_u(ublasx::size(u));

	// pre: d > 0
	DCS_ASSERT(
		d > 0,
		throw ::std::invalid_argument("[dcs::sysid::rls_ff_arx_miso] The input delay cannot be less than 1.")
	);
	// pre: size(theta_hat) == n_a+n_b*n_u
	DCS_ASSERT(
		n == (n_a+n_b*n_u),
		throw ::std::invalid_argument("[dcs::sysid::rls_ff_arx_miso] The parameter vector has an invalid size.")
	);
	// pre: P is a square matrix of order n
	DCS_ASSERT(
		ublasx::num_rows(P) == n && ublasx::num_columns(P) == n,
		throw ::std::invalid_argument("[dcs::sysid::rls_ff_arx_miso] The covariance matrix has an invalid size.")
	);
	// pre: size(phi) == n_a+(n_b+d)*n_u
	DCS_ASSERT(
		//n_phi == (n_a+(n_b+d)*n_u),
		n_phi == (n_a+(n_b+d-1)*n_u),
		throw ::std::invalid_argument("[dcs::sysid::rls_ff_arx_miso] The regression vector has an invalid size.")
	);

	--d;

	// Create an auxiliary regression vector which takes into consideration the
	// actual input delay d.
	work_vector_type aux_phi;
	if (d > 0)
	{
		aux_phi.resize(n, false);
		ublas::subrange(aux_phi, 0, n_a) = ublas::subrange(phi(), 0, n_a);
		//ublas::subrange(aux_phi, n_a, n) = ublas::subrange(phi(), n_a+d*n_u, n_phi);
		for (size_type i = n_a; i < n; i += n_b+d)
		{
			size_type k(i+n_b);
			ublas::subrange(aux_phi, i, k) = ublas::subrange(phi(), i+d, k+d);
		}
	}
	else
	{
		aux_phi = phi;
	}

	// Compute the Gain
	// l(k+1) = \frac{P(k)\phi(k+1)}{\lambda(k)+\phi^T(k+1)P(k)\phi(k+1)}
	work_vector_type l(n);
	l = ublas::prod(P, aux_phi)
		/ (
			lambda
			+ ublas::inner_prod(
				ublas::prod(aux_phi, P),
				aux_phi
			)
	);
//	l = ublas::prec_prod(P, aux_phi)
//		/ (
//			lambda
//			+ ublas::inner_prod(
//				ublas::prec_prod(aux_phi, P),
//				aux_phi
//			)
//	);

	// Update the covariance matrix
	// P(k+1) = \frac{1}{\lambda(k)}\left[P(k)-l(k+1)\Phi^T(k+1)P(k)\right]
//	P() = (P - ublas::prod(ublas::outer_prod(l, aux_phi), P)) / lambda;
	//value_type check_cov((value_type(1)/(lambda+ublas::inner_prod(ublas::prod(aux_phi, P), aux_phi)))*(::std::pow(ublas::inner_prod(theta_hat, aux_phi),2)));
	//bool safe_cov(check_cox >= 0.001);
//	P() = ublas::prod(
//			ublas::identity_matrix<value_type>(n)
//			-
//			ublas::outer_prod(l, aux_phi),
//			P
//		)
//		/ lambda;
	P() = (P - ublas::prod(ublas::outer_prod(l, aux_phi), P)) / lambda;
//	P() = (P - ublas::prec_prod(ublas::outer_prod(l, aux_phi), P)) / lambda;
	// TODO: Should we use this trick
	// Apply the Bittanti's correction.
	//   S. Bittanti, P. Bolzern, and M. Campi.
	//   Exponential convergence of a modified directional forgetting identification algorithm.
	//   Systems & Control Letters, 14:131–137, 1990.
	//P() = P()+0.01*ublas::identity_matrix<value_type>(n);

	// Compute output estimate
	value_type y_hat = ublas::inner_prod(aux_phi, theta_hat);

	// Update parameters estimate
	// \hat{\theta}(k+1) = \hat{\theta}(k)+(y(k+1)-\Phi^T(k+1)\hat{\theta}(k))l^T(k+1)
	theta_hat() = theta_hat + l*(y-y_hat);

	// Clean-up unused memory
	aux_phi.resize(0, false);
	l.resize(0, false);

	// Update the Regression vector
	work_vector_type phi_new(n_phi, 0);
	phi_new(0) = -y;
	// phi = [y(k-1) ... y(k-n_a) u_1(k-1) ... u_{n_u}(k-1) ... u_1(k-1-d) ... u_{n_u}(k-1-d) ... u_1(k-n_b-d) ... u_{n_u}(k-n_b-d)]^T
//	ublas::subrange(phi_new, 1, n_a) = ublas::subrange(phi(), 0, n_a-1);
//	ublas::subrange(phi_new, n_a, n_a+n_u) = u;
//	ublas::subrange(phi_new, n_a+n_u, n_phi) = ublas::subrange(phi(), n_a, n_phi-n_u);

	// MATLAB uses this convention for the regression vector:
	// phi = [y(k-1) ... y(k-n_a) u_1(k-1) ... u_1(k-1-d) ... u_1(k-n_b-d) ... u_{n_u}(k-1) ... u_{n_u}(k-1-d) ... u_{n_u}(k-n_b-d)]^T
	ublas::subrange(phi_new, 1, n_a) = ublas::subrange(phi(), 0, n_a-1);
//	ublas::subslice(phi_new, n_a, n_b, n_u) = u;
	ublas::subslice(phi_new, n_a, n_b+d, n_u) = u;
//	ublas::subslice(phi_new, n_a+1, n_b, (n_b+d-1)*n_u) = ublas::subslice(phi(), n_a, n_b, (n_b+d-1)*n_u);
	//for (size_type i = n_a; i < (n_b+d-1)*n_u; i += n_b+d)
	for (size_type i = n_a; i < n_phi; i += n_b+d)
	{
		size_type k(i+n_b+d-1);
		ublas::subrange(phi_new, i+1, k+1) = ublas::subrange(phi(), i, k);
	}
	phi() = phi_new;
//DCS_DEBUG_TRACE("[rarx_miso] phi(k) = " << phi);//XXX

	return y_hat;
}


/**
 * \brief Execute one step of the Recursive Least-Square with Directional
 *  Forgetting algorithm for MISO system models with ARX structure.
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
 * \param delta A multiplication factor used for applying the
 *  Bittanti's correction [2]. The correction is applied only for values
 *  of \a delta greater than zero.
 *
 * This variant of the RLS algorithm uses a time-varying forgetting factor and
 * it is based on [1,2].
 *
 * References:
 * -# R. Kulhavy, M. Karny.
 *    "Tracking of Slowly Varying Parameters by Directional Forgetting",
 *    IFAC Proc. Ser. 1985, 687-692.
 * -# S. Bittanti, P. Bolzern, M. Campi.
 *    "Exponential Convergence of a Modified Directional Forgetting
 *     Identification Algorithm",
 *    Syst. Control Lett. 1990, 14, 131-137.
 * .
 */
template <
	typename RealT,
	typename UVectorExprT,
	typename UIntT,
	typename ThetaVectorExprT,
	typename PMatrixExprT,
	typename PhiVectorExprT
>
RealT rls_bittanti1990_arx_miso(RealT y,
								::boost::numeric::ublas::vector_expression<UVectorExprT> const& u,
								RealT lambda,
								UIntT n_a,
								UIntT n_b,
								UIntT d,
								::boost::numeric::ublas::vector_expression<ThetaVectorExprT>& theta_hat,
								::boost::numeric::ublas::matrix_expression<PMatrixExprT>& P,
								::boost::numeric::ublas::vector_expression<PhiVectorExprT>& phi,
								RealT delta = RealT/*zero*/())
{
	namespace ublas = ::boost::numeric::ublas;
	namespace ublasx = ::boost::numeric::ublasx;

	typedef typename ublas::promote_traits<
					typename ublas::promote_traits<
							typename ublas::promote_traits<
									typename ublas::promote_traits<
											RealT,
											typename ublas::vector_traits<UVectorExprT>::value_type
										>::promote_type,
									typename ublas::vector_traits<ThetaVectorExprT>::value_type
								>::promote_type,
							typename ublas::matrix_traits<PMatrixExprT>::value_type
						>::promote_type,
					typename ublas::vector_traits<PhiVectorExprT>::value_type
				>::promote_type value_type;
	typedef typename ublas::promote_traits<
					typename ublas::promote_traits<
							typename ublas::promote_traits<
									typename ublas::promote_traits<
											UIntT,
											typename ublas::vector_traits<UVectorExprT>::size_type
										>::promote_type,
									typename ublas::vector_traits<ThetaVectorExprT>::size_type
								>::promote_type,
							typename ublas::matrix_traits<PMatrixExprT>::size_type
						>::promote_type,
					typename ublas::vector_traits<PhiVectorExprT>::size_type
				>::promote_type size_type;
	typedef ublas::vector<value_type> work_vector_type;

	const size_type n(ublasx::size(theta_hat));
	const size_type n_phi(ublasx::size(phi));
	const size_type n_u(ublasx::size(u));

	// pre: d > 0
	DCS_ASSERT(
			d > 0,
			throw ::std::invalid_argument("[dcs::sysid::rls_df_arx_miso] The input delay cannot be less than 1.")
		);
	// pre: size(theta_hat) == n_a+n_b*n_u
	DCS_ASSERT(
			n == (n_a+n_b*n_u),
			throw ::std::invalid_argument("[dcs::sysid::rls_df_arx_miso] The parameter vector has an invalid size.")
		);
	// pre: P is a square matrix of order n
	DCS_ASSERT(
			ublasx::num_rows(P) == n && ublasx::num_columns(P) == n,
			throw ::std::invalid_argument("[dcs::sysid::rls_df_arx_miso] The covariance matrix has an invalid size.")
		);
	// pre: size(phi) == n_a+(n_b+d)*n_u
	DCS_ASSERT(
			//n_phi == (n_a+(n_b+d)*n_u),
			n_phi == (n_a+(n_b+d-1)*n_u),
			throw ::std::invalid_argument("[dcs::sysid::rls_df_arx_miso] The regression vector has an invalid size.")
		);
	// pre: delta >= 0
	DCS_ASSERT(
			delta >= 0,
			throw ::std::invalid_argument("[dcs::sysid::rls_df_arx_miso] The Bittanti's correction factor must be greater than or equal to 0.")
		);

	--d;

	// Create an auxiliary regression vector which takes into consideration the
	// actual input delay d.
	work_vector_type aux_phi;
	if (d > 0)
	{
		aux_phi.resize(n, false);
		ublas::subrange(aux_phi, 0, n_a) = ublas::subrange(phi(), 0, n_a);
		//ublas::subrange(aux_phi, n_a, n) = ublas::subrange(phi(), n_a+d*n_u, n_phi);
		for (size_type i = n_a; i < n; i += n_b+d)
		{
			size_type k(i+n_b);
			ublas::subrange(aux_phi, i, k) = ublas::subrange(phi(), i+d, k+d);
		}
	}
	else
	{
		aux_phi = phi;
	}

	// Compute output estimate
	value_type y_hat = ublas::inner_prod(aux_phi, theta_hat);

	// Compute the prediction error
	value_type epsi = y-y_hat;

	// Compute r(k+1) = \phi^T(k+1)P(k)\phi(k+1)
	value_type r = ublas::inner_prod(
			ublas::prod(aux_phi, P),
			aux_phi
		);

	// Compute the Gain
	// l(k+1) = \frac{P(k)\phi(k+1)}{1+r(k+1)}
	work_vector_type l = ublas::prod(P, aux_phi) / (1+r);

	// Update parameters estimate
	// \hat{\theta}(k+1) = \hat{\theta}(k)+(y(k+1)-\Phi^T(k+1)\hat{\theta}(k))l^T(k+1)
	theta_hat() = theta_hat + l*epsi;

	// Compute the Directional Forgetting factor
	// \beta(k+1) = \begin{cases}
	//               \lambda-\frac{1-\lambda}{r(k+1)}, & r(k+1)>0
	//               1, & r(k+1)=0
	//              \end{cases}
	value_type beta = (r > 0) ? (lambda-(1-lambda)/r) : static_cast<value_type>(1);

	// Update the covariance matrix
	// P(k+1) = P(k)-\frac{P(k)\phi^T(k+1)\phi(k+1)P(k)}{\beta(k+1)^{-1}+r(k+1)}+\delta I
	//P() = P - ublas::prod(ublas::outer_prod(ublas::prod(P, aux_phi), aux_phi), P) / (static_cast<value_type>(1)/beta+r);
	work_vector_type t = ublas::prod(P, aux_phi);
	P() = P - ublas::prod(ublas::outer_prod(t, aux_phi), P) / (static_cast<value_type>(1)/beta+r);
	if (delta > 0)
	{
		// Apply the Bittanti's correction.
		P() += delta*ublas::identity_matrix<value_type>(n);
	}

	// Clean-up unused memory
	t.resize(0, false);
	l.resize(0, false);
	aux_phi.resize(0, false);

	// Update the Regression vector
	work_vector_type phi_new(n_phi, 0);
	phi_new(0) = -y;
	// phi = [y(k-1) ... y(k-n_a) u_1(k-1) ... u_{n_u}(k-1) ... u_1(k-1-d) ... u_{n_u}(k-1-d) ... u_1(k-n_b-d) ... u_{n_u}(k-n_b-d)]^T
//	ublas::subrange(phi_new, 1, n_a) = ublas::subrange(phi(), 0, n_a-1);
//	ublas::subrange(phi_new, n_a, n_a+n_u) = u;
//	ublas::subrange(phi_new, n_a+n_u, n_phi) = ublas::subrange(phi(), n_a, n_phi-n_u);

	// MATLAB uses this convention for the regression vector:
	// phi = [y(k-1) ... y(k-n_a) u_1(k-1) ... u_1(k-1-d) ... u_1(k-n_b-d) ... u_{n_u}(k-1) ... u_{n_u}(k-1-d) ... u_{n_u}(k-n_b-d)]^T
	ublas::subrange(phi_new, 1, n_a) = ublas::subrange(phi(), 0, n_a-1);
//	ublas::subslice(phi_new, n_a, n_b, n_u) = u;
	ublas::subslice(phi_new, n_a, n_b+d, n_u) = u;
//	ublas::subslice(phi_new, n_a+1, n_b, (n_b+d-1)*n_u) = ublas::subslice(phi(), n_a, n_b, (n_b+d-1)*n_u);
	//for (size_type i = n_a; i < (n_b+d-1)*n_u; i += n_b+d)
	for (size_type i = n_a; i < n_phi; i += n_b+d)
	{
		size_type k(i+n_b+d-1);
		ublas::subrange(phi_new, i+1, k+1) = ublas::subrange(phi(), i, k);
	}
	phi() = phi_new;
//DCS_DEBUG_TRACE("[rarx_miso] phi(k) = " << phi);//XXX

	return y_hat;
}


/**
 * \brief Execute one step of the Recursive Least-Square with Directional
 *  Forgetting algorithm for MISO system models with ARX structure.
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
 *
 * This variant of the RLS algorithm uses a time-varying forgetting factor and
 * it is based on [1].
 *
 * References:
 * -# R. Kulhavy, M. Karny.
 *    "Tracking of Slowly Varying Parameters by Directional Forgetting",
 *    IFAC Proc. Ser. 1985, 687-692.
 * .
 */
template <
	typename RealT,
	typename UVectorExprT,
	typename UIntT,
	typename ThetaVectorExprT,
	typename PMatrixExprT,
	typename PhiVectorExprT
>
RealT rls_kulhavy1984_arx_miso(RealT y,
							   ::boost::numeric::ublas::vector_expression<UVectorExprT> const& u,
							   RealT lambda,
							   UIntT n_a,
							   UIntT n_b,
							   UIntT d,
							   ::boost::numeric::ublas::vector_expression<ThetaVectorExprT>& theta_hat,
							   ::boost::numeric::ublas::matrix_expression<PMatrixExprT>& P,
							   ::boost::numeric::ublas::vector_expression<PhiVectorExprT>& phi)
{
	return rls_bittanti1990_arx_miso(y, u, lambda, n_a, n_b, d, theta_hat, P, phi, static_cast<RealT>(0));
}


/**
 * \brief Execute one step of the Exponential Weighting Recursive Least-Square
 *  algorithm for MISO system models with ARX structure.
 *
 * \tparam RealT The type for real numbers.
 * \tparam UIntT The type for unsigned integral numbers.
 * \tparam VectorT The type for vectors.
 * \tparam MatrixT The type for matrices.
 *
 * \param y The current measurement (output) vector.
 * \param u The current regressor (input) vector.
 * \param lambda The minimum value for the forgetting factor.
 * \param rho A design parameter (see [1]).
 * \param n_a The memory of the ARX model with respect to the output variables.
 * \param n_b The memory of the ARX model with respect to the input variables.
 * \param d The delay of the ARX model.
 * \param theta_hat The current parameter estimate matrix.
 * \param P Covariance matrix.
 * \param phi The current regression vector.
 * \return The output estimate \f$\hat{y}\f$. Furthermore, matrix \a theta_hat,
 *  matrix \a P and vector \a phi are changed in order to reflect the current
 *  RLS update step.
 *
 * This variant of the RLS algorithm uses a time-varying forgetting factor and
 * it is based on [1,2].
 *
 * References:
 * -# D.J. Park, B.E. Jun, J.H. Kim,
 *    "Fast Tracking RLS Algorithm using Novel Variable Forgetting Factor with Unity Zone",
 *    Elettron. Lett. 1991, 27, 2150-2151.
 * .
 */
template <
	typename RealT,
	typename UVectorExprT,
	typename UIntT,
	typename ThetaVectorExprT,
	typename PMatrixExprT,
	typename PhiVectorExprT
>
RealT rls_park1991_arx_miso(RealT y,
							::boost::numeric::ublas::vector_expression<UVectorExprT> const& u,
							RealT lambda,
							RealT rho,
							UIntT n_a,
							UIntT n_b,
							UIntT d,
							::boost::numeric::ublas::vector_expression<ThetaVectorExprT>& theta_hat,
							::boost::numeric::ublas::matrix_expression<PMatrixExprT>& P,
							::boost::numeric::ublas::vector_expression<PhiVectorExprT>& phi)
{
	namespace ublas = ::boost::numeric::ublas;
	namespace ublasx = ::boost::numeric::ublasx;

	typedef typename ublas::promote_traits<
					typename ublas::promote_traits<
							typename ublas::promote_traits<
									typename ublas::promote_traits<
											RealT,
											typename ublas::vector_traits<UVectorExprT>::value_type
										>::promote_type,
									typename ublas::vector_traits<ThetaVectorExprT>::value_type
								>::promote_type,
							typename ublas::matrix_traits<PMatrixExprT>::value_type
						>::promote_type,
					typename ublas::vector_traits<PhiVectorExprT>::value_type
				>::promote_type value_type;
	typedef typename ublas::promote_traits<
					typename ublas::promote_traits<
							typename ublas::promote_traits<
									typename ublas::promote_traits<
											UIntT,
											typename ublas::vector_traits<UVectorExprT>::size_type
										>::promote_type,
									typename ublas::vector_traits<ThetaVectorExprT>::size_type
								>::promote_type,
							typename ublas::matrix_traits<PMatrixExprT>::size_type
						>::promote_type,
					typename ublas::vector_traits<PhiVectorExprT>::size_type
				>::promote_type size_type;
	typedef ublas::vector<value_type> work_vector_type;

	const size_type n(ublasx::size(theta_hat));
	const size_type n_phi(ublasx::size(phi));
	const size_type n_u(ublasx::size(u));

	// pre: d > 0
	DCS_ASSERT(
		d > 0,
		throw ::std::invalid_argument("[dcs::sysid::rls_ff_arx_miso] The input delay cannot be less than 1.")
	);
	// pre: size(theta_hat) == n_a+n_b*n_u
	DCS_ASSERT(
		n == (n_a+n_b*n_u),
		throw ::std::invalid_argument("[dcs::sysid::rls_ff_arx_miso] The parameter vector has an invalid size.")
	);
	// pre: P is a square matrix of order n
	DCS_ASSERT(
		ublasx::num_rows(P) == n && ublasx::num_columns(P) == n,
		throw ::std::invalid_argument("[dcs::sysid::rls_ff_arx_miso] The covariance matrix has an invalid size.")
	);
	// pre: size(phi) == n_a+(n_b+d)*n_u
	DCS_ASSERT(
		//n_phi == (n_a+(n_b+d)*n_u),
		n_phi == (n_a+(n_b+d-1)*n_u),
		throw ::std::invalid_argument("[dcs::sysid::rls_ff_arx_miso] The regression vector has an invalid size.")
	);

	--d;

	// Create an auxiliary regression vector which takes into consideration the
	// actual input delay d.
	work_vector_type aux_phi;
	if (d > 0)
	{
		aux_phi.resize(n, false);
		ublas::subrange(aux_phi, 0, n_a) = ublas::subrange(phi(), 0, n_a);
		//ublas::subrange(aux_phi, n_a, n) = ublas::subrange(phi(), n_a+d*n_u, n_phi);
		for (size_type i = n_a; i < n; i += n_b+d)
		{
			size_type k(i+n_b);
			ublas::subrange(aux_phi, i, k) = ublas::subrange(phi(), i+d, k+d);
		}
	}
	else
	{
		aux_phi = phi;
	}

	// Compute output estimate
	value_type y_hat = ublas::inner_prod(aux_phi, theta_hat);

	// Compute the prediction error
	value_type epsi = y-y_hat;

	// Compute the time-varying forgetting factor
	lambda = lambda+(1-lambda)*::std::pow(2, -::round(rho*epsi*epsi));

	// Compute the Gain
	// l(k+1) = \frac{P(k)\phi(k+1)}{\lambda(k)+\phi^T(k+1)P(k)\phi(k+1)}
	work_vector_type l(n);
	l = ublas::prod(P, aux_phi)
		/ (
			lambda
			+ ublas::inner_prod(
				ublas::prod(aux_phi, P),
				aux_phi
			)
	);

	// Update the covariance matrix
	P() = (P - ublas::prod(ublas::outer_prod(l, aux_phi), P)) / lambda;

	// Update parameters estimate
	// \hat{\theta}(k+1) = \hat{\theta}(k)+(y(k+1)-\Phi^T(k+1)\hat{\theta}(k))l^T(k+1)
	theta_hat() = theta_hat + l*epsi;

	// Clean-up unused memory
	aux_phi.resize(0, false);
	l.resize(0, false);

	// Update the Regression vector
	work_vector_type phi_new(n_phi, 0);
	phi_new(0) = -y;
	// phi = [y(k-1) ... y(k-n_a) u_1(k-1) ... u_{n_u}(k-1) ... u_1(k-1-d) ... u_{n_u}(k-1-d) ... u_1(k-n_b-d) ... u_{n_u}(k-n_b-d)]^T
//	ublas::subrange(phi_new, 1, n_a) = ublas::subrange(phi(), 0, n_a-1);
//	ublas::subrange(phi_new, n_a, n_a+n_u) = u;
//	ublas::subrange(phi_new, n_a+n_u, n_phi) = ublas::subrange(phi(), n_a, n_phi-n_u);

	// MATLAB uses this convention for the regression vector:
	// phi = [y(k-1) ... y(k-n_a) u_1(k-1) ... u_1(k-1-d) ... u_1(k-n_b-d) ... u_{n_u}(k-1) ... u_{n_u}(k-1-d) ... u_{n_u}(k-n_b-d)]^T
	ublas::subrange(phi_new, 1, n_a) = ublas::subrange(phi(), 0, n_a-1);
//	ublas::subslice(phi_new, n_a, n_b, n_u) = u;
	ublas::subslice(phi_new, n_a, n_b+d, n_u) = u;
//	ublas::subslice(phi_new, n_a+1, n_b, (n_b+d-1)*n_u) = ublas::subslice(phi(), n_a, n_b, (n_b+d-1)*n_u);
	//for (size_type i = n_a; i < (n_b+d-1)*n_u; i += n_b+d)
	for (size_type i = n_a; i < n_phi; i += n_b+d)
	{
		size_type k(i+n_b+d-1);
		ublas::subrange(phi_new, i+1, k+1) = ublas::subrange(phi(), i, k);
	}
	phi() = phi_new;
//DCS_DEBUG_TRACE("[rarx_miso] phi(k) = " << phi);//XXX

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
	typename ThetaMatrixExprT,
	typename PMatrixExprT,
	typename PhiVectorExprT
>
void rls_arx_mimo_init(UIntT n_a,
					   UIntT n_b,
					   UIntT d,
					   UIntT n_y,
					   UIntT n_u,
					   ::boost::numeric::ublas::matrix_container<ThetaMatrixExprT>& Theta0_hat,
					   ::boost::numeric::ublas::matrix_container<PMatrixExprT>& P0,
					   ::boost::numeric::ublas::vector_container<PhiVectorExprT>& phi0)
{
	namespace ublas = ::boost::numeric::ublas;

	typedef typename ublas::promote_traits<
					typename ublas::promote_traits<
							typename ublas::matrix_traits<ThetaMatrixExprT>::value_type,
							typename ublas::matrix_traits<PMatrixExprT>::value_type
						>::promote_type,
					typename ublas::vector_traits<PhiVectorExprT>::value_type
				>::promote_type value_type;
	typedef typename ublas::promote_traits<
					typename ublas::promote_traits<
							typename ublas::matrix_traits<ThetaMatrixExprT>::size_type,
							typename ublas::matrix_traits<PMatrixExprT>::size_type
						>::promote_type,
					typename ublas::vector_traits<PhiVectorExprT>::size_type
				>::promote_type size_type;

	const size_type n(n_a*n_y+n_b*n_u);
	const size_type n_phi(n_a*n_y+(n_b+d)*n_u);

	Theta0_hat() = ublas::scalar_matrix<value_type>(n, n_y, ::std::numeric_limits<value_type>::epsilon());
	P0() = 1.0e+4*ublas::identity_matrix<value_type>(n,n);
	phi0() = ublas::zero_vector<value_type>(n_phi);
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
	typename YVectorExprT,
	typename UVectorExprT,
	typename RealT,
	typename UIntT,
	typename ThetaMatrixExprT,
	typename PMatrixExprT,
	typename PhiVectorExprT
>
::boost::numeric::ublas::vector<RealT> rls_ff_arx_mimo(::boost::numeric::ublas::vector_expression<YVectorExprT> const& y,
													   ::boost::numeric::ublas::vector_expression<UVectorExprT> const& u,
													   RealT lambda,
													   UIntT n_a,
													   UIntT n_b,
													   UIntT d,
													   ::boost::numeric::ublas::matrix_container<ThetaMatrixExprT>& Theta_hat,
													   ::boost::numeric::ublas::matrix_container<PMatrixExprT>& P,
													   ::boost::numeric::ublas::vector_container<PhiVectorExprT>& phi)
{
	namespace ublas = ::boost::numeric::ublas;
	namespace ublasx = ::boost::numeric::ublasx;

	typedef typename ublas::promote_traits<
					typename ublas::promote_traits<
							typename ublas::vector_traits<YVectorExprT>::value_type,
							typename ublas::promote_traits<
								typename ublas::vector_traits<UVectorExprT>::value_type,
								typename ublas::promote_traits<
									typename ublas::matrix_traits<ThetaMatrixExprT>::value_type,
									typename ublas::matrix_traits<PMatrixExprT>::value_type
								>::promote_type
							>::promote_type
						>::promote_type,
					typename ublas::vector_traits<PhiVectorExprT>::value_type
				>::promote_type value_type;
	typedef typename ublas::promote_traits<
					typename ublas::promote_traits<
							typename ublas::vector_traits<YVectorExprT>::size_type,
							typename ublas::promote_traits<
								typename ublas::vector_traits<UVectorExprT>::size_type,
								typename ublas::promote_traits<
									typename ublas::matrix_traits<ThetaMatrixExprT>::size_type,
									typename ublas::matrix_traits<PMatrixExprT>::size_type
								>::promote_type
							>::promote_type
						>::promote_type,
					typename ublas::vector_traits<PhiVectorExprT>::size_type
				>::promote_type size_type;
	typedef ublas::vector<value_type> work_vector_type;

	const size_type n = ublasx::num_rows(Theta_hat);
	const size_type n_phi = ublasx::size(phi);
	const size_type n_y = ublasx::size(y);
	const size_type n_u = ublasx::size(u);

	// pre: size(theta_hat) == n_a+n_b*n_u
	DCS_ASSERT(
		n == (n_a*n_y+n_b*n_u) && ublasx::num_columns(Theta_hat) == n_y,
		throw ::std::invalid_argument("[dcs::sysid::rls_ff_arx_mimo] The parameter vector has an invalid size.")
	);
	// pre: P is a square matrix of order n
	DCS_ASSERT(
		ublasx::num_rows(P) == n && ublasx::num_columns(P) == n,
		throw ::std::invalid_argument("[dcs::sysid::rls_ff_arx_mimo] The covariance matrix has an invalid size.")
	);
	// pre: size(phi) == n_a+(n_b+d)*n_u
	DCS_ASSERT(
		n_phi == (n_a*n_y+(n_b+d)*n_u),
		throw ::std::invalid_argument("[dcs::sysid::rls_ff_arx_mimo] The regression vector has an invalid size.")
	);

	// Create an auxiliary regression vector which takes into consideration the
	// actual input delay d.
	work_vector_type aux_phi(n);
	if (d > 0)
	{
		ublas::subrange(aux_phi, 0, n_a*n_y) = ublas::subrange(phi(), 0, n_a*n_y);
		ublas::subrange(aux_phi, n_a*n_y, n) = ublas::subrange(phi(), n_a*n_y+d*n_u, n_phi);
	}
	else
	{
		aux_phi = phi;
	}
//DCS_DEBUG_TRACE("[rarx_mimo] aux_phi(k) = " << aux_phi);//XXX


	// Compute the Gain
	//  l(k+1) = \frac{P(k)\phi(k+1)}{\lambda(k)+\phi^T(k+1)P(k)\phi(k+1)}
	work_vector_type l(n);
	l = ublas::prod(P, aux_phi)
		/ (
			lambda
			+ ublas::inner_prod(
				ublas::prod(aux_phi, P),
				aux_phi
			)
	);
//DCS_DEBUG_TRACE("[rarx_mimo] l(k) = " << l);//XXX

	// Update the covariance matrix
	//  P(k+1) = \frac{1}{\lambda(k)}\left[I-l(k+1)\Phi^T(k+1)\right]P(k)
	// value_type check_cov((value_type(1)/(lambda+ublas::inner_prod(ublas::prod(aux_phi, P), aux_phi)))*ublas::inner_prod(ublas::prod(ublas::trans(Theta_hat), aux_phi),ublas::prod(ublas::trans(Theta_hat), aux_phi)));
	// bool safe_cov(check_cov >= 0.001);
	P() = ublas::prod(
			ublas::identity_matrix<value_type>(n)
			-
			ublas::outer_prod(l, aux_phi),
			P
		)
		/ lambda;
	// TODO: Should we use this trick
	// Apply the Bittanti's correction.
	//   S. Bittanti, P. Bolzern, and M. Campi.
	//   Exponential convergence of a modified directional forgetting identification algorithm.
	//   Systems & Control Letters, 14:131–137, 1990.
	//P() = P()+0.01*ublas::identity_matrix<value_type>(n);
//DCS_DEBUG_TRACE("[rarx_mimo] P(k) = " << P);//XXX

	// Compute the output estimate
	//  \hat{y}(k) = (\phi^T(k)\Theta(k))^T = \Theta^T(k)\phi
	//vector_type y_hat_t = ublas::prod(ublas::trans(aux_phi), theta_hat);
	work_vector_type y_hat = ublas::prod(ublas::trans(Theta_hat), aux_phi);

	// Update parameters estimate
	//  \hat{\Theta}(k+1) = \hat{\Theta}(k)+l^T(k+1)(y^T(k+1)-\phi^T(k+1)\hat{\Theta}(k))
	//Theta_hat() = Theta_hat + ublas::outer_prod(l, ublas::trans(y) - y_hat_t);
	Theta_hat() = Theta_hat + ublas::outer_prod(l, ublas::trans(y - y_hat));
//DCS_DEBUG_TRACE("[rarx_mimo] Theta_hat(k) = " << Theta_hat);//XXX

	// Clean-up unused memory
	aux_phi.resize(0, false);
	l.resize(0, false);
 
	// Update the Regression vector
	work_vector_type phi_new(n_phi, 0);
	// MATLAB uses the following form for the regression vector:
	//  phi = [y_1(k-1) ... y_1(k-n_a) ... y_{n_y}(k-1) ... y_{n_y}(k-n_a) u_1(k-1) ... u_1(k-1-d) ... u_1(k-n_b-d) ... u_{n_u}(k-1) ... u_{n_u}(k-1-d) ... u_{n_u}(k-n_b-d)]^T
	// According to this convention, \hat{\Theta} has the following meaning:
	//   \hat{\Theta} = [a_{11}^{1},     a_{21}^{1},     ...,  a_{n_y1}^{1};
	//                   ...,            ...,            ...,  ...;
	//                   a_{11}^{n_a},   a_{21}^{n_a},   ...,  a_{n_y1}^{n_a};
	//                   ...,            ...,            ...,  ...;
	//                   a_{1n_y}^{1},   a_{2n_y}^{1},   ...,  a_{n_yn_y}^{1};
	//                   ...,            ...,            ...,  ...;
	//                   a_{1n_y}^{n_a}, a_{2n_y}^{n_a}, ...,  a_{n_yn_y}^{n_a};
	//                   b_{11}^{1},     b_{21}^{1},     ...,  b_{n_y1}^{1};
	//                   ...,            ...,            ...,  ...;
	//                   b_{11}^{n_b},   b_{21}^{n_b},   ...,  b_{n_y1}^{n_b};
	//                   ...,            ...,            ...,  ...;
	//                   b_{1n_u}^{1},   b_{2n_u}^{1},   ...,  b_{n_yn_u}^{1};
	//                   ...,            ...,            ...,  ...;
	//                   b_{1n_u}^{n_b}, b_{2n_u}^{n_b}, ...,  b_{n_yn_u}^{n_b}]
	ublas::subslice(phi_new, 0, n_a, n_y) = -y;
    //ublas::subrange(phi_new, n_y, n_y*n_a) = ublas::subrange(phi(), 0, n_y*(n_a-1));
	for (size_type i = 1; i < n_a; ++i)
	{
    	ublas::subslice(phi_new, i, n_a, n_y) = ublas::subslice(phi(), i-1, n_a, n_y);
	}
	const size_type nytna(n_y*n_a);
	const size_type nbpdm1(n_b+d-1);
	ublas::subslice(phi_new, nytna, n_b, n_u) = u;
//	ublas::subslice(phi_new, n_y*n_a+1, n_b, (n_b+d-1)*n_u) = ublas::subslice(phi(), n_y*n_a, n_b, (n_b+d-1)*n_u);
	for (size_type i = 1; i < nbpdm1; ++i)
	{
		size_type start(nytna+i);
    	ublas::subslice(phi_new, start, n_b, n_u) = ublas::subslice(phi(), start-1, n_b, n_u);
	}
	// Alternative form
	// phi = [y_1(k-1) ... y_{n_y}(k-1) ... y_1(k-n_a) ... y_{n_y}(k-n_a) u_1(k-1) ... u_{n_u}(k-1) ... u_1(k-1-d) ... u_{n_u}(k-1-d) ... u_1(k-n_b-d) ... u_{n_u}(k-n_b-d)]^T
//	ublas::subrange(phi_new, 0, n_y) = -y;
//	ublas::subrange(phi_new, n_y, n_y*n_a) = ublas::subrange(phi(), 0, n_y*(n_a-1));
//	ublas::subrange(phi_new, n_y*n_a, n_y*n_a+n_u) = u;
//	ublas::subrange(phi_new, n_y*n_a+n_u, n_phi) = ublas::subrange(phi(), n_y*n_a, n_phi-n_u);

	phi() = phi_new;
//DCS_DEBUG_TRACE("[rarx_mimo] phi(k) = " << phi);//XXX

	return y_hat;
}


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
	typename UIntT,
	typename ThetaVectorExprT,
	typename PMatrixExprT,
	typename PhiVectorExprT
>
void rls_arx_siso_init(UIntT n_a,
					   UIntT n_b,
					   UIntT d,
					   ::boost::numeric::ublas::vector_container<ThetaVectorExprT>& theta0_hat,
					   ::boost::numeric::ublas::matrix_container<PMatrixExprT>& P0,
					   ::boost::numeric::ublas::vector_container<PhiVectorExprT>& phi0)
{
	rls_arx_miso_init(n_a,
					  n_b,
					  d,
					  UIntT(1),
					  theta0_hat,
					  P0,
					  phi0);
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
 * \param delta A multiplication factor used for applying the
 *  Bittanti's correction [2]. The correction is applied only for values
 *  of \a delta greater than zero.
 *
 * This variant of the RLS algorithm uses a time-varying forgetting factor and
 * it is based on [1,2].
 *
 * References:
 * -# R. Kulhavy, M. Karny.
 *    "Tracking of Slowly Varying Parameters by Directional Forgetting",
 *    IFAC Proc. Ser. 1985, 687-692.
 * -# S. Bittanti, P. Bolzern, M. Campi.
 *    "Exponential Convergence of a Modified Directional Forgetting
 *     Identification Algorithm",
 *    Syst. Control Lett. 1990, 14, 131-137.
 * .
 */
template <
	typename RealT,
	typename UIntT,
	typename ThetaVectorExprT,
	typename PMatrixExprT,
	typename PhiVectorExprT
>
RealT rls_bittanti1990_arx_siso(RealT y,
								RealT u,
								RealT lambda,
								UIntT n_a,
								UIntT n_b,
								UIntT d,
								::boost::numeric::ublas::vector_container<ThetaVectorExprT>& theta_hat,
								::boost::numeric::ublas::matrix_container<PMatrixExprT>& P,
								::boost::numeric::ublas::vector_container<PhiVectorExprT>& phi,
								RealT delta = RealT/*zero*/())
{
	return rls_bittanti1990_arx_miso(y,
									 ::boost::numeric::ublas::scalar_vector<RealT>(1, u),
									 lambda,
									 n_a,
									 n_b,
									 d,
									 theta_hat,
									 P,
									 phi,
									 delta);
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
	typename ThetaVectorExprT,
	typename PMatrixExprT,
	typename PhiVectorExprT
>
RealT rls_ff_arx_siso(RealT y,
					  RealT u,
					  RealT lambda,
					  UIntT n_a,
					  UIntT n_b,
					  UIntT d,
					  ::boost::numeric::ublas::vector_container<ThetaVectorExprT>& theta_hat,
					  ::boost::numeric::ublas::matrix_container<PMatrixExprT>& P,
					  ::boost::numeric::ublas::vector_container<PhiVectorExprT>& phi)
{
	return rls_ff_arx_miso(y,
						   ::boost::numeric::ublas::scalar_vector<RealT>(1, u),
						   lambda,
						   n_a,
						   n_b,
						   d,
						   theta_hat,
						   P,
						   phi);
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
 * \param delta A multiplication factor used for applying the
 *  Bittanti's correction [2]. The correction is applied only for values
 *  of \a delta greater than zero.
 *
 * This variant of the RLS algorithm uses a time-varying forgetting factor and
 * it is based on [1].
 *
 * References:
 * -# R. Kulhavy, M. Karny.
 *    "Tracking of Slowly Varying Parameters by Directional Forgetting",
 *    IFAC Proc. Ser. 1985, 687-692.
 * .
 */
template <
	typename RealT,
	typename UIntT,
	typename ThetaVectorExprT,
	typename PMatrixExprT,
	typename PhiVectorExprT
>
RealT rls_kulhavy1984_arx_siso(RealT y,
								RealT u,
								RealT lambda,
								UIntT n_a,
								UIntT n_b,
								UIntT d,
								::boost::numeric::ublas::vector_container<ThetaVectorExprT>& theta_hat,
								::boost::numeric::ublas::matrix_container<PMatrixExprT>& P,
								::boost::numeric::ublas::vector_container<PhiVectorExprT>& phi)
{
	return rls_kulhavy1984_arx_miso(y,
									::boost::numeric::ublas::scalar_vector<RealT>(1, u),
									lambda,
									n_a,
									n_b,
									d,
									theta_hat,
									P,
									phi);
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
 * \param lambda The minimum value for the forgetting factor.
 * \param rho A design parameter (see [1]).
 * \param n_a The memory of the ARX model with respect to the output variables.
 * \param n_b The memory of the ARX model with respect to the input variables.
 * \param d The delay of the ARX model.
 * \param theta_hat The current parameter estimate vector.
 * \param P Covariance matrix.
 * \param theta_hat The current regressor vector.
 * \return The predicted value of the outputs.
 *  Furthermore vector \a \theta_hat, matrix \a P and vector \a phi
 *  are changed in order to reflect the current RLS update step.
 *
 * This variant of the RLS algorithm uses a time-varying forgetting factor and
 * it is based on [1,2].
 *
 * References:
 * -# D.J. Park, B.E. Jun, J.H. Kim,
 *    "Fast Tracking RLS Algorithm using Novel Variable Forgetting Factor with Unity Zone",
 *    Elettron. Lett. 1991, 27, 2150-2151.
 * .
 */
template <
	typename RealT,
	typename UIntT,
	typename ThetaVectorExprT,
	typename PMatrixExprT,
	typename PhiVectorExprT
>
RealT rls_park1991_arx_siso(RealT y,
							RealT u,
							RealT lambda,
							RealT rho,
							UIntT n_a,
							UIntT n_b,
							UIntT d,
							::boost::numeric::ublas::vector_container<ThetaVectorExprT>& theta_hat,
							::boost::numeric::ublas::matrix_container<PMatrixExprT>& P,
							::boost::numeric::ublas::vector_container<PhiVectorExprT>& phi)
{
	return rls_park1991_arx_miso(y,
								 ::boost::numeric::ublas::scalar_vector<RealT>(1, u),
								 lambda,
								 rho,
								 n_a,
								 n_b,
								 d,
								 theta_hat,
								 P,
								 phi);
}

}} // Namespace dcs::sysid


#endif // DCS_SYSID_ALGORITHM_RLS_HPP
