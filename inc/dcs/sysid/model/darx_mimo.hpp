/**
 * \file dcs/sysid/model/darx_mimo.hpp
 *
 * \brief A multi-input multi-output (MIMO) Autoregressive with exogenous
 * inputs (ARX) discrete model.
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

#ifndef DCS_SYSID_MODEL_DARX_MIMO_HPP
#define DCS_SYSID_MODEL_DARX_MIMO_HPP


#include <algorithm>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <boost/numeric/ublasx/operation/size.hpp>
#include <cstddef>
#include <dcs/assert.hpp>
//#include <dcs/math/la/container/identity_matrix.hpp>
//#include <dcs/math/la/container/zero_vector.hpp>
//#include <dcs/math/la/operation/num_columns.hpp>
//#include <dcs/math/la/operation/num_columns.hpp>
//#include <dcs/math/la/operation/row.hpp>
//#include <dcs/math/la/operation/size.hpp>
//#include <dcs/math/la/operation/subslice.hpp>
//#include <dcs/math/la/operation/matrix_basic_operations.hpp>
#include <iostream>
#include <iterator>
#include <utility>


namespace dcs { namespace sysid {

/**
 * \brief A multi-input multi-output (MIMO) Autoregressive with exogenous
 * inputs (ARX) discrete model.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
template <
	typename MatrixT,
	typename RealT = double,
	typename UIntT = unsigned int
>
class darx_mimo_model
{
	public: typedef MatrixT matrix_type;
	public: typedef RealT real_type;
	public: typedef UIntT uint_type;
	private: typedef ::std::size_t size_type;


	public: template <
				typename ForwardIteratorAT,
				typename ForwardIteratorBT
			>
		darx_mimo_model(ForwardIteratorAT A_begin, ForwardIteratorAT A_end, ForwardIteratorBT B_begin, ForwardIteratorBT B_end, uint_type delay = 0, uint_type sampling_time = 1)
		: As_(A_begin, A_end),
		  Bs_(B_begin, B_end),
		  E_covar_(::boost::numeric::ublas::identity_matrix<real_type>(::boost::numeric::ublasx::num_rows(As_[0]))),
		  d_(delay),
		  ts_(sampling_time)
	{
		// Empty
	}


	public: template <
				typename ForwardIteratorAT,
				typename ForwardIteratorBT
			>
		darx_mimo_model(ForwardIteratorAT A_begin, ForwardIteratorAT A_end, ForwardIteratorBT B_begin, ForwardIteratorBT B_end, matrix_type const& noise_covar, uint_type delay = 0, uint_type sampling_time = 1)
		: As_(A_begin, A_end),
		  Bs_(B_begin, B_end),
		  E_covar_(noise_covar),
		  d_(delay),
		  ts_(sampling_time)
	{
		// Empty
	}


	public: ::std::pair<uint_type,uint_type> order() const
	{
		return ::std::make_pair(
					As_.size(),
					(Bs_.size() > 0)
						? (Bs_.size()-1)
						: 0
			);
	}



	public: uint_type delay() const
	{
		return d_;
	}


	/**
	 * \brief Simulate the MIMO model with the given input data.
	 *
	 * \param U The input data matrix. For $\fn_b\f$ input channels and
	 *  \f$n_{obs}\f$ samples, \a U is an \f$n_{obs} \times n_b\f$ matrix.
	 * \return An output data matrix of size \f$n_{obs} \times n_a\f$, where
	 *  \f$n_a\f$ is the number of output channels.
	 */
	public: matrix_type simulate(matrix_type const& U, real_type na_value = real_type(0)) const
	{
		namespace ublas = ::boost::numeric::ublas;
		namespace ublasx = ::boost::numeric::ublasx;

		size_type n_obs = ublasx::num_rows(U); // # of samples
		size_type n_a = As_.size(); // # of output channels
		size_type n_b = Bs_.size(); // # of input channels
		size_type n_y = ublasx::num_rows(As_[0]); // # of outputs
		//size_type n_u = ::dcs::math::la::num_columns(Bs_[0]); // # of inputs
		//size_type n_min = ::std::max(n_a*ts_, (n_b+d_)*ts_)-1;
		//size_type k_min = (n_b > 0) ? ((n_b+d_)*ts_-1) : 0;
		size_type k_min = 0;
		size_type k_max = n_obs - ::std::min(static_cast<size_type>(d_*ts_), n_obs);

		matrix_type Y(n_obs, n_y, na_value);

		// For n_y output channels and N samples, this is an N-by-n_y matrix.
		ublas::zero_vector<real_type> zero_vec(n_y);
		for (size_type k = k_min; k < k_max; ++k)
		{
			// Initialize current output to zero
			ublas::row(Y, k) = zero_vec;

			// Add the outputs contribution (if any)
			if (n_a > 0 && k > 0)
			{
				size_type j_max = ::std::min(n_a, k);
				for (size_type j = 0; j < j_max; ++j)
				{
					ublas::row(Y, k) -= ublas::prod(
							As_[j], //n_yxn_y
							ublas::row(Y, k-1-j) //n_ux1
						);
				}
			}

			// Add the inputs contribution (if any)
			if (n_b > 0)
			{
				size_type j_max = ::std::min(n_b, k+1);
				for (size_type j = 0; j < j_max; ++j)
				{
					ublas::row(Y, k) += ublas::prod(
							Bs_[j], //n_yxn_u
							ublas::row(U, k-j) //n_ux1
						);
				}
			}
		}

		return Y;
	}


	public: matrix_type simulate(matrix_type const& U, matrix_type const& E, real_type na_value = real_type(0)) const
	{
		namespace ublas = ::boost::numeric::ublas;
		namespace ublasx = ::boost::numeric::ublasx;

		// preconditions
		DCS_ASSERT(
			ublasx::num_rows(U) == ublasx::num_rows(E),
			throw ::std::logic_error("The number of input and noise samples does not match.")
		);
		DCS_ASSERT(
			ublasx::num_columns(E) == ublasx::num_rows(As_[0]),
			throw ::std::logic_error("The number of noise and output channels does not match.")
		);
		DCS_ASSERT(
			ublasx::num_columns(U) == ublasx::num_rows(Bs_[0]),
			throw ::std::logic_error("The number of sample input channels and input channels does not match.")
		);

		size_type n_obs = ublasx::num_rows(U); // # of samples
		size_type n_a = As_.size(); // # of output channels
		size_type n_b = Bs_.size(); // # of input channels
		size_type n_y = ublasx::num_rows(As_[0]); // # of outputs
		//size_type n_u = ublas::num_columns(Bs_[0]); // # of inputs
		//size_type n_min = ::std::max(n_a*ts_, (n_b+d_)*ts_)-1;
		//size_type k_min = (n_b > 0) ? ((n_b+d_)*ts_-1) : 0;
		size_type k_min = 0;
		size_type k_max = n_obs - ::std::min(static_cast<size_type>(d_*ts_), n_obs);

		matrix_type Y(n_obs, n_y, na_value);

		// For n_y output channels and N samples, this is an N-by-n_y matrix.
		ublas::zero_vector<real_type> zero_vec(n_y);
		for (size_type k = k_min; k < k_max; ++k)
		{
			// Initialize current output to zero
			ublas::row(Y, k) = zero_vec;

			// Add the outputs contribution (if any)
			if (n_a > 0 && k > 0)
			{
				size_type j_max = ::std::min(n_a, k);
				for (size_type j = 0; j < j_max; ++j)
				{
					ublas::row(Y, k) -= ublas::prod(
							As_[j], //n_yxn_y
							ublas::row(Y, k-1-j) //n_ux1
						);
				}
			}

			// Add the inputs contribution (if any)
			if (n_b > 0)
			{
				size_type j_max = ::std::min(n_b, k+1);
				for (size_type j = 0; j < j_max; ++j)
				{
					ublas::row(Y, k) += ublas::prod(
							Bs_[j], //n_yxn_u
							ublas::row(U, k-j) //n_ux1
						);
				}
			}

			// Add the noise contribution
			ublas::row(Y, k) += ublas::prod(
						E_covar_,
						ublas::row(E, k)
				);
		}

		return Y;
	}


	public: uint_type sampling_time() const
	{
		return ts_;
	}


	public: ::std::vector<matrix_type> input_parameters() const
	{
		return Bs_;
	}


	public: ::std::vector<matrix_type> output_parameters() const
	{
		return As_;
	}


	public: matrix_type noise_covariance() const
	{
		return E_covar_;
	}


	private: ::std::vector<matrix_type> As_;
	private: ::std::vector<matrix_type> Bs_;
	private: matrix_type E_covar_;
	private: uint_type d_;
	private: uint_type ts_;
};


template <
	typename MatrixT,
	typename RealT,
	typename UIntT
>
::std::pair<UIntT,UIntT> order(darx_mimo_model<MatrixT,RealT,UIntT> const& model)
{
	return model.order();
}


template <
	typename MatrixT,
	typename RealT,
	typename UIntT
>
UIntT delay(darx_mimo_model<MatrixT,RealT,UIntT> const& model)
{
	return model.delay();
}


template <
	typename MatrixT,
	typename RealT,
	typename UIntT
>
UIntT sampling_time(darx_mimo_model<MatrixT,RealT,UIntT> const& model)
{
	return model.sampling_time();
}


template <
	typename MatrixT,
	typename RealT,
	typename UIntT
>
::std::vector<MatrixT> output_parameters(darx_mimo_model<MatrixT,RealT,UIntT> const& model)
{
	return model.output_parameters();
}


template <
	typename MatrixT,
	typename RealT,
	typename UIntT
>
::std::vector<MatrixT> input_parameters(darx_mimo_model<MatrixT,RealT,UIntT> const& model)
{
	return model.input_parameters();
}


template <
	typename MatrixT,
	typename RealT,
	typename UIntT
>
MatrixT noise_covariance(darx_mimo_model<MatrixT,RealT,UIntT> const& model)
{
	return model.noise_covariance();
}


template <
	typename MatrixT,
	typename RealT,
	typename UIntT
>
MatrixT simulate(darx_mimo_model<MatrixT,RealT,UIntT> const& model, MatrixT const& u, RealT na_value = RealT(0))
{
	return model.simulate(u, na_value);
}


template <
	typename MatrixT,
	typename RealT,
	typename UIntT
>
MatrixT simulate(darx_mimo_model<MatrixT,RealT,UIntT> const& model, MatrixT const& u, MatrixT const& e, RealT na_value = RealT(0))
{
	return model.simulate(u, e, na_value);
}


template <
	typename CharT,
	typename CharTraitsT,
	typename MatrixT,
	typename RealT,
	typename UIntT
>
::std::basic_ostream<CharT,CharTraitsT>& operator<<(::std::basic_ostream<CharT,CharTraitsT>& os, darx_mimo_model<MatrixT,RealT,UIntT> const& model)
{
	::std::pair<UIntT,UIntT> ord = order(model);

	os << "<MIMO ARX model y(k)+A_1*y(k-t_s)+...+A_{n_a}*y(k-n_a*t_s)=B_0*u(k-d)+...+B_{n_b}*u(k-d-n_b*t_s):"
		<<  " t_s=" << sampling_time(model)
		<< ", n_a=" << ord.first
		<< ", n_b=" << ord.second
		<< ", d=" << delay(model);

	::std::vector<MatrixT> output_parms = output_parameters(model);
	for (
		::std::size_t i = 0;
		i < output_parms.size();
		++i
	) {
		os << ", A_" << i << "=" << output_parms[i];
	}
	::std::vector<MatrixT> input_parms = input_parameters(model);
	for (
		::std::size_t i = 0;
		i < input_parms.size();
		++i
	) {
		os << ", B_" << i << "=" << input_parms[i];
	}

	os << ", Noise-Covariance: " << noise_covariance(model);

	os << ">";

	return os;
}


}} // Namespace dcs::sysid


#endif // DCS_SYSID_MODEL_DARX_MIMO_HPP
