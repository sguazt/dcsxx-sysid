/**
 * \file dcs/sysid/model/darx_siso.hpp
 *
 * \brief A single-input single-output (SISO) Autoregressive with exogeneous
 * inputs (ARX) discrete model.
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

#ifndef DCS_SYSID_MODEL_DARX_SISO_HPP
#define DCS_SYSID_MODEL_DARX_SISO_HPP


#include <algorithm>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublasx/operation/size.hpp>
#include <cstddef>
//#include <dcs/math/la/operation/size.hpp>
//#include <dcs/math/la/operation/subrange.hpp>
//#include <dcs/math/la/operation/subslice.hpp>
//#include <dcs/math/la/operation/vector_basic_operations.hpp>
#include <iostream>
#include <iterator>
#include <utility>


namespace dcs { namespace sysid {

/**
 * \brief A single-input single-output (SISO) Autoregressive with exogeneous
 * inputs (ARX) discrete model.
 *
 * Creates an object containing parameters that describe the general
 * single-input, single-output model structure of ARX type:
 * \f[
 *  y(k) + a_1 y(k-t_s) + a_2 y(k - 2 t_s) + \cdots + a_{n_a} y(k - n_a t_s) =
 *         b_0 u(k - d t_s) + b_1 u(k-(d+1)t_s) + \cdots + b_{n_b} u(k-(d+n_b) t_s) + e(k)
 * \f]
 * where:
 * - \f$y(k) \in \mathbb{R}\f$ is the system \e output at time \f$k\f$;
 * - \f$\mathbf{a}=\left(a_1\, \ldots\, a_{n_a}\right)^T\f$ is the vector of
 *   <em>output parameters</em>;
 * - \f$u(k) \in \mathbb{R}\f$ is the system \e input at time \f$k\f$;
 * - \f$\mathbf{b}=\left(b_0\, b_1\, \ldots\, b_{n_b}\right)^T\f$ is the vector
 *   of <em>input parameters</em>;
 * - \f$e(k)\f$ is the <em>zero-mean white noise</em> (or <em>system
 *   disturbance</em>) at time \f$k\f$;
 * - \f$n_a, n_b \in \mathbb{N}\f$ are the model \e orders, with \f$n_a\f$ being
 *   the number of poles and \f$n_b\f$ the number of zeros;
 * - \f$t_s \in \mathbb{N}\f$ is the <em>sampling time</em>;
 * - \f$d \in \mathbb{N}\f$ is the input \e delay.
 * .
 * The model can be expressed in a more compact way through the <em>backward
 * shift operator</em> \f$q^{-1}\f$, for which \f$q^{-1}x(k)=x(k-1)\f$:
 * \f[
 *  A(q^{-1},t_s)y(k) = q^{-d}B(q,t_s)u(k)
 * \f]
 * where:
 * \f{align*}{
 *  A(q^{-1},t_s) &= 1 + a_1 q^{-t_s} + \cdots a_{n_a} q^{-n_a t_s} \\
 *  B(q^{-1},t_s) &= b_0 + b_1 q^{-t_s} + \cdots b_{n_b} q^{-n_b t_s}
 * \f}
 *
 * \author Marco Guazzone, &lt;marco.guazzone@mfn.unipmn.it&gt;
 */
template <
	typename VectorT,
	typename RealT = double,
	typename UIntT = unsigned int
>
class darx_siso_model
{
	public: typedef VectorT vector_type;
	public: typedef RealT real_type;
	public: typedef UIntT uint_type;
	private: typedef ::std::size_t size_type;


	public: darx_siso_model(VectorT a, VectorT b, uint_type delay = 0, uint_type sampling_time = 1)
		: a_(a),
		  b_(b),
		  e_var_(1),
		  d_(delay),
		  ts_(sampling_time)
	{
	}


//	public: template <
//				typename ForwardIteratorAT,
//				typename ForwardIteratorBT
//			>
//		darx_siso_model(ForwardIteratorAT a_begin, ForwardIteratorAT a_end, ForwardIteratorBT b_begin, ForwardIteratorBT b_end, uint_type delay = 0, uint_type sampling_time = 1)
//		: a_(a_begin, a_end),
//		  b_(b_begin, b_end),
//		  e_var_(1),
//		  d_(delay),
//		  ts_(sampling_time)
//	{
//	}


	public: darx_siso_model(VectorT a, VectorT b, real_type const& noise_var, uint_type delay = 0, uint_type sampling_time = 1)
		: a_(a),
		  b_(b),
		  e_var_(noise_var),
		  d_(delay),
		  ts_(sampling_time)
	{
	}


//	public: template <
//				typename ForwardIteratorAT,
//				typename ForwardIteratorBT
//			>
//		darx_siso_model(ForwardIteratorAT a_begin, ForwardIteratorAT a_end, ForwardIteratorBT b_begin, ForwardIteratorBT b_end, real_type const& noise_var, uint_type delay = 0, uint_type sampling_time = 1)
//		: a_(a_begin, a_end),
//		  b_(b_begin, b_end),
//		  e_var_(noise_var),
//		  d_(delay),
//		  ts_(sampling_time)
//	{
//	}


	public: ::std::pair<uint_type,uint_type> order() const
	{
		return ::std::make_pair(
					::boost::numeric::ublasx::size(a_),
					::boost::numeric::ublasx::size(b_)-1
			);
	}



	public: uint_type delay() const
	{
		return d_;
	}


	public: vector_type simulate(vector_type const& u, real_type na_value = real_type(0)) const
	{
		namespace ublas = ::boost::numeric::ublas;
		namespace ublasx = ::boost::numeric::ublasx;

		size_type n_obs = u.size(); // # of samples
		size_type n_a = ublasx::size(a_); // # of output channels
		size_type n_b = ublasx::size(b_); // # of input channels
		//size_type k_min = ::std::max(n_a*ts_, (n_b+d_)*ts_)-1;
		//size_type k_min = (n_b > 0) ? ((n_b+d_)*ts_-1) : 0;
		size_type k_min = 0;
		//size_type k_max = n_obs - ::std::min(d_*ts_, n_obs);
		size_type k_max = n_obs - ::std::min(static_cast<size_type>(d_*ts_), n_obs);

		vector_type y(n_obs, na_value);

//		// Set non-computable outputs to the given N/A value
//		if (na_value != real_type(0))
//		{
//			// Actually, this setting is done only if the N/A value
//			// is different from the default value '0'.
//			for (size_type k = 0; k < k_min; ++k)
//			{
//				y(k) = na_value;
//			}
//		}
		for (size_type k = k_min; k < k_max; ++k)
		{
			y(k) = 0;

			if (n_a > 0 && k > 0)
			{
				size_type nn_a = ::std::min(n_a, k);

				y(k) -=	ublas::inner_prod(
						//a_,
						ublas::subrange(a_, 0, nn_a),
						//ublas::subslice(y, k-1, -1, n_a)
						ublas::subslice(y, k-1, -1, nn_a)
					);
			}

			if (n_b > 0)
			{
				size_type nn_b = ::std::min(n_b, k+1);

				y(k) += ublas::inner_prod(
						//b_,
						ublas::subrange(b_, 0, nn_b),
						//::dcs::math::la::subslice(u, k, -1, n_b)
						ublas::subslice(u, k, -1, nn_b)
					);
			}
		}

		return y;
	}


	public: vector_type simulate(vector_type const& u, vector_type const& e, real_type na_value = real_type(0)) const
	{
		namespace ublas = ::boost::numeric::ublas;
		namespace ublasx = ::boost::numeric::ublasx;

		size_type n_obs = u.size();
		size_type n_e = e.size();

		DCS_ASSERT(
			n_obs == n_e,
			throw ::std::logic_error("Size of Input Data and Noise Data does not match.")
		);

		size_type n_a = ublasx::size(a_); // # of output channels
		size_type n_b = ublasx::size(b_); // # of input channels
		size_type k_min = 0;
		size_type k_max = n_obs - ::std::min(static_cast<size_type>(d_*ts_), n_obs);

		vector_type y(n_obs, na_value);

//		for (size_type k = 0; k < k_min; ++k)
//		{
//			y(k) = e_var_*e(k);
//		}
		for (size_type k = k_min; k < k_max; ++k)
		{
			y(k) = 0;

			if (n_a > 0 && k > 0)
			{
				size_type nn_a = ::std::min(n_a, k);

				y(k) -=	ublas::inner_prod(
						//a_,
						ublas::subrange(a_, 0, nn_a),
						//::dcs::math::la::subslice(y, k-1, -1, n_a)
						ublas::subslice(y, k-1, -1, nn_a)
					);
			}

			if (n_b > 0)
			{
				size_type nn_b = ::std::min(n_b, k+1);

				y(k) += ublas::inner_prod(
						//b_,
						ublas::subrange(b_, 0, nn_b),
						//::dcs::math::la::subslice(u, k, -1, n_b)
						ublas::subslice(u, k, -1, nn_b)
					);
			}

			y(k) += e_var_*e(k);
		}

		return y;
	}


	public: uint_type sampling_time() const
	{
		return ts_;
	}


	public: vector_type input_parameters() const
	{
		return b_;
	}


	public: vector_type output_parameters() const
	{
		return a_;
	}


	public: real_type noise_covariance() const
	{
		return e_var_;
	}


	private: vector_type a_;
	private: vector_type b_;
	private: real_type e_var_;
	private: uint_type d_;
	private: uint_type ts_;
};


template <
	typename VectorT,
	typename RealT,
	typename UIntT
>
::std::pair<UIntT,UIntT> order(darx_siso_model<VectorT,RealT,UIntT> const& model)
{
	return model.order();
}


template <
	typename VectorT,
	typename RealT,
	typename UIntT
>
UIntT delay(darx_siso_model<VectorT,RealT,UIntT> const& model)
{
	return model.delay();
}


template <
	typename VectorT,
	typename RealT,
	typename UIntT
>
UIntT sampling_time(darx_siso_model<VectorT,RealT,UIntT> const& model)
{
	return model.sampling_time();
}


template <
	typename VectorT,
	typename RealT,
	typename UIntT
>
VectorT output_parameters(darx_siso_model<VectorT,RealT,UIntT> const& model)
{
	return model.output_parameters();
}


template <
	typename VectorT,
	typename RealT,
	typename UIntT
>
VectorT input_parameters(darx_siso_model<VectorT,RealT,UIntT> const& model)
{
	return model.input_parameters();
}


template <
	typename VectorT,
	typename RealT,
	typename UIntT
>
RealT noise_covariance(darx_siso_model<VectorT,RealT,UIntT> const& model)
{
	return model.noise_covariance();
}


template <
	typename VectorT,
	typename RealT,
	typename UIntT
>
VectorT simulate(darx_siso_model<VectorT,RealT,UIntT> const& model, VectorT const& u, RealT na_value = RealT(0))
{
	return model.simulate(u, na_value);
}


template <
	typename VectorT,
	typename RealT,
	typename UIntT
>
VectorT simulate(darx_siso_model<VectorT,RealT,UIntT> const& model, VectorT const& u, VectorT const& e, RealT na_value = RealT(0))
{
	return model.simulate(u, e, na_value);
}


template <
	typename CharT,
	typename CharTraitsT,
	typename VectorT,
	typename RealT,
	typename UIntT
>
::std::basic_ostream<CharT,CharTraitsT>& operator<<(::std::basic_ostream<CharT,CharTraitsT>& os, darx_siso_model<VectorT,RealT,UIntT> const& model)
{
	::std::pair<UIntT,UIntT> ord = order(model);

	os << "<SISO ARX model y(k)+a_1*y(k-t_s)+...+a_{n_a}*y(k-n_a*t_s)=b_0*u(k-d)+...+b_{n_b}*u(k-d-n_b*t_s):"
		<<  " t_s=" << sampling_time(model)
		<< ", n_a=" << ord.first
		<< ", n_b=" << ord.second
		<< ", d=" << delay(model)
		<< ", a=" << output_parameters(model)
		<< ", b=" << input_parameters(model)
		<< ">";

	return os;
}

}} // Namespace dcs::sysid


#endif // DCS_SYSID_MODEL_DARX_SISO_HPP
