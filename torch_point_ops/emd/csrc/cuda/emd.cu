#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Atomic.cuh>
#include <torch/all.h>
#include <torch/library.h>
#include <vector>
#include <tuple>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/OpMathType.h>
#include <c10/cuda/CUDAMathCompat.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

namespace torch_point_ops {

template<typename scalar_t>
__global__ void approxmatch(int b,int n,int m,const scalar_t * __restrict__ xyz1,const scalar_t * __restrict__ xyz2,scalar_t * __restrict__ match,scalar_t * temp){
	using opmath_t = at::opmath_type<scalar_t>;
	scalar_t * remainL=temp+blockIdx.x*(n+m)*2, * remainR=temp+blockIdx.x*(n+m)*2+n,*ratioL=temp+blockIdx.x*(n+m)*2+n+m,*ratioR=temp+blockIdx.x*(n+m)*2+n+m+n;
	scalar_t multiL,multiR;
	if (n>=m){
		multiL=1;
		multiR=float(n)/m;
	}else{
		multiL=float(m)/n;
		multiR=1;
	}
	const int Block=1024;
	__shared__ scalar_t buf[Block*4];
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int j=threadIdx.x;j<n*m;j+=blockDim.x)
			match[i*n*m+j]=0;
		for (int j=threadIdx.x;j<n;j+=blockDim.x)
			remainL[j]=multiL;
		for (int j=threadIdx.x;j<m;j+=blockDim.x)
			remainR[j]=multiR;
		__syncthreads();
		for (int j=7;j>=-2;j--){
			opmath_t level=-::pow(static_cast<opmath_t>(4.0f), static_cast<opmath_t>(j));
			if (j==-2){
				level=0;
			}
			

			
			for (int k0=0;k0<n;k0+=blockDim.x){
				int k=k0+threadIdx.x;
				opmath_t x1=0,y1=0,z1=0;
				if (k<n){
					x1=static_cast<opmath_t>(xyz1[i*n*3+k*3+0]);
					y1=static_cast<opmath_t>(xyz1[i*n*3+k*3+1]);
					z1=static_cast<opmath_t>(xyz1[i*n*3+k*3+2]);
				}
				opmath_t suml = std::is_same_v<scalar_t, at::Half> ? static_cast<opmath_t>(1e-4f) : static_cast<opmath_t>(1e-9f);
				for (int l0=0;l0<m;l0+=Block){
					int lend=min(m,l0+Block)-l0;
					for (int l=threadIdx.x;l<lend;l+=blockDim.x){
						scalar_t x2=xyz2[i*m*3+l0*3+l*3+0];
						scalar_t y2=xyz2[i*m*3+l0*3+l*3+1];
						scalar_t z2=xyz2[i*m*3+l0*3+l*3+2];
						buf[l*4+0]=x2;
						buf[l*4+1]=y2;
						buf[l*4+2]=z2;
						buf[l*4+3]=remainR[l0+l];
					}
					__syncthreads();
					for (int l=0;l<lend;l++){
						opmath_t x2=static_cast<opmath_t>(buf[l*4+0]);
						opmath_t y2=static_cast<opmath_t>(buf[l*4+1]);
						opmath_t z2=static_cast<opmath_t>(buf[l*4+2]);
						opmath_t d=level*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1));
						

						
						// Clamp exponential argument only for float16 to prevent extreme underflow/overflow
						if (std::is_same_v<scalar_t, at::Half>) {
							d = d < static_cast<opmath_t>(-20.0f) ? static_cast<opmath_t>(-20.0f) : 
							    (d > static_cast<opmath_t>(20.0f) ? static_cast<opmath_t>(20.0f) : d);
						}
						
						opmath_t w=::exp(d)*static_cast<opmath_t>(buf[l*4+3]);
						
						suml+=w;
					}
					__syncthreads();
				}
				if (k<n) {
					opmath_t epsilon = std::is_same_v<scalar_t, at::Half> ? static_cast<opmath_t>(1e-4f) : static_cast<opmath_t>(1e-9f);
					opmath_t ratio = static_cast<opmath_t>(remainL[k])/(suml + epsilon);
					

					
					ratioL[k]=static_cast<scalar_t>(ratio);
				}
			}
			__syncthreads();
			for (int l0=0;l0<m;l0+=blockDim.x){
				int l=l0+threadIdx.x;
				opmath_t x2=0,y2=0,z2=0;
				if (l<m){
					x2=static_cast<opmath_t>(xyz2[i*m*3+l*3+0]);
					y2=static_cast<opmath_t>(xyz2[i*m*3+l*3+1]);
					z2=static_cast<opmath_t>(xyz2[i*m*3+l*3+2]);
				}
				opmath_t sumr=0;
				for (int k0=0;k0<n;k0+=Block){
					int kend=min(n,k0+Block)-k0;
					for (int k=threadIdx.x;k<kend;k+=blockDim.x){
						buf[k*4+0]=xyz1[i*n*3+k0*3+k*3+0];
						buf[k*4+1]=xyz1[i*n*3+k0*3+k*3+1];
						buf[k*4+2]=xyz1[i*n*3+k0*3+k*3+2];
						buf[k*4+3]=ratioL[k0+k];
					}
					__syncthreads();
					for (int k=0;k<kend;k++){
						opmath_t x1=static_cast<opmath_t>(buf[k*4+0]);
						opmath_t y1=static_cast<opmath_t>(buf[k*4+1]);
						opmath_t z1=static_cast<opmath_t>(buf[k*4+2]);
						opmath_t d=level*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1));
						// Clamp exponential argument only for float16 to prevent extreme underflow/overflow
						if (std::is_same_v<scalar_t, at::Half>) {
							d = d < static_cast<opmath_t>(-20.0f) ? static_cast<opmath_t>(-20.0f) : 
							    (d > static_cast<opmath_t>(20.0f) ? static_cast<opmath_t>(20.0f) : d);
						}
						opmath_t w=::exp(d)*static_cast<opmath_t>(buf[k*4+3]);
						sumr+=w;
					}
					__syncthreads();
				}
				if (l<m){
					sumr*=static_cast<opmath_t>(remainR[l]);
					opmath_t epsilon = std::is_same_v<scalar_t, at::Half> ? static_cast<opmath_t>(1e-4f) : static_cast<opmath_t>(1e-9f);
					opmath_t consumption=fminf(static_cast<opmath_t>(remainR[l])/(sumr+epsilon),1.0f);
					ratioR[l]=static_cast<scalar_t>(consumption*static_cast<opmath_t>(remainR[l]));
					remainR[l]=static_cast<scalar_t>(fmaxf(0.0f,static_cast<opmath_t>(remainR[l])-sumr));
				}
			}
			__syncthreads();
			for (int k0=0;k0<n;k0+=blockDim.x){
				int k=k0+threadIdx.x;
				opmath_t x1=0,y1=0,z1=0;
				if (k<n){
					x1=static_cast<opmath_t>(xyz1[i*n*3+k*3+0]);
					y1=static_cast<opmath_t>(xyz1[i*n*3+k*3+1]);
					z1=static_cast<opmath_t>(xyz1[i*n*3+k*3+2]);
				}
				opmath_t suml=0;
				for (int l0=0;l0<m;l0+=Block){
					int lend=min(m,l0+Block)-l0;
					for (int l=threadIdx.x;l<lend;l+=blockDim.x){
						buf[l*4+0]=xyz2[i*m*3+l0*3+l*3+0];
						buf[l*4+1]=xyz2[i*m*3+l0*3+l*3+1];
						buf[l*4+2]=xyz2[i*m*3+l0*3+l*3+2];
						buf[l*4+3]=ratioR[l0+l];
					}
					__syncthreads();
					opmath_t rl=static_cast<opmath_t>(ratioL[k]);
					if (k<n){
						for (int l=0;l<lend;l++){
							opmath_t x2=static_cast<opmath_t>(buf[l*4+0]);
							opmath_t y2=static_cast<opmath_t>(buf[l*4+1]);
							opmath_t z2=static_cast<opmath_t>(buf[l*4+2]);
							opmath_t d=level*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1));
							// Clamp exponential argument only for float16 to prevent extreme underflow/overflow
							if (std::is_same_v<scalar_t, at::Half>) {
								d = d < static_cast<opmath_t>(-20.0f) ? static_cast<opmath_t>(-20.0f) : 
								    (d > static_cast<opmath_t>(20.0f) ? static_cast<opmath_t>(20.0f) : d);
							}
							opmath_t w=::exp(d)*rl*static_cast<opmath_t>(buf[l*4+3]);
							match[i*n*m+(l0+l)*n+k]+=static_cast<scalar_t>(w);
							suml+=w;
						}
					}
					__syncthreads();
				}
				if (k<n)
					remainL[k]=static_cast<scalar_t>(fmaxf(0.0f,static_cast<opmath_t>(remainL[k])-suml));
			}
			__syncthreads();
		}
	}
}


template<typename scalar_t>
__global__ void matchcost(int b,int n,int m,const scalar_t * __restrict__ xyz1,const scalar_t * __restrict__ xyz2,const scalar_t * __restrict__ match,scalar_t * __restrict__ out){
	using opmath_t = at::opmath_type<scalar_t>;
	__shared__ opmath_t allsum[512];
	const int Block=1024;
	__shared__ scalar_t buf[Block*3];
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		opmath_t subsum=0;
		for (int k0=0;k0<n;k0+=blockDim.x){
			int k=k0+threadIdx.x;
			opmath_t x1=0,y1=0,z1=0;
			if (k<n){
				x1=static_cast<opmath_t>(xyz1[i*n*3+k*3+0]);
				y1=static_cast<opmath_t>(xyz1[i*n*3+k*3+1]);
				z1=static_cast<opmath_t>(xyz1[i*n*3+k*3+2]);
			}
			for (int l0=0;l0<m;l0+=Block){
				int lend=min(m,l0+Block)-l0;
				for (int l=threadIdx.x;l<lend*3;l+=blockDim.x)
					buf[l]=xyz2[i*m*3+l0*3+l];
				__syncthreads();
				if (k<n){
					for (int l=0;l<lend;l++){
						opmath_t x2=static_cast<opmath_t>(buf[l*3+0]);
						opmath_t y2=static_cast<opmath_t>(buf[l*3+1]);
						opmath_t z2=static_cast<opmath_t>(buf[l*3+2]);
						opmath_t d=(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
						subsum+=d*static_cast<opmath_t>(match[i*n*m+(l0+l)*n+k]);
					}
				}
				__syncthreads();
			}
		}
		allsum[threadIdx.x]=subsum;
		for (int j=1;j<blockDim.x;j<<=1){
			__syncthreads();
			if ((threadIdx.x&j)==0 && threadIdx.x+j<blockDim.x){
				allsum[threadIdx.x]+=allsum[threadIdx.x+j];
			}
		}
		if (threadIdx.x==0)
			out[i]=static_cast<scalar_t>(allsum[0]);
		__syncthreads();
	}
}


template<typename scalar_t>
__global__ void matchcostgrad1(int b,int n,int m,const scalar_t * __restrict__ grad_cost,const scalar_t * __restrict__ xyz1,const scalar_t * __restrict__ xyz2,const scalar_t * __restrict__ match,scalar_t * __restrict__ grad1){
	using opmath_t = at::opmath_type<scalar_t>;
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int l=threadIdx.x;l<n;l+=blockDim.x){
			opmath_t x1=static_cast<opmath_t>(xyz1[i*n*3+l*3+0]);
			opmath_t y1=static_cast<opmath_t>(xyz1[i*n*3+l*3+1]);
			opmath_t z1=static_cast<opmath_t>(xyz1[i*n*3+l*3+2]);
			opmath_t dx=0,dy=0,dz=0;
			for (int k=0;k<m;k++){
				opmath_t x2=static_cast<opmath_t>(xyz2[i*m*3+k*3+0]);
				opmath_t y2=static_cast<opmath_t>(xyz2[i*m*3+k*3+1]);
				opmath_t z2=static_cast<opmath_t>(xyz2[i*m*3+k*3+2]);
				opmath_t d=static_cast<opmath_t>(match[i*n*m+k*n+l])*2;
				dx+=(x1-x2)*d;
				dy+=(y1-y2)*d;
				dz+=(z1-z2)*d;
			}
			opmath_t grad_cost_i = static_cast<opmath_t>(grad_cost[i]);
			grad1[i*n*3+l*3+0]=static_cast<scalar_t>(dx*grad_cost_i);
			grad1[i*n*3+l*3+1]=static_cast<scalar_t>(dy*grad_cost_i);
			grad1[i*n*3+l*3+2]=static_cast<scalar_t>(dz*grad_cost_i);
		}
	}
}


template<typename scalar_t>
__global__ void matchcostgrad2(int b,int n,int m,const scalar_t * __restrict__ grad_cost,const scalar_t * __restrict__ xyz1,const scalar_t * __restrict__ xyz2,const scalar_t * __restrict__ match,scalar_t * __restrict__ grad2){
	using opmath_t = at::opmath_type<scalar_t>;
	__shared__ opmath_t sum_grad[256*3];
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		int kbeg=m*blockIdx.y/gridDim.y;
		int kend=m*(blockIdx.y+1)/gridDim.y;
		for (int k=kbeg;k<kend;k++){
			opmath_t x2=static_cast<opmath_t>(xyz2[(i*m+k)*3+0]);
			opmath_t y2=static_cast<opmath_t>(xyz2[(i*m+k)*3+1]);
			opmath_t z2=static_cast<opmath_t>(xyz2[(i*m+k)*3+2]);
			opmath_t subsumx=0,subsumy=0,subsumz=0;
			for (int j=threadIdx.x;j<n;j+=blockDim.x){
				opmath_t x1=x2-static_cast<opmath_t>(xyz1[(i*n+j)*3+0]);
				opmath_t y1=y2-static_cast<opmath_t>(xyz1[(i*n+j)*3+1]);
				opmath_t z1=z2-static_cast<opmath_t>(xyz1[(i*n+j)*3+2]);
				opmath_t d=static_cast<opmath_t>(match[i*n*m+k*n+j])*2;
				subsumx+=x1*d;
				subsumy+=y1*d;
				subsumz+=z1*d;
			}
			sum_grad[threadIdx.x*3+0]=subsumx;
			sum_grad[threadIdx.x*3+1]=subsumy;
			sum_grad[threadIdx.x*3+2]=subsumz;
			for (int j=1;j<blockDim.x;j<<=1){
				__syncthreads();
				int j1=threadIdx.x;
				int j2=threadIdx.x+j;
				if ((j1&j)==0 && j2<blockDim.x){
					sum_grad[j1*3+0]+=sum_grad[j2*3+0];
					sum_grad[j1*3+1]+=sum_grad[j2*3+1];
					sum_grad[j1*3+2]+=sum_grad[j2*3+2];
				}
			}
			if (threadIdx.x==0){
				opmath_t grad_cost_i = static_cast<opmath_t>(grad_cost[i]);
				grad2[(i*m+k)*3+0]=static_cast<scalar_t>(sum_grad[0]*grad_cost_i);
				grad2[(i*m+k)*3+1]=static_cast<scalar_t>(sum_grad[1]*grad_cost_i);
				grad2[(i*m+k)*3+2]=static_cast<scalar_t>(sum_grad[2]*grad_cost_i);
			}
			__syncthreads();
		}
	}
}

std::tuple<torch::Tensor, torch::Tensor>
emd_cuda_forward(torch::Tensor xyz1, torch::Tensor xyz2) {
    CHECK_INPUT(xyz1);
    CHECK_INPUT(xyz2);
    TORCH_CHECK(xyz1.dim() == 3 && xyz2.dim() == 3, "Input tensors must be 3D");
    TORCH_CHECK(xyz1.size(0) == xyz2.size(0), "Input tensors must have the same batch size");
    TORCH_CHECK(xyz1.size(2) == 3 && xyz2.size(2) == 3, "Input tensors must have 3 channels");

    if (xyz1.size(1) == 0 || xyz2.size(1) == 0) {
        return {
            torch::zeros({xyz1.size(0)}, xyz1.options()),
            torch::zeros({xyz1.size(0), xyz2.size(1), xyz1.size(1)}, xyz1.options())
        };
    }

    const int b = xyz1.size(0);
    const int n = xyz1.size(1);
    const int m = xyz2.size(1);

    auto match = torch::zeros({b, m, n}, xyz1.options());
    auto temp = torch::zeros({b, (n+m)*2}, xyz1.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(xyz1.scalar_type(), "emd_forward_approxmatch", ([&] {
        approxmatch<scalar_t><<<32,512>>>(
            b, n, m, xyz1.data_ptr<scalar_t>(), xyz2.data_ptr<scalar_t>(), match.data_ptr<scalar_t>(), temp.data_ptr<scalar_t>()
        );
    }));

    auto cost = torch::zeros({b}, xyz1.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(xyz1.scalar_type(), "emd_forward_matchcost", ([&] {
        matchcost<scalar_t><<<32,512>>>(
            b, n, m, xyz1.data_ptr<scalar_t>(), xyz2.data_ptr<scalar_t>(), match.data_ptr<scalar_t>(), cost.data_ptr<scalar_t>()
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in emd_cuda_forward: %s\n", cudaGetErrorString(err));
    }

    return {cost, match};
}

std::tuple<torch::Tensor, torch::Tensor>
emd_cuda_backward(
    torch::Tensor grad_cost,
    torch::Tensor xyz1,
    torch::Tensor xyz2,
    torch::Tensor match
) {
    CHECK_INPUT(grad_cost);
    CHECK_INPUT(xyz1);
    CHECK_INPUT(xyz2);
    CHECK_INPUT(match);

    const int b = xyz1.size(0);
    const int n = xyz1.size(1);
    const int m = xyz2.size(1);

    if (n == 0 || m == 0) {
        auto grad1 = torch::zeros_like(xyz1);
        auto grad2 = torch::zeros_like(xyz2);
        return {grad1, grad2};
    }

    auto grad1 = torch::zeros_like(xyz1);
    auto grad2 = torch::zeros_like(xyz2);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(xyz1.scalar_type(), "emd_backward", ([&] {
        matchcostgrad1<scalar_t><<<32,512>>>(
            b, n, m, grad_cost.data_ptr<scalar_t>(), xyz1.data_ptr<scalar_t>(), xyz2.data_ptr<scalar_t>(), match.data_ptr<scalar_t>(), grad1.data_ptr<scalar_t>()
        );
        matchcostgrad2<scalar_t><<<dim3(32,32),256>>>(
            b, n, m, grad_cost.data_ptr<scalar_t>(), xyz1.data_ptr<scalar_t>(), xyz2.data_ptr<scalar_t>(), match.data_ptr<scalar_t>(), grad2.data_ptr<scalar_t>()
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in emd_cuda_backward: %s\n", cudaGetErrorString(err));
    }

    return {grad1, grad2};
}


TORCH_LIBRARY_IMPL(torch_point_ops_emd, CUDA, m) {
    m.impl("emd_forward", &emd_cuda_forward);
    m.impl("emd_backward", &emd_cuda_backward);
}

} // namespace torch_point_ops 