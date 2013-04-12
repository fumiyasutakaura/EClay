#ifdef __APPLE__
#define EC_ENV_OS_APPLE
#else
#endif


#include <cuda.h>
#include <cutil.h>
#include <cutil_inline.h>

#ifdef EC_ENV_OS_APPLE
#include <shrQATest.h>
#include <shrUtils.h>
#endif



extern "C" {
    
    void gpuInit( int argc, char** argv ) {
#ifdef EC_ENV_OS_APPLE
        shrQAStart(argc, argv);
#endif
        CUT_DEVICE_INIT(argc, argv);
    }
    
    void gpuExit( int argc, char** argv ) {
#ifdef EC_ENV_OS_APPLE
        shrEXIT(argc, (const char**)argv);
#endif
        CUT_EXIT(argc, argv);
    }
    
    
    
    void gpuMalloc( void*& dev_ptr, int size ) {
        CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_ptr, size ) );
    }
    
    void gpuFree( void*& dev_ptr ) {
        CUDA_SAFE_CALL( cudaFree( dev_ptr ) );
        dev_ptr = 0;
    }
    
    void gpuMemcpyDeviceToHost( void*& host_ptr, void*& dev_ptr, int size ) {
        CUDA_SAFE_CALL( cudaMemcpy( host_ptr, dev_ptr, size, cudaMemcpyDeviceToHost ) );
    }
    
    void gpuMemcpyHostToDevice( void*& dev_ptr, void*& host_ptr, int size ) {
        CUDA_SAFE_CALL( cudaMemcpy( dev_ptr, host_ptr, size, cudaMemcpyHostToDevice ) );
    }
    
    void gpuMemcpyDeviceToDevice( void*& to, void*& from, int size ) {
        CUDA_SAFE_CALL( cudaMemcpy( to, from, size, cudaMemcpyDeviceToDevice ) );
    }
    
    void gpuBindTexture( const texture<float>& texture, const float* texElmArray, const int size ) {
        CUDA_SAFE_CALL( cudaBindTexture( NULL, texture, texElmArray, size ) );
    }
    void gpuUnbindTexture( const texture<float>& texture ) {
        CUDA_SAFE_CALL( cudaUnbindTexture( texture ) );
    }
    
}

