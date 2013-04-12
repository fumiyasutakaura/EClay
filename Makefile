UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
# > Mac ------------------------------------------------------------------------
OS_CPP_COMPILER = g++
PATH_VECTOR_TYPES_H = /Developer/NVIDIA/CUDA-5.0/include  # vector_type.h path
PATH_NVCC = /usr/local/cuda/bin/nvcc
OS_NVCC_FLAG = -I/usr/local/cuda/include \
               -I/Developer/GPU\ Computing/C/common/inc \
               -I/Developer/GPU\ Computing/shared/inc \
               -m64
OS_GPP_CPP_FLAG = -arch x86_64 \
			      -I/usr/local/cuda/include \
                  -I/Developer/GPU\ Computing/C/common/inc \
                  -I/Developer/GPU\ Computing/shared/inc \
			      -I$(PATH_VECTOR_TYPES_H)
OS_GPP_CPP_CUDA_FLAG = -arch x86_64 \
					-I$(PATH_VECTOR_TYPES_H) \
					-I/usr/local/cuda/include \
	                -I/Developer/GPU\ Computing/C/common/inc \
                    -I/Developer/GPU\ Computing/shared/inc \
                    -L/usr/local/cuda/lib \
                    -L/Developer/GPU\ Computing/C/lib \
                    -L/Developer/GPU\ Computing/shared/lib \
                    -lshrutil_x86_64 -lcudart -lcutil_x86_64 \
                    -Xlinker -rpath /usr/local/cuda/lib \
                    -framework OpenGL -framework GLUT 
# < Mac ------------------------------------------------------------------------
endif
ifeq ($(UNAME), Linux)
# > Linux ------------------------------------------------------------
OS_CPP_COMPILER = /usr/bin/nvcc
PATH_VECTOR_TYPES_H = /usr/include/cuda32  # vector_type.h path
PATH_NVCC = /usr/bin/nvcc
OS_NVCC_FLAG = 
OS_GPP_CPP_FLAG = -I$(PATH_VECTOR_TYPES_H)
OS_GPP_CPP_CUDA_FLAG = -I$(PATH_VECTOR_TYPES_H) \
					   -I/usr/include/cuda32 \
					   -L/usr/lib64/cuda32 \
					   -lglut -lGL -lcutil -lcudart
# < Linux ------------------------------------------------------------
endif


NVCC = $(PATH_NVCC)
NVCC_FLAG = $(OS_NVCC_FLAG)

GPP = $(OS_CPP_COMPILER)
GPP_CPP_FLAG = $(OS_GPP_CPP_FLAG)
GPP_CPP_CUDA_FLAG = $(OS_GPP_CPP_CUDA_FLAG)


# ALL -----------------------------
all: bin bin/gpuECCudaOperator.o bin/gpuECPixelBuffer.o bin/gpuECMatrix.o bin/gpuECVector.o bin/gpuECQuaternion.o bin/gpuECModel.o bin/ECUtil.o bin/ECPixelBuffer.o bin/ECMatrix.o bin/ECVector.o bin/ECQuaternion.o bin/ECModel.o main.cpp bin/ECCanvas.o
	$(GPP) $(GPP_CPP_CUDA_FLAG) -o bin/app main.cpp SampleApplication.cpp bin/gpuECCudaOperator.o bin/gpuECPixelBuffer.o bin/gpuECMatrix.o bin/gpuECVector.o bin/gpuECQuaternion.o bin/gpuECModel.o bin/ECUtil.o bin/ECPixelBuffer.o bin/ECMatrix.o bin/ECVector.o bin/ECQuaternion.o bin/ECModel.o bin/ECCanvas.o


# bin dir -------------------------
bin :
	mkdir bin

# GPU -------------------------
bin/gpuECCudaOperator.o: EClaySDK/cuda_helper/ECCudaOperator.cu
	$(NVCC) $(NVCC_FLAG) -o bin/gpuECCudaOperator.o -c EClaySDK/cuda_helper/ECCudaOperator.cu

bin/gpuECPixelBuffer.o: EClaySDK/renderer/ECPixelBuffer.cu EClaySDK/math/ECMatrix.hpp EClaySDK/math/ECQuaternion.hpp EClaySDK/math/ECVector.hpp EClaySDK/math/ECMath.cuh
	$(NVCC) $(NVCC_FLAG) -o bin/gpuECPixelBuffer.o -c EClaySDK/renderer/ECPixelBuffer.cu

bin/gpuECMatrix.o: EClaySDK/math/ECMatrix.cu EClaySDK/math/ECMatrix.cuh
	$(NVCC) $(NVCC_FLAG) -o bin/gpuECMatrix.o -c EClaySDK/math/ECMatrix.cu

bin/gpuECVector.o: EClaySDK/math/ECVector.cu EClaySDK/math/ECVector.cuh
	$(NVCC) $(NVCC_FLAG) -o bin/gpuECVector.o -c EClaySDK/math/ECVector.cu

bin/gpuECQuaternion.o: EClaySDK/math/ECQuaternion.cu EClaySDK/math/ECQuaternion.cuh
	$(NVCC) $(NVCC_FLAG) -o bin/gpuECQuaternion.o -c EClaySDK/math/ECQuaternion.cu
	
bin/gpuECModel.o: EClaySDK/application/ECModel.cu 
	$(NVCC) $(NVCC_FLAG) -o bin/gpuECModel.o -c EClaySDK/application/ECModel.cu


# CPU -------------------------
bin/ECUtil.o: EClaySDK/etc/ECUtil.cpp EClaySDK/etc/ECUtil.hpp EClaySDK/smart_pointer/ECSmtPtr.hpp
	$(GPP) $(GPP_CPP_FLAG) -o bin/ECUtil.o -c EClaySDK/etc/ECUtil.cpp

bin/ECPixelBuffer.o: EClaySDK/renderer/ECPixelBuffer.cpp EClaySDK/renderer/ECPixelBuffer.hpp EClaySDK/math/ECMath.cuh EClaySDK/smart_pointer/ECSmtPtr.hpp
	$(GPP) $(GPP_CPP_FLAG) -o bin/ECPixelBuffer.o -c EClaySDK/renderer/ECPixelBuffer.cpp

bin/ECCanvas.o: EClaySDK/application/ECCanvas.cpp EClaySDK/application/ECCanvas.hpp EClaySDK/smart_pointer/ECSmtPtr.hpp
	$(GPP) $(GPP_CPP_FLAG) -o bin/ECCanvas.o -c EClaySDK/application/ECCanvas.cpp

bin/ECMatrix.o: EClaySDK/math/ECMatrix.cpp EClaySDK/math/ECMatrix.hpp EClaySDK/smart_pointer/ECSmtPtr.hpp
	$(GPP) $(GPP_CPP_FLAG) -o bin/ECMatrix.o -c EClaySDK/math/ECMatrix.cpp

bin/ECVector.o: EClaySDK/math/ECVector.cpp EClaySDK/math/ECVector.hpp EClaySDK/smart_pointer/ECSmtPtr.hpp
	$(GPP) $(GPP_CPP_FLAG) -o bin/ECVector.o -c EClaySDK/math/ECVector.cpp

bin/ECQuaternion.o: EClaySDK/math/ECQuaternion.cpp EClaySDK/math/ECQuaternion.hpp EClaySDK/smart_pointer/ECSmtPtr.hpp
	$(GPP) $(GPP_CPP_FLAG) -o bin/ECQuaternion.o -c EClaySDK/math/ECQuaternion.cpp

bin/ECModel.o: EClaySDK/application/ECModel.cpp EClaySDK/application/ECModel.hpp EClaySDK/smart_pointer/ECSmtPtr.hpp
	$(GPP) $(GPP_CPP_FLAG) -o bin/ECModel.o -c EClaySDK/application/ECModel.cpp 




clean:
	rm -fr bin
