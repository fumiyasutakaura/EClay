#include "ECMatrix.cuh"


extern "C" void devSetMatrix( ECSmtDevPtr<ECDevMatrix>& dest, ECMatrix& src ) {
    int matrixSize = sizeof(ECDevMatrix);
    ECDevMatrix *hostMatrix = (ECDevMatrix*)malloc( matrixSize ); {
        *hostMatrix = src;
        dest = ECSmtDevPtr<ECDevMatrix>( hostMatrix, matrixSize );
    } free( hostMatrix );
}
extern "C" void devCopyMatrix( ECSmtDevPtr<ECDevMatrix>& dest, ECMatrix& src ) {
    int matrixSize = sizeof(ECDevMatrix);
    ECDevMatrix *hostMatrix = (ECDevMatrix*)malloc( matrixSize ); {
        hostMatrix->set( src );
        dest.copyHostToDevice( hostMatrix );
    } free( hostMatrix );
}
