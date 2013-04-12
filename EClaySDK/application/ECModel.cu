#include <string>
#include <map>

#include "../math/ECMath.cuh"
#include "../renderer/ECPixelBuffer.hpp"
#include "../etc/ECUtil.hpp"
#include "ECModel.hpp"

#include "../renderer/ECRenderState.hpp"

#define PERSPECTIVE_CORRECTION (true)



// > textrue -------------------------
texture<float> colorTexture;
texture<float> normalTexture;
texture<float> specularTexture;
// < textrue -------------------------


extern "C" void devMatrixInit( ECModel* model ) {
    
    int matrixSize = sizeof(ECDevMatrix);
    ECDevMatrix *host_identityMatrix = (ECDevMatrix*)malloc( matrixSize );
    host_identityMatrix->identify();
    {
        model->devWorldMatrix = ECSmtDevPtr<ECDevMatrix>( host_identityMatrix, matrixSize );
        model->devViewMatrix = ECSmtDevPtr<ECDevMatrix>( host_identityMatrix, matrixSize );
        model->devProjectionMatrix = ECSmtDevPtr<ECDevMatrix>( host_identityMatrix, matrixSize );
        model->devScreenMatrix = ECSmtDevPtr<ECDevMatrix>( host_identityMatrix, matrixSize );
    }
    free( host_identityMatrix );
    
}



__device__ ECDevVertex interpolate( const float4& pos, const ECDevVertex& ver1, const ECDevVertex& ver2, const ECDevVertex& ver3 ) {
    ECDevVertex ret;
    const float4 to1 = ver1.position - pos;
    const float4 to2 = ver2.position - pos;
    const float4 to3 = ver3.position - pos;
    const float s1 = abs( det2D(to2,to3) );
    const float s2 = abs( det2D(to3,to1) );
    const float s3 = abs( det2D(to1,to2) );
    const float reciprocal_sum = 1.0f/(s1+s2+s3);
    const float w1 = s1 * reciprocal_sum;
    const float w2 = s2 * reciprocal_sum;
    const float w3 = s3 * reciprocal_sum;
    
    // position  -----------------
    ret.position = ver1.position*w1 + ver2.position*w2 + ver3.position*w3;
    
    // normal --------------------
    ret.normal = normalize3D(ver1.normal*w1 + ver2.normal*w2 + ver3.normal*w3);
    
    // uv ------------------------
#if (PERSPECTIVE_CORRECTION == false)  // without perspective correction
    ret.uv = ver1.uv*w1 + ver2.uv*w2 + ver3.uv*w3;
#else // with perspective correction
    ret.uv = (ver1.uv/ver1.uv.w)*w1 + (ver2.uv/ver2.uv.w)*w2 + (ver3.uv/ver3.uv.w)*w3;
    const float reciprocal_uv_z = 1.0f / ret.uv.z;
    ret.uv.x = ret.uv.x * reciprocal_uv_z;
    ret.uv.y = ret.uv.y * reciprocal_uv_z;
#endif
    
    // color ---------------------
    ret.color = ver1.color*w1 + ver2.color*w2 + ver3.color*w3;
    if(ret.color.w>0.99999f) { ret.color.w = 1.0f; }
    
    return ret;
}


__global__ void pixelShade( ECPixel* pixels,
                                     ECDevVertex* vertexes, int num_of_vertexes,
                                     ECDevMatrix* worldM,
                                     ECDevMatrix* viewM,
                                     ECDevMaterial* material,
                                     float4* zBuffer,
                                     ECDevLight* light ) {
    
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int offset = x + y * blockDim.x * gridDim.x;
    const float4 point = make_float4(x,y,1.0f,1.0f);
    
    for( int i=0; i<num_of_vertexes ; i+=3 ) {
        if( devInTheTriangleCounterClockwise( &point, vertexes[i].position, vertexes[i+1].position, vertexes[i+2].position ) ) {
            ECDevVertex interpolatedVertex = interpolate( point, vertexes[i], vertexes[i+1], vertexes[i+2] );
            if( zBuffer[offset].x < interpolatedVertex.position.z ) {
                
                // ----------------------------------------
                //float4 color = interpolatedVertex.uv;
                const int color_texColorW = material->colorTextureSize.x*4;
                const int color_texX = color_texColorW * interpolatedVertex.uv.x;
                const int color_texY = material->colorTextureSize.y-material->colorTextureSize.y * interpolatedVertex.uv.y;
                const int color_textureOffsetR = ((color_texX + color_texColorW * color_texY)/4) * 4;
                const float colorR = tex1Dfetch( colorTexture, color_textureOffsetR   );
                const float colorG = tex1Dfetch( colorTexture, color_textureOffsetR+1 );
                const float colorB = tex1Dfetch( colorTexture, color_textureOffsetR+2 );
                const float colorA = tex1Dfetch( colorTexture, color_textureOffsetR+3 );
                float4 color = make_float4( colorR, colorG, colorB, colorA );
                
//                const int normal_texColorW = material->normalTextureSize.x*4;
//                const int normal_texX = normal_texColorW * interpolatedVertex.uv.x;
//                const int normal_texY = material->normalTextureSize.y-material->normalTextureSize.y * interpolatedVertex.uv.y;
//                const int normal_textureOffsetR = ((normal_texX + normal_texColorW * normal_texY)/4) * 4;
//                const float normalR = tex1Dfetch( normalTexture, normal_textureOffsetR );
//                const float normalG = tex1Dfetch( normalTexture, normal_textureOffsetR+1 );
//                const float normalB = tex1Dfetch( normalTexture, normal_textureOffsetR+2 );
//                const float normalA = tex1Dfetch( normalTexture, normal_textureOffsetR+3 );
//                const float4 scale_1 = make_float4(1.0f,1.0f,1.0f,1.0f);
                
                float4 Norm_nor = interpolatedVertex.normal;
                
                // ----------------------------------------
                
                
                // element
                //const float4 lightDir_nor = normalize3D(-light->direction);
                const float4 lightDir_nor = normalize3D( interpolatedVertex.position - light->position );
                const float dotNL = dot3D(normalize3D(Norm_nor), lightDir_nor);
                const float4 halfVec_nor = normalize3D( normalize3D(interpolatedVertex.position) - lightDir_nor );
                // ambient
                float4 ambient = light->ambient * material->ambient;
                // diffuse
                float4 diffuse = max(0.0f,dotNL) * (light->diffuse * material->diffuse);
                // specular
                float4 specular = (dotNL<=0.0)?make_float4(0.0f,0.0f,0.0f,1.0f):powf( max(0.0f,-dot3D(Norm_nor,halfVec_nor)), material->shininess.x ) * (light->specular * material->specular);
                
                
                float4 reflection = ambient + diffuse + specular;
                reflection.w = 1.0f;
                
                if( interpolatedVertex.color.w >= 1.0f ) {
                    
                    pixels[offset].r = color.x * reflection.x;
                    pixels[offset].g = color.y * reflection.y;
                    pixels[offset].b = color.z * reflection.z;
                    pixels[offset].a = color.w * reflection.w;
                    
                    zBuffer[offset].x = interpolatedVertex.position.z;
                }
                else { // TODO: 各種アルファブレンディングに対応する
                    pixels[offset].r += color.x * reflection.x;
                    pixels[offset].g += color.y * reflection.y;
                    pixels[offset].b += color.z * reflection.z;
                    pixels[offset].a += color.w * reflection.w;
                }
            }
        }
    }
    
}

__global__ void pixelShade_perVertex( ECPixel* pixels,
                                               ECDevVertex* vertexes, int index,
                                               int origin_x, int origin_y,
                                               int screenWidth,
                                               ECDevMatrix* worldM,
                                               ECDevMatrix* viewM,
                                               ECDevMaterial* material,
                                               float4* zBuffer,
                                               ECDevLight* light ) {
    
    const int x = threadIdx.x + blockIdx.x * blockDim.x + origin_x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y + origin_y;
    const int offset = x + y * screenWidth;
    const float4 point = make_float4(x,y,1.0f,1.0f);
    
    if( devInTheTriangleCounterClockwise( &point, vertexes[index].position, vertexes[index+1].position, vertexes[index+2].position ) ) {
        ECDevVertex interpolatedVertex = interpolate( point, vertexes[index], vertexes[index+1], vertexes[index+2] );
        if( zBuffer[offset].x < interpolatedVertex.position.z ) {
            
            // ----------------------------------------
            const int color_texColorW = material->colorTextureSize.x*4;
            const int color_texX = color_texColorW * interpolatedVertex.uv.x;
            const int color_texY = material->colorTextureSize.y-material->colorTextureSize.y * interpolatedVertex.uv.y;
            const int color_textureOffsetR = ((color_texX + color_texColorW * color_texY)/4) * 4;
            const float colorR = tex1Dfetch( colorTexture, color_textureOffsetR   );
            const float colorG = tex1Dfetch( colorTexture, color_textureOffsetR+1 );
            const float colorB = tex1Dfetch( colorTexture, color_textureOffsetR+2 );
            const float colorA = tex1Dfetch( colorTexture, color_textureOffsetR+3 );
            float4 color = make_float4( colorR, colorG, colorB, colorA );
            
//            const int normal_texColorW = material->normalTextureSize.x*4;
//            const int normal_texX = normal_texColorW * interpolatedVertex.uv.x;
//            const int normal_texY = material->normalTextureSize.y-material->normalTextureSize.y * interpolatedVertex.uv.y;
//            const int normal_textureOffsetR = ((normal_texX + normal_texColorW * normal_texY)/4) * 4;
//            const float normalR = tex1Dfetch( normalTexture, normal_textureOffsetR );
//            const float normalG = tex1Dfetch( normalTexture, normal_textureOffsetR+1 );
//            const float normalB = tex1Dfetch( normalTexture, normal_textureOffsetR+2 );
//            const float normalA = tex1Dfetch( normalTexture, normal_textureOffsetR+3 );
//            const float4 scale_1 = make_float4(1.0f,1.0f,1.0f,1.0f);
            
            // > normal ----------------------
//            float4 Norm_nor = normalize3D( getRotateMatrix(viewM,&scale_1) * (getRotateMatrix(worldM,&scale_1) * (ECDevQuaternion( interpolatedVertex.normal, make_float4(0,0,1,0) ) * normalize3D( make_float4( normalR, normalG, normalB, normalA ) ) ) ) );
            float4 Norm_nor = interpolatedVertex.normal;
            //color = Norm_nor;
            // < normal ----------------------
            
            // ----------------------------------------
            
            
            // element
            //const float4 lightDir_nor = normalize3D(-light->direction);
            const float4 lightDir_nor = normalize3D(-light->direction);//normalize3D( interpolatedVertex.position - light->position );
            const float dotNL = dot3D(normalize3D(Norm_nor), lightDir_nor);
            const float4 halfVec_nor = normalize3D( normalize3D(interpolatedVertex.position) - lightDir_nor );
            // ambient
            float4 ambient = light->ambient * material->ambient;
            // diffuse
            float4 diffuse = max(0.0f,dotNL) * (light->diffuse * material->diffuse);
            // specular
            float4 specular = (dotNL<=0.0)?make_float4(0.0f,0.0f,0.0f,1.0f):powf( max(0.0f,-dot3D(Norm_nor,halfVec_nor)), material->shininess.x ) * (light->specular * material->specular);
            
            
            float4 reflection = ambient + diffuse + specular;
            reflection.w = 1.0f;
            
            if( interpolatedVertex.color.w >= 1.0f ) {
                
                pixels[offset].r = color.x * reflection.x;
                pixels[offset].g = color.y * reflection.y;
                pixels[offset].b = color.z * reflection.z;
                pixels[offset].a = color.w * reflection.w;
                
                zBuffer[offset].x = interpolatedVertex.position.z;
            }
            else { // TODO: 各種アルファブレンディングに対応する
                pixels[offset].r += color.x * reflection.x;
                pixels[offset].g += color.y * reflection.y;
                pixels[offset].b += color.z * reflection.z;
                pixels[offset].a += color.w * reflection.w;
            }
        }
    }
    
}


__global__ void pixelShade_NormalMap( ECPixel* pixels,
                                    ECDevVertex* vertexes, int num_of_vertexes,
                                    ECDevMatrix* worldM,
                                    ECDevMatrix* viewM,
                                    ECDevMaterial* material,
                                    float4* zBuffer,
                                    ECDevLight* light ) {
    
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int offset = x + y * blockDim.x * gridDim.x;
    const float4 point = make_float4(x,y,1.0f,1.0f);
    
    for( int i=0; i<num_of_vertexes ; i+=3 ) {
        if( devInTheTriangleCounterClockwise( &point, vertexes[i].position, vertexes[i+1].position, vertexes[i+2].position ) ) {
            ECDevVertex interpolatedVertex = interpolate( point, vertexes[i], vertexes[i+1], vertexes[i+2] );
            if( zBuffer[offset].x < interpolatedVertex.position.z ) {
                
                // ----------------------------------------
                //float4 color = interpolatedVertex.uv;
                const int color_texColorW = material->colorTextureSize.x*4;
                const int color_texX = color_texColorW * interpolatedVertex.uv.x;
                const int color_texY = material->colorTextureSize.y-material->colorTextureSize.y * interpolatedVertex.uv.y;
                const int color_textureOffsetR = ((color_texX + color_texColorW * color_texY)/4) * 4;
                const float colorR = tex1Dfetch( colorTexture, color_textureOffsetR   );
                const float colorG = tex1Dfetch( colorTexture, color_textureOffsetR+1 );
                const float colorB = tex1Dfetch( colorTexture, color_textureOffsetR+2 );
                const float colorA = tex1Dfetch( colorTexture, color_textureOffsetR+3 );
                float4 color = make_float4( colorR, colorG, colorB, colorA );
                
                const int normal_texColorW = material->normalTextureSize.x*4;
                const int normal_texX = normal_texColorW * interpolatedVertex.uv.x;
                const int normal_texY = material->normalTextureSize.y-material->normalTextureSize.y * interpolatedVertex.uv.y;
                const int normal_textureOffsetR = ((normal_texX + normal_texColorW * normal_texY)/4) * 4;
                const float normalR = tex1Dfetch( normalTexture, normal_textureOffsetR );
                const float normalG = tex1Dfetch( normalTexture, normal_textureOffsetR+1 );
                const float normalB = tex1Dfetch( normalTexture, normal_textureOffsetR+2 );
                const float normalA = tex1Dfetch( normalTexture, normal_textureOffsetR+3 );
                const float4 scale_1 = make_float4(1.0f,1.0f,1.0f,1.0f);
                
                // > normal ----------------------
                float4 Norm_nor = normalize3D( getRotateMatrix(viewM,&scale_1) * (getRotateMatrix(worldM,&scale_1) * (ECDevQuaternion( interpolatedVertex.normal, make_float4(0,0,1,0) ) * normalize3D( make_float4( normalR, normalG, normalB, normalA ) ) ) ) );
                //Norm_nor = interpolatedVertex.normal;
                //color = Norm_nor;
                // < normal ----------------------
                
                // ----------------------------------------
                
                
                // element
                //const float4 lightDir_nor = normalize3D(-light->direction);
                const float4 lightDir_nor = normalize3D( interpolatedVertex.position - light->position );
                const float dotNL = dot3D(normalize3D(Norm_nor), lightDir_nor);
                const float4 halfVec_nor = normalize3D( normalize3D(interpolatedVertex.position) - lightDir_nor );
                // ambient
                float4 ambient = light->ambient * material->ambient;
                // diffuse
                float4 diffuse = max(0.0f,dotNL) * (light->diffuse * material->diffuse);
                // specular
                float4 specular = (dotNL<=0.0)?make_float4(0.0f,0.0f,0.0f,1.0f):powf( max(0.0f,-dot3D(Norm_nor,halfVec_nor)), material->shininess.x ) * (light->specular * material->specular);
                
                
                float4 reflection = ambient + diffuse + specular;
                reflection.w = 1.0f;
                
                if( interpolatedVertex.color.w >= 1.0f ) {
                    
                    pixels[offset].r = color.x * reflection.x;
                    pixels[offset].g = color.y * reflection.y;
                    pixels[offset].b = color.z * reflection.z;
                    pixels[offset].a = color.w * reflection.w;
                    
                    zBuffer[offset].x = interpolatedVertex.position.z;
                }
                else { // TODO: 各種アルファブレンディングに対応する
                    pixels[offset].r += color.x * reflection.x;
                    pixels[offset].g += color.y * reflection.y;
                    pixels[offset].b += color.z * reflection.z;
                    pixels[offset].a += color.w * reflection.w;
                }
            }
        }
    }
    
}

__global__ void pixelShade_NormalMap_perVertex( ECPixel* pixels,
                                               ECDevVertex* vertexes, int index,
                                               int origin_x, int origin_y,
                                               int screenWidth,
                                               ECDevMatrix* worldM,
                                               ECDevMatrix* viewM,
                                               ECDevMaterial* material,
                                               float4* zBuffer,
                                               ECDevLight* light ) {
    
    const int x = threadIdx.x + blockIdx.x * blockDim.x + origin_x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y + origin_y;
    const int offset = x + y * screenWidth;
    const float4 point = make_float4(x,y,1.0f,1.0f);
    
    if( devInTheTriangleCounterClockwise( &point, vertexes[index].position, vertexes[index+1].position, vertexes[index+2].position ) ) {
        ECDevVertex interpolatedVertex = interpolate( point, vertexes[index], vertexes[index+1], vertexes[index+2] );
        if( zBuffer[offset].x < interpolatedVertex.position.z ) {
            
            // ----------------------------------------
            const int color_texColorW = material->colorTextureSize.x*4;
            const int color_texX = color_texColorW * interpolatedVertex.uv.x;
            const int color_texY = material->colorTextureSize.y-material->colorTextureSize.y * interpolatedVertex.uv.y;
            const int color_textureOffsetR = ((color_texX + color_texColorW * color_texY)/4) * 4;
            const float colorR = tex1Dfetch( colorTexture, color_textureOffsetR   );
            const float colorG = tex1Dfetch( colorTexture, color_textureOffsetR+1 );
            const float colorB = tex1Dfetch( colorTexture, color_textureOffsetR+2 );
            const float colorA = tex1Dfetch( colorTexture, color_textureOffsetR+3 );
            float4 color = make_float4( colorR, colorG, colorB, colorA );
            
            const int normal_texColorW = material->normalTextureSize.x*4;
            const int normal_texX = normal_texColorW * interpolatedVertex.uv.x;
            const int normal_texY = material->normalTextureSize.y-material->normalTextureSize.y * interpolatedVertex.uv.y;
            const int normal_textureOffsetR = ((normal_texX + normal_texColorW * normal_texY)/4) * 4;
            const float normalR = tex1Dfetch( normalTexture, normal_textureOffsetR );
            const float normalG = tex1Dfetch( normalTexture, normal_textureOffsetR+1 );
            const float normalB = tex1Dfetch( normalTexture, normal_textureOffsetR+2 );
            const float normalA = tex1Dfetch( normalTexture, normal_textureOffsetR+3 );
            const float4 scale_1 = make_float4(1.0f,1.0f,1.0f,1.0f);
            
            // > normal ----------------------
            float4 Norm_nor = normalize3D( getRotateMatrix(viewM,&scale_1) * (getRotateMatrix(worldM,&scale_1) * (ECDevQuaternion( interpolatedVertex.normal, make_float4(0,0,1,0) ) * normalize3D( make_float4( normalR, normalG, normalB, normalA ) ) ) ) );
            //Norm_nor = interpolatedVertex.normal;
            //color = Norm_nor;
            // < normal ----------------------
            
            // ----------------------------------------
            
            
            // element
            //const float4 lightDir_nor = normalize3D(-light->direction);
            const float4 lightDir_nor = normalize3D( interpolatedVertex.position - light->position );
            const float dotNL = dot3D(normalize3D(Norm_nor), lightDir_nor);
            const float4 halfVec_nor = normalize3D( normalize3D(interpolatedVertex.position) - lightDir_nor );
            // ambient
            float4 ambient = light->ambient * material->ambient;
            // diffuse
            float4 diffuse = max(0.0f,dotNL) * (light->diffuse * material->diffuse);
            // specular
            float4 specular = (dotNL<=0.0)?make_float4(0.0f,0.0f,0.0f,1.0f):powf( max(0.0f,-dot3D(Norm_nor,halfVec_nor)), material->shininess.x ) * (light->specular * material->specular);
            
            
            float4 reflection = ambient + diffuse + specular;
            reflection.w = 1.0f;
            
            if( interpolatedVertex.color.w >= 1.0f ) {
                
                pixels[offset].r = color.x * reflection.x;
                pixels[offset].g = color.y * reflection.y;
                pixels[offset].b = color.z * reflection.z;
                pixels[offset].a = color.w * reflection.w;
                
                zBuffer[offset].x = interpolatedVertex.position.z;
            }
            else { // TODO: 各種アルファブレンディングに対応する
                pixels[offset].r += color.x * reflection.x;
                pixels[offset].g += color.y * reflection.y;
                pixels[offset].b += color.z * reflection.z;
                pixels[offset].a += color.w * reflection.w;
            }
        }
    }
    
}

__global__ void vertexShade(ECDevVertex* destVertexes, const ECDevVertex* srcVertexes,
                            const float4* scale,
                            const ECDevMatrix* worldM,
                            const ECDevMatrix* viewM,
                            const ECDevMatrix* projectionM,
                            const ECDevMatrix* screenM ) {
    
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int offset = x + y * blockDim.x * gridDim.x;
    
    // position ----------------------
    destVertexes[offset].position = (screenM * (projectionM * (viewM * (worldM * (scale * srcVertexes[offset].position)))));
    
    // normal ------------------------
    const float4 scale_1 = make_float4(1.0f,1.0f,1.0f,1.0f);
    destVertexes[offset].normal = normalize3D(getRotateMatrix(viewM,&scale_1) * (getRotateMatrix(worldM,&scale_1) * srcVertexes[offset].normal));
    
    // uv ----------------------------
    // x:u y:v
    destVertexes[offset].uv = srcVertexes[offset].uv;
    // z,w: to use perspective correction
#if (PERSPECTIVE_CORRECTION == true)
    destVertexes[offset].uv.z = 1.0f;
    destVertexes[offset].uv.w = multMatVecNoDivW(screenM, multMatVecNoDivW(projectionM, multMatVecNoDivW(viewM, multMatVecNoDivW(worldM, scale * srcVertexes[offset].position)))).w;
#endif
    
    // color -------------------------
    destVertexes[offset].color = srcVertexes[offset].color;
    
}



static bool triangleCounterClockwise(const float4 &v1,
                                const float4 &v2,
                                const float4 &v3) {
    return (v3.x * (v1.y - v2.y) + v1.x * (v2.y - v3.y) + v2.x * (v3.y - v1.y) >= 0.0f)?true:false;
}
                         
static void setDevMatrix( ECSmtDevPtr<ECDevMatrix> devMatrix, const ECMatrix& hostMatrix ) {
    ECDevMatrix *host = (ECDevMatrix*)malloc( sizeof(ECDevMatrix) );
    *host = hostMatrix;
    devMatrix.copyHostToDevice( host );
    free(host);
}

extern "C" {
    void gpuBindTexture( const texture<float>& texture, const float* texElmArray, const int size );
    void gpuUnbindTexture( const texture<float>& texture );
}
extern "C" void gpuRenderModel( ECModel* model, ECSmtPtr<ECRenderState> rState, ECPixel* pixels, ECSmtDevPtr<float4> zBuffer, ECSmtDevPtr<ECDevLight> light ) {
    
    setDevMatrix( model->devWorldMatrix, model->getWorldMatrix() );
    setDevMatrix( model->devViewMatrix, rState->getViewMatrix() );
    setDevMatrix( model->devProjectionMatrix, rState->getProjectionMatrix() );
    setDevMatrix( model->devScreenMatrix, rState->getScreenMatrix() );
    
    std::map<std::string,ECSmtDevPtr<ECDevMaterial> >::iterator mtlIt = model->materials.begin();
    while( mtlIt != model->materials.end() ) {
        std::string mtlName = (*mtlIt).first;
        ECSmtDevPtr<ECDevMaterial> material = (model->materials)[mtlName];
        ECSmtDevPtr<ECDevVertex> srcVertexes = (model->srcVertexes)[mtlName];
        ECSmtDevPtr<ECDevVertex> destVertexes = (model->destVertexes)[mtlName];
        int num_of_vertexes = (model->srcVertexes)[mtlName].getSize() / sizeof(ECDevVertex);
        
        // > ----------------------------------------------------------------------------------------------
        dim3 vertexShade_block(16, 1);
        const int gridX = num_of_vertexes/16 + num_of_vertexes%16;
        const dim3 vertexShade_grid(gridX, 1);
        if( !(model->hasAnim()) ) {
            vertexShade<<<vertexShade_grid,
                           vertexShade_block>>>(destVertexes.getPtr(),
                                                srcVertexes.getPtr(),
                                                model->scale.getPtr(),
                                                model->devWorldMatrix.getPtr(),
                                                model->devViewMatrix.getPtr(),
                                                model->devProjectionMatrix.getPtr(),
                                                model->devScreenMatrix.getPtr() );
        }
        else {
            /*
             std::map<std::string,ECSmtDevPtr<ECDevVertex> > boneAnimedVertexes;
             std::map<std::string,ECSmtDevPtr<DXDevWeight> > skinWeights;
             std::map<std::string,ECSmtDevPtr<ECDevMatrix> > offsetMatrixes;
             std::map<std::string,ECSmtDevPtr<ECDevMatrix> > currentBoneMatrixes;
             */
            { // ECDirectXModel
                ECSmtDevPtr<ECDevVertex> boneAnimedVertexes = ((ECDirectXModel*)model)->boneAnimedVertexes[mtlName];
                boneAnimedVertexes.copyDeviceToDevice( model->srcVertexes[mtlName].getPtr() );
                
                std::map<std::string,ECSmtDevPtr<DXDevWeight> >::iterator it = ((ECDirectXModel*)model)->skinWeights.begin();
                while ( it != ((ECDirectXModel*)model)->skinWeights.end() ) {
                    std::string boneName = (*it).first;
                    ECSmtDevPtr<DXDevWeight> devWeights = (*it).second;
                    
                    /*
                    const int num_of_weight_indexes = devWeights.getSize() / sizeof(DXDevWeight);
                    dim3 boneAnimShade_block(16, 1);
                    const int boneAnimShade_gridX = num_of_weight_indexes/16 + num_of_weight_indexes%16;
                    dim3 boneAnimShade_grid(boneAnimShade_gridX, 1);
                    resetOffsetShade<<<vertexShade_grid,
                                       vertexShade_block>>>(boneAnimedVertexes.getPtr(),
                                                              devWeights.getPtr(),
                                                              ((ECDirectXModel*)model)->offsetMatrixes[boneName].getPtr());
                    ECSmtPtr<DXDevWeight> hostWeights = (DXDevWeight*)malloc( devWeights.getSize() );
                    hostWeights.copyDeviceToHost( devWeights );
                    for( int i=0; i<hostWeights.getSize()/sizeof(DXDevWeight); ++i ) {
                        hostWeights[i]
                     }
                     */
                    
                    ++it;
                }
                it = ((ECDirectXModel*)model)->skinWeights.begin();
                while ( it != ((ECDirectXModel*)model)->skinWeights.end() ) {
                    std::string boneName = (*it).first;
                    ECSmtDevPtr<DXDevWeight> devWeights = (*it).second;
                    
                    const int num_of_weight_indexes = devWeights.getSize() / sizeof(DXDevWeight);
                    dim3 boneAnimShade_block(16, 1);
                    const int boneAnimShade_gridX = num_of_weight_indexes/16 + num_of_weight_indexes%16;
                    dim3 boneAnimShade_grid(boneAnimShade_gridX, 1);
//                    boneAnimShade<<<boneAnimShade_grid,
//                                    boneAnimShade_block>>>(boneAnimedVertexes.getPtr(),
//                                                           devWeights.getPtr(),
//                                                           ((ECDirectXModel*)model)->currentBoneMatrixes[boneName].getPtr() );
                    
                    ++it;
                }
                
                vertexShade<<<vertexShade_grid,
                              vertexShade_block>>>(destVertexes.getPtr(),
                                                   boneAnimedVertexes.getPtr(),
                                                   model->scale.getPtr(),
                                                   model->devWorldMatrix.getPtr(),
                                                   model->devViewMatrix.getPtr(),
                                                   model->devProjectionMatrix.getPtr(),
                                                   model->devScreenMatrix.getPtr() );
            }
            { // Other type model
                
            }
        }
        // < ----------------------------------------------------------------------------------------------
        
        
        // > -----------------------------------------------------------
        bool useColorTexture = false;
        bool useNormalTexture = false;
        bool useSpecularTexture = false;
        if( model->textures[mtlName]->hasColorTexture ) {
            useColorTexture = true;
        }
        if( model->textures[mtlName]->hasNormalTexture ) {
            useNormalTexture = true;
        }
        if( model->textures[mtlName]->hasSpecularTexture ) {
            useSpecularTexture = true;
        }
        
        if( useColorTexture ) {
            const int color_num_of_texture_pixel = model->textures[mtlName]->colorTextureWidth * 4 * model->textures[mtlName]->colorTextureHeight;
            gpuBindTexture(colorTexture,
                           model->textures[mtlName]->colorTexture.getPtr(),
                           sizeof(float) * color_num_of_texture_pixel);
        }
        if( useNormalTexture ) {
            const int normal_num_of_texture_pixel = model->textures[mtlName]->normalTextureWidth * 4 * model->textures[mtlName]->normalTextureHeight;
            gpuBindTexture(normalTexture,
                           model->textures[mtlName]->normalTexture.getPtr(),
                           sizeof(float) * normal_num_of_texture_pixel);
        }
        if( useSpecularTexture ) {
            const int specular_num_of_texture_pixel = model->textures[mtlName]->specularTextureWidth * 4 * model->textures[mtlName]->specularTextureHeight;
            gpuBindTexture(specularTexture,
                           model->textures[mtlName]->specularTexture.getPtr(),
                           sizeof(float) * specular_num_of_texture_pixel);
        }
        
        // > ---------------------------
        static const bool culling = true;
        if( !culling ) {
            dim3 screen_grid(rState->getScreenWidth()/16, rState->getScreenHeight()/4);
            dim3 screen_block(16, 16);
            pixelShade_NormalMap<<<screen_grid,screen_block>>>( pixels,
                                                               destVertexes.getPtr(),
                                                               destVertexes.getSize()/sizeof(ECDevVertex),
                                                               model->devWorldMatrix.getPtr(),
                                                               model->devViewMatrix.getPtr(),
                                                               material.getPtr(),
                                                               zBuffer.getPtr(),
                                                               light.getPtr() );
        }
        else {  // culling
            ECSmtPtr<ECDevVertex> hostVertexes = (ECDevVertex*)malloc( destVertexes.getSize() );
            destVertexes.copyDeviceToHost( hostVertexes.getPtr() );
            for( int index=0; index<destVertexes.getSize()/sizeof(ECDevVertex); index+=3 ) {
                if(
                   triangleCounterClockwise( hostVertexes[index].position,
                                            hostVertexes[index+1].position,
                                            hostVertexes[index+2].position )
                   
                   &&
                   
                   !((int)hostVertexes[index].position.x == (int)hostVertexes[index+1].position.x &&
                     (int)hostVertexes[index+1].position.x == (int)hostVertexes[index+2].position.x &&
                     (int)hostVertexes[index].position.y == (int)hostVertexes[index+1].position.y &&
                     (int)hostVertexes[index+1].position.y == (int)hostVertexes[index+2].position.y
                     )
                   
                   &&
                   
                   !( !(0 <= (int)hostVertexes[index].position.x && (int)hostVertexes[index].position.x <= rState->getScreenWidth() && 0 <= (int)hostVertexes[index].position.y && (int)hostVertexes[index].position.y <= rState->getScreenHeight() ) &&
                     !(0 <= (int)hostVertexes[index+1].position.x && (int)hostVertexes[index+1].position.x <= rState->getScreenWidth() && 0 <= (int)hostVertexes[index+1].position.y && (int)hostVertexes[index+1].position.y <= rState->getScreenHeight() ) &&
                     !(0 <= (int)hostVertexes[index+2].position.x && (int)hostVertexes[index+2].position.x <= rState->getScreenWidth() && 0 <= (int)hostVertexes[index+2].position.y && (int)hostVertexes[index+2].position.y <= rState->getScreenHeight() )
                     )
                   
                   ) {
                    float minX = min3( hostVertexes[index].position.x, hostVertexes[index+1].position.x, hostVertexes[index+2].position.x );
                    float minY = min3( hostVertexes[index].position.y, hostVertexes[index+1].position.y, hostVertexes[index+2].position.y );
                    float maxX = max3( hostVertexes[index].position.x, hostVertexes[index+1].position.x, hostVertexes[index+2].position.x );
                    float maxY = max3( hostVertexes[index].position.y, hostVertexes[index+1].position.y, hostVertexes[index+2].position.y );
                    int vertexWidth = maxX - minX + 4;
                    int vertexHeight = maxY - minY + 4;
                    int origin_x = minX - 2;
                    int origin_y = minY - 2;
                    int screen_grid_x = vertexWidth/16;
                    int screen_grid_y = vertexHeight/4;
                    if( vertexWidth%16 > 0 ) { screen_grid_x += 16; }
                    if( screen_grid_x == 0 ) { screen_grid_x = 1; }
                    if( screen_grid_y == 0 ) { screen_grid_y = 1; }
                    dim3 screen_grid( screen_grid_x, screen_grid_y );
                    dim3 screen_block(16, 16);
                    if( useColorTexture && !useNormalTexture && !useSpecularTexture ) {
                        pixelShade_perVertex<<<screen_grid,screen_block>>>( pixels,
                                                                           destVertexes.getPtr(),
                                                                           index,
                                                                           origin_x, origin_y,
                                                                           rState->getScreenWidth(),
                                                                           model->devWorldMatrix.getPtr(),
                                                                           model->devViewMatrix.getPtr(),
                                                                           material.getPtr(),
                                                                           zBuffer.getPtr(),
                                                                           light.getPtr() );
                        
                    }
                    else if( useColorTexture && useNormalTexture && !useSpecularTexture ) {
                        pixelShade_NormalMap_perVertex<<<screen_grid,screen_block>>>( pixels,
                                                                                     destVertexes.getPtr(),
                                                                                     index,
                                                                                     origin_x, origin_y,
                                                                                     rState->getScreenWidth(),
                                                                                     model->devWorldMatrix.getPtr(),
                                                                                     model->devViewMatrix.getPtr(),
                                                                                     material.getPtr(),
                                                                                     zBuffer.getPtr(),
                                                                                     light.getPtr() );
                    }
                }
            }
        }
        // < ---------------------------
        
        if( useSpecularTexture ) { gpuUnbindTexture( specularTexture ); }
        if( useNormalTexture ) { gpuUnbindTexture( normalTexture ); }
        if( useColorTexture ) { gpuUnbindTexture( colorTexture ); }
        
        ++mtlIt;
    }
    // < -----------------------------------------------------------
    
}


