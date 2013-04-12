#ifndef EC_MODEL_HPP
#define EC_MODEL_HPP

#include <vector>
#include <string>
#include <map>

#include "../smart_pointer/ECSmtPtr.hpp"
#include "../renderer/ECPixelBuffer.hpp"
#include "../math/ECMatrix.hpp"
#include "../renderer/ECRenderState.hpp"

#include <vector_types.h>




struct ECDevVertex {
    float4 position;
    float4 normal;
    float4 uv; // x:u y:v z,w: to use perspective correction
    float4 color;
};
struct ECDevBone {
    // boneName
    // boneMatrix
};
struct ECDevLight {
    float4 position;
    float4 direction;
    float4 ambient;
    float4 diffuse;
    float4 specular;
};
struct VertexIndex {
    int index_v;
    int index_vt;
    int index_vn;
};
struct ECDevMaterial {
    float4 ambient;
    float4 diffuse;
    float4 specular;
    float4 shininess;
    float4 emission;
    
    float4 colorTextureSize; // x:width y:height
    float4 normalTextureSize;
    float4 specularTextureSize;
};
struct ECModelTextures {
    bool hasColorTexture;
    int colorTextureWidth;
    int colorTextureHeight;
    ECSmtDevPtr<float> colorTexture;
    bool hasNormalTexture;
    int normalTextureWidth;
    int normalTextureHeight;
    ECSmtDevPtr<float> normalTexture;
    bool hasSpecularTexture;
    int specularTextureWidth;
    int specularTextureHeight;
    ECSmtDevPtr<float> specularTexture;
    
    ECModelTextures() {
        hasColorTexture = false;
        colorTextureWidth = 0;
        colorTextureHeight = 0;
        hasNormalTexture = false;
        normalTextureWidth = 0;
        normalTextureHeight = 0;
        hasSpecularTexture = false;
        specularTextureWidth = 0;
        specularTextureHeight = 0;
    }
};


class ECDevMatrix;
class ECModel {
public:
    ECMatrix worldMatrix;
    ECSmtDevPtr<float4> scale;
    
    ECSmtDevPtr<ECDevMatrix> devWorldMatrix;
    ECSmtDevPtr<ECDevMatrix> devViewMatrix;
    ECSmtDevPtr<ECDevMatrix> devScreenMatrix;
    ECSmtDevPtr<ECDevMatrix> devProjectionMatrix;
    
    std::map<std::string,ECSmtDevPtr<ECDevVertex> > srcVertexes;
    std::map<std::string,ECSmtDevPtr<ECDevVertex> > destVertexes;
    std::map<std::string,ECSmtDevPtr<ECDevMaterial> > materials;
    std::map<std::string,ECSmtPtr<ECModelTextures> > textures;
    
    std::vector<std::string> getMaterialNames();
    
    virtual ~ECModel() {}
    
    void setWorldMatrix( const ECMatrix& m );
    ECMatrix getWorldMatrix() { return worldMatrix; };
    void setScale( const float4& s );
    
    virtual void render( ECPixel* pixels, ECSmtPtr<ECRenderState> rState, ECSmtDevPtr<float4> zBuffer, ECSmtDevPtr<ECDevLight> light ) = 0;
    
    bool hasAnim();
    
protected:
    ECSmtDevPtr<float> loadTexture( std::string filePath, int* ret_width, int* ret_height );
    virtual void load( std::string filePath ) = 0;
    
    bool hasAnimation;
    
    // > block --------
    ECModel();
private:
    ECModel( const ECModel& obj );
    ECModel& operator=( const ECModel& obj );
    // < block --------
    
};


class ECWaveFrontObject : public ECModel {
public:
    static ECSmtPtr<ECWaveFrontObject> Create( std::string filePath ) {
        ECSmtPtr<ECWaveFrontObject> ret = new ECWaveFrontObject();
        ret->load( filePath );
        return ret;
    }
    
    void render( ECPixel* pixels, ECSmtPtr<ECRenderState> rState, ECSmtDevPtr<float4> zBuffer, ECSmtDevPtr<ECDevLight> light );
    
private:
    void load( std::string filePath );
    std::map<std::string,ECDevMaterial> loadMtlFile( std::string filePath );
    
    // block
    ECWaveFrontObject() {}
    ECWaveFrontObject( const ECWaveFrontObject& obj );
    ECWaveFrontObject& operator=( const ECWaveFrontObject& obj );
    
};


struct DXFrameTree {
    std::string frameName;
    ECMatrix transformMatrix;
    std::vector<DXFrameTree> children;
};
struct DXDevWeight {
    int index;
    float weight;
};
class ECDirectXModel : public ECModel {
public:
    static ECSmtPtr<ECDirectXModel> Create( std::string filePath ) {
        ECSmtPtr<ECDirectXModel> ret = new ECDirectXModel();
        ret->load( filePath );
        ret->hasAnimation = true;
        return ret;
    }
    std::vector<std::string> getAnimationNames();
    void animate( const float& f0_f1 );
    
    std::map<std::string,ECSmtDevPtr<ECDevVertex> > boneAnimedVertexes;
    std::map<std::string,ECSmtDevPtr<DXDevWeight> > skinWeights;
    std::map<std::string,ECSmtDevPtr<ECDevMatrix> > offsetMatrixes;
    std::map<std::string,ECSmtDevPtr<ECDevMatrix> > currentBoneMatrixes;
    
    
    void render( ECPixel* pixels, ECSmtPtr<ECRenderState> rState, ECSmtDevPtr<float4> zBuffer, ECSmtDevPtr<ECDevLight> light );
private:
    void load( std::string filePath );
    
    std::vector<std::string> animationNames;
    std::map<std::string, std::vector<std::pair<float,ECMatrix> > > animationKeys;
    std::vector<DXFrameTree> frameTree;
    float maxTime;
    
    // block
    ECDirectXModel() {}
    ECDirectXModel( const ECDirectXModel& obj ) {}
    ECDirectXModel& operator=( const ECDirectXModel& obj );
    
};



#endif
