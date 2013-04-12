//
//  ECModel.cpp
//  EClay001
//
//  Created by Fumiyasu Takaura on 3/5/13.
//
//


#include "ECModel.hpp"
#include "../etc/stbi.hpp" // thank you very very very much!!!!!!!!!

#include "../etc/ECUtil.hpp"
#include "../math/ECVector.hpp"

#include <fstream>

#include <cutil_math.h>
#include <cutil_inline.h>

////////////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <sstream>
////////////////////////////////////////////////////////////////////////////////////

using namespace std;


extern "C" void devMatrixInit( ECModel* model );
extern "C" void devSetMatrix( ECSmtDevPtr<ECDevMatrix>& dest, ECMatrix& src );
extern "C" void devCopyMatrix( ECSmtDevPtr<ECDevMatrix>& dest, ECMatrix& src );
extern "C" void gpuRenderModel( ECModel* model, ECSmtPtr<ECRenderState> rState, ECPixel* pixels, ECSmtDevPtr<float4> zBuffer, ECSmtDevPtr<ECDevLight> light );


// ECModel **************************************************************************************************
ECModel::ECModel() {
    
    devMatrixInit( this );
    
    setWorldMatrix( ECMatrix() );
    
    scale = ECSmtDevPtr<float4>( sizeof(float4) );
    setScale( make_float4(1.0f,1.0f,1.0f,1.0f) );
    
    
    hasAnimation = false;
}

bool ECModel::hasAnim() {
    return hasAnimation;
}

vector<string> ECModel::getMaterialNames() {
    vector<string> ret;
    map<string,ECSmtDevPtr<ECDevMaterial> >::iterator it = materials.begin();
    while ( it != materials.end() ) {
        ret.push_back( (*it).first );
        ++it;
    }
    return ret;
}

void ECModel::setWorldMatrix( const ECMatrix& m ) {
    worldMatrix = m;
}

void ECModel::setScale( const float4& s ) {
    float4* sca = (float4*)malloc( sizeof(float4) );
    *sca = s;
    scale.copyHostToDevice( sca );
    free( sca );
}


ECSmtDevPtr<float> ECModel::loadTexture( string filePath, int* ret_width, int* ret_height ) {
    unsigned char* img;
    int imgW;
    int imgH;
    int bpp;
    img = stbi_load(filePath.c_str(), &imgW, &imgH, &bpp, 0);
    const int num_of_texture_pixel = imgW * 4 * imgH;
    float *hostTexture = (float*)malloc( sizeof(float) * num_of_texture_pixel );
    int pixelCount = 0;
    for( int i=0; i<num_of_texture_pixel; i+=4 ) {
        hostTexture[i  ] = img[pixelCount]/(float)255.0;
        hostTexture[i+1] = img[++pixelCount]/(float)255.0;
        hostTexture[i+2] = img[++pixelCount]/(float)255.0;
        hostTexture[i+3] = (bpp==4)?(unsigned int)img[++pixelCount]/(float)255.0:1.0f;
        ++pixelCount;
    }
    ECSmtDevPtr<float> ret = ECSmtDevPtr<float>(hostTexture, sizeof(float) * num_of_texture_pixel);
    *ret_width = imgW;
    *ret_height = imgH;
    free( hostTexture );
    stbi_image_free( img );
    return ret;
}


// ECWaveFrontObject ******************************************************************************************
void ECWaveFrontObject::render( ECPixel* pixels, ECSmtPtr<ECRenderState> rState, ECSmtDevPtr<float4> zBuffer, ECSmtDevPtr<ECDevLight> light ) {
    gpuRenderModel( this, rState, pixels, zBuffer, light );
}

void ECWaveFrontObject::load( string filePath ) {
    
    ifstream ifs( filePath.c_str() );
    
    map<string,ECDevMaterial> temp_materials;
    vector<float4> temp_v;
    vector<float4> temp_vn;
    vector<float4> temp_vt;
    map<string,vector<VertexIndex> > temp_f;
    string currentMtlName = "";
    
    string line;
    while( ifs && getline(ifs, line) )
    {
        if( line.substr(0,1) == "#" || line.size() == 0 ) { continue; }
        line = ReduceSpaceChar( line );
        
        vector<string> lineData = Separate( line, " " );
        
        if( lineData.size() ) {
            string tag = lineData.front();
            
            if( tag == "mtllib" ) {
                string dir = filePath;
                dir = dir.substr(0,dir.find_last_of("/"));
                string mtlFilePath = dir + "/" + lineData[1];
                temp_materials = loadMtlFile( mtlFilePath );
            }
            else if( tag == "usemtl" ) {
                currentMtlName = lineData[1];
            }
            else if( tag == "v" ) {
                const float x = atof( lineData[1].c_str() );
                const float y = atof( lineData[2].c_str() );
                const float z = atof( lineData[3].c_str() );
                temp_v.push_back( make_float4( x,y,z,1.0f ) );
            }
            else if( tag == "vn" ) {
                float x = atof( lineData[1].c_str() );
                float y = atof( lineData[2].c_str() );
                float z = atof( lineData[3].c_str() );
                float recipical_length = 1.0f;
                if( sqrt(x*x+y*y+z*z) > 0.0f ) { 1.0f/sqrt(x*x+y*y+z*z); }
                x *= recipical_length;
                y *= recipical_length;
                z *= recipical_length;
                temp_vn.push_back( make_float4( x,y,z,1.0f ) );
            }
            else if( tag == "vt" ) {
                const float u = atof( lineData[1].c_str() );
                const float v = atof( lineData[2].c_str() );
                temp_vt.push_back( make_float4( u,v,1.0f,1.0f ) );
            }
            else if( tag == "f" ) {
                vector<string> strA = Separate( lineData[1], "/" );
                vector<string> strB = Separate( lineData[2], "/" );
                vector<string> strC = Separate( lineData[3], "/" );
                VertexIndex indexA;
                indexA.index_v = atoi( strA[0].c_str() ) - 1;
                indexA.index_vt = atoi( strA[1].c_str() ) - 1;
                indexA.index_vn = atoi( strA[2].c_str() ) - 1;
                temp_f[currentMtlName].push_back( indexA );
                VertexIndex indexB;
                indexB.index_v = atoi( strB[0].c_str() ) - 1;
                indexB.index_vt = atoi( strB[1].c_str() ) - 1;
                indexB.index_vn = atoi( strB[2].c_str() ) - 1;
                temp_f[currentMtlName].push_back( indexB );
                VertexIndex indexC;
                indexC.index_v = atoi( strC[0].c_str() ) - 1;
                indexC.index_vt = atoi( strC[1].c_str() ) - 1;
                indexC.index_vn = atoi( strC[2].c_str() ) - 1;
                temp_f[currentMtlName].push_back( indexC );
            }
        }
    }
    
    
    map<string,ECDevMaterial>::iterator it = temp_materials.begin();
    while ( it != temp_materials.end() ) {
        string mtlName = (*it).first;
        ECDevMaterial mtl = (*it).second;
        
        // Vertex
        int num_of_vertexes = temp_f[mtlName].size();
        ECDevVertex *hostVertexes = (ECDevVertex*)malloc( sizeof(ECDevVertex)*num_of_vertexes );
        for( int i=0; i<num_of_vertexes; ++i ) {
            hostVertexes[i].position = temp_v[(temp_f[mtlName])[i].index_v];
            hostVertexes[i].uv = temp_vt[(temp_f[mtlName])[i].index_vt];
            hostVertexes[i].normal = temp_vn[(temp_f[mtlName])[i].index_vn];
            /////////////////
            float r = 1.0f;// if(i%3==0) { x=1.0f; }
            float g = 1.0f;// if(i%3==1) { y=1.0f; }
            float b = 1.0f;// if(i%3==2) { z=1.0f; }
            float a = 1.0f;//////////////////////////////////////////////////
            ////////////////
            hostVertexes[i].color = make_float4( r,g,b,a );
        }
        srcVertexes[mtlName]  = ECSmtDevPtr<ECDevVertex>( hostVertexes, sizeof(ECDevVertex)*num_of_vertexes );
        destVertexes[mtlName] = ECSmtDevPtr<ECDevVertex>( hostVertexes, sizeof(ECDevVertex)*num_of_vertexes );
        free( hostVertexes );
        
        // Material
        ECDevMaterial *hostMaterial = (ECDevMaterial*)malloc( sizeof(ECDevMaterial) );
        *hostMaterial = mtl;
        materials[mtlName] = ECSmtDevPtr<ECDevMaterial>( hostMaterial, sizeof(ECDevMaterial) );
        free( hostMaterial );
        
        ++it;
    }
}

map<string,ECDevMaterial> ECWaveFrontObject::loadMtlFile( string filePath ) {
    
    map<string,ECDevMaterial> ret;
    
    ifstream ifs( filePath.c_str() );
    
    string currentMtlName = "";
    ECDevMaterial currentMaterial;
    currentMaterial.ambient = make_float4( 0.0f, 0.0f, 0.0f, 1.0f );
    currentMaterial.diffuse = make_float4( 0.0f, 0.0f, 0.0f, 1.0f );
    currentMaterial.specular = make_float4( 0.0f, 0.0f, 0.0f, 1.0f );
    currentMaterial.shininess = make_float4( 0.0f, 0.0f, 0.0f, 1.0f );
    currentMaterial.emission = make_float4( 0.0f, 0.0f, 0.0f, 1.0f );
    currentMaterial.colorTextureSize = make_float4( 0.0f, 0.0f, 0.0f, 1.0f );
    currentMaterial.normalTextureSize = make_float4( 0.0f, 0.0f, 0.0f, 1.0f );
    
    string line;
    while( ifs && getline(ifs, line) )
    {
        if( line.substr(0,1) == "#" || line.size() == 0 ) { continue; }
        line = ReduceSpaceChar( line );
        
        vector<string> lineData = Separate( line, " " );
        
        if( lineData.size() ) {
            string tag = lineData.front();
            
            if( tag == "newmtl" ) {
                if( currentMtlName.size() ) {
                    ret[currentMtlName] = currentMaterial;
                }
                currentMtlName = lineData[1];
                textures[currentMtlName] = new ECModelTextures();
            }
            else if( tag == "Ka" ) {
                currentMaterial.ambient = make_float4( atof(lineData[1].c_str()), atof(lineData[2].c_str()), atof(lineData[3].c_str()), 1.0f );
            }
            else if( tag == "Kd" ) {
                currentMaterial.diffuse = make_float4( atof(lineData[1].c_str()), atof(lineData[2].c_str()), atof(lineData[3].c_str()), 1.0f );
            }
            else if( tag == "Ks" ) {
                currentMaterial.specular = make_float4( atof(lineData[1].c_str()), atof(lineData[2].c_str()), atof(lineData[3].c_str()), 1.0f );
            }
            else if( tag == "Ns" ) {
                currentMaterial.shininess = make_float4( atof(lineData[1].c_str()), atof(lineData[1].c_str()), atof(lineData[1].c_str()), atof(lineData[1].c_str()) );
            }
            else if( tag == "Ni" ) {
                
            }
            else if( tag == "d" ) {
                
            }
            else if( tag == "illum" ) {
                /*
                 * 0. Color on and Ambient off
                 * 1. Color on and Ambient on
                 * 2. Highlight on
                 * 3. Reflection on and Ray trace on
                 * 4. Transparency: Glass on, Reflection: Ray trace on
                 * 5. Reflection: Fresnel on and Ray trace on
                 * 6. Transparency: Refraction on, Reflection: Fresnel off and Ray trace on
                 * 7. Transparency: Refraction on, Reflection: Fresnel on and Ray trace on
                 * 8. Reflection on and Ray trace off
                 * 9. Transparency: Glass on, Reflection: Ray trace off
                 * 10. Casts shadows onto invisible surfaces
                 */
                if( atoi(lineData[1].c_str()) == 0 ) {
                    currentMaterial.ambient = make_float4( 0.0f, 0.0f, 0.0f, 1.0f );
                }
                else if( atoi(lineData[1].c_str()) == 1 ) {
                }
                else if( atoi(lineData[1].c_str()) == 2 ) {
                }
            }
            else if( tag == "map_Ka" ) {
                
            }
            else if( tag == "map_Kd" ) {
                string dir = filePath;
                dir = dir.substr(0,dir.find_last_of("/"));
                string imgFilePath = dir + "/" + lineData[1];
                
                int texW = 0;
                int texH = 0;
                textures[currentMtlName]->colorTexture = loadTexture( imgFilePath, &texW, &texH );
                textures[currentMtlName]->colorTextureWidth = texW;
                textures[currentMtlName]->colorTextureHeight = texH;
                textures[currentMtlName]->hasColorTexture = true;
                currentMaterial.colorTextureSize.x = texW;
                currentMaterial.colorTextureSize.y = texH;
            }
            else if( tag == "map_Ks" ) {
                
            }
            else if( tag == "map_Ns" ) {
                
            }
            else if( tag == "map_d" ) {
            }
            else if( tag == "map_bump" ) {
                string dir = filePath;
                dir = dir.substr(0,dir.find_last_of("/"));
                string imgFilePath = dir + "/" + lineData[1];
                
                int texW = 0;
                int texH = 0;
                textures[currentMtlName]->normalTexture = loadTexture( imgFilePath, &texW, &texH );
                textures[currentMtlName]->normalTextureWidth = texW;
                textures[currentMtlName]->normalTextureHeight = texH;
                textures[currentMtlName]->hasNormalTexture = true;
                currentMaterial.normalTextureSize.x = texW;
                currentMaterial.normalTextureSize.y = texH;
            }
        }
    }
    
    if( currentMtlName.size() ) {
        ret[currentMtlName] = currentMaterial;
    }
    
    return ret;
}




// ECDirectXModel ***********************************************************************************************
void ECDirectXModel::render( ECPixel* pixels, ECSmtPtr<ECRenderState> rState, ECSmtDevPtr<float4> zBuffer, ECSmtDevPtr<ECDevLight> light ) {
    gpuRenderModel( this, rState, pixels, zBuffer, light );
}

vector<string> ECDirectXModel::getAnimationNames() {
    return animationNames;
}


static void ReadAndcutSpace( const string& filePath, string* dest ) {
    
    ifstream ifs( filePath.c_str() );
    string line;
    while( ifs && getline(ifs, line) ) {
        string tempLine = ReduceSpaceChar( line );
        string::size_type hash_index = tempLine.find('#');
        if( hash_index != string::npos ) {  // delete # comment
            tempLine = ReduceSpaceChar( tempLine.substr( 0, hash_index ) );
        }
        if( tempLine.size() > 0 ) {
            *dest += tempLine;
            *dest += '\n';
        }
    }
    
}


static const string DX_Template = "template";
static const string DX_AnimTicksPerSecond = "AnimTicksPerSecond";
static const string DX_Frame = "Frame";
static const string DX_FrameTransformMatrix = "FrameTransformMatrix";
static const string DX_Mesh = "Mesh";
static const string DX_MeshNormals = "MeshNormals";
static const string DX_MeshTextureCoords = "MeshTextureCoords";
static const string DX_MeshMaterialList = "MeshMaterialList";
static const string DX_Material = "Material";
static const string DX_TextureFilename = "TextureFilename";
static const string DX_XSkinMeshHeader = "XSkinMeshHeader";
static const string DX_SkinWeights = "SkinWeights";
static const string DX_AnimationSet = "AnimationSet";
static const string DX_Animation = "Animation";
static const string DX_AnimationKey = "AnimationKey";
static const string DX_VertexDuplicationIndices = "VertexDuplicationIndices";

struct TriangleIndexes {
    int index1, index2, index3;
    TriangleIndexes( int i1, int i2, int i3 ) {
        index1 = i1; index2 = i2; index3 = i3;
    }
};
struct DXElm {
    string type;
    string name;
    vector<ECSmtPtr<DXElm> > children;
    
    DXElm( const string& t ) {
        type = t;
    }
private:
    DXElm();
    DXElm( const DXElm& obj );
    DXElm& operator=( const DXElm& obj );
};
struct DXTemplate : public DXElm {
    
    DXTemplate( const string& str ) : DXElm( DX_Template ) {
        // nothing to do.
    }
};
struct DXAnimTicksPerSecond : public DXElm {
    int ticksPerSecond;
    
    DXAnimTicksPerSecond( const string& str ) : DXElm( DX_AnimTicksPerSecond ) {
        vector<string> temp = Separate( str, ";" );
        ticksPerSecond = atoi( temp[0].c_str() );
    }
};
struct DXFrame : public DXElm {
    
    DXFrame( const string& str ) : DXElm( DX_Frame ) {
        
    }
};
struct DXFrameTransformMatrix : public DXElm {
    ECMatrix transformMatrix;
    
    DXFrameTransformMatrix( const string& str ) : DXElm( DX_FrameTransformMatrix ) {
        vector<string> temp = Separate( str, "," );
        vector<string>::iterator it = temp.begin();
        
        transformMatrix = ECMatrix(atof(temp[0].c_str()), atof(temp[1].c_str()), atof(temp[2].c_str()), atof(temp[3].c_str()),
                                   atof(temp[4].c_str()), atof(temp[5].c_str()), atof(temp[6].c_str()), atof(temp[7].c_str()),
                                   atof(temp[8].c_str()), atof(temp[9].c_str()), atof(temp[10].c_str()), atof(temp[11].c_str()),
                                   atof(temp[12].c_str()), atof(temp[13].c_str()), atof(temp[14].c_str()), atof(Separate(temp[15],";;")[0].c_str()) ).transposition();
    }
};
struct DXMesh : public DXElm {    
    vector<ECVector3D> vertexes;
    vector<TriangleIndexes> indexes;
    
    DXMesh( const string& str ) : DXElm( DX_Mesh ) {
        vector<string> temp = Separate( str, "," );
        const int num_of_triangles = atoi( Separate(temp[0],";")[0].c_str() );
        const vector<string> sept = Separate(temp[0],";"); temp[0] = sept[1]+";"+sept[2]+";"+sept[3];
        for( int i=0; i<num_of_triangles; ++i ) {
            const vector<string> vStr = Separate( temp[i], ";" );
            vertexes.push_back( ECVector3D(atof( vStr[0].c_str() ),
                                           atof( vStr[1].c_str() ),
                                           atof( vStr[2].c_str() ) ) );
        }
        const string indexesStr = Separate( str, ";;" )[1];
        const int num_of_indexes = atoi( Separate( indexesStr, ";" )[0].c_str() );
        vector<string> indStr = Separate( indexesStr, ";," );
        const vector<string> sepi = Separate(indStr[0],";"); indStr[0] = sepi[1]+";"+sepi[2];
        for( int i=0; i<num_of_indexes; ++i ) {
            if( atoi(Separate( indStr[i], ";" )[0].c_str()) != 3 ) {
                cout << "ERROR: not implemented in: " << __PRETTY_FUNCTION__ << endl;
            }
            vector<string> inds  = Separate( Separate( indStr[i], ";" )[1], "," );
            TriangleIndexes ti = TriangleIndexes( atoi(inds[0].c_str()), atoi(inds[1].c_str()), atoi(inds[2].c_str()) );
            indexes.push_back( ti );
        }
    }
};
struct DXMeshNormals : public DXElm {
    vector<ECVector3D> normals;
    vector<TriangleIndexes> indexes;
    
    DXMeshNormals( const string& str ) : DXElm( DX_MeshNormals ) {
        vector<string> temp = Separate( str, "," );
        const int num_of_triangles = atoi( Separate(temp[0],";")[0].c_str() );
        const vector<string> sept = Separate(temp[0],";"); temp[0] = sept[1]+";"+sept[2]+";"+sept[3];
        for( int i=0; i<num_of_triangles; ++i ) {
            const vector<string> nStr = Separate( temp[i], ";" );
            normals.push_back( ECVector3D(atof( nStr[0].c_str() ),
                                          atof( nStr[1].c_str() ),
                                          atof( nStr[2].c_str() ) ) );
        }
        const string indexesStr = Separate( str, ";;" )[1];
        const int num_of_indexes = atoi( Separate( indexesStr, ";" )[0].c_str() );
        vector<string> indStr = Separate( indexesStr, ";," );
        const vector<string> sepi = Separate(indStr[0],";"); indStr[0] = sepi[1]+";"+sepi[2];
        for( int i=0; i<num_of_indexes; ++i ) {
            if( atoi(Separate( indStr[i], ";" )[0].c_str()) != 3 ) {
                cout << "ERROR: not implemented in: " << __PRETTY_FUNCTION__ << endl;
            }
            vector<string> inds  = Separate( Separate( indStr[i], ";" )[1], "," );
            TriangleIndexes ti = TriangleIndexes( atoi(inds[0].c_str()), atoi(inds[1].c_str()), atoi(inds[2].c_str()) );
            indexes.push_back( ti );
        }
    }
};
struct DXMeshTextureCoords : public DXElm {
    vector<ECVector2D> uvs;
    
    DXMeshTextureCoords( const string& str ) : DXElm( DX_MeshTextureCoords ) {
        vector<string> temp = Separate( str, "," );
        const int num_of_triangles = atoi( Separate(temp[0],";")[0].c_str() );
        const vector<string> sept = Separate(temp[0],";"); temp[0] = sept[1]+";"+sept[2];
        for( int i=0; i<num_of_triangles; ++i ) {
            const vector<string> uvStr = Separate( temp[i], ";" );
            uvs.push_back( ECVector2D(atof( uvStr[0].c_str() ),
                                      1.0f-atof( uvStr[1].c_str() ) ) );
        }
    }
};
struct DXMeshMaterialList : public DXElm {
    vector<int> materialIndexes;
//    vector<string> materialNames;
    
    DXMeshMaterialList( const string& str ) : DXElm( DX_MeshMaterialList ) {
        const vector<string> temp = Separate( str, ";" );
        const int num_of_materialNames = atoi( temp[0].c_str() );
        const int num_of_materialIndexes = atoi( temp[1].c_str() );
        const vector<string> indStr = Separate( temp[2], "," );
        for( int i=0; i<num_of_materialIndexes; ++i ) {
            materialIndexes.push_back( atoi( indStr[i].c_str() ) );
        }
//        const vector<string> temp_materialNames = Separate( str, "{" );
//        for( int i=1; i<=num_of_materialNames; ++i ) {
//            materialNames.push_back( Separate( temp_materialNames[i], "}" )[0] );
//        }
    }
};
struct DXMaterial : public DXElm {
  
    float4 color;
    float shininess;
    float4 specular;
    float4 emission;
    
    DXMaterial( const string& str ) : DXElm( DX_Material ) {
        const vector<string> temp = Separate( str, ";;" );
        const vector<string> temp_color = Separate( temp[0], ";" );
        const vector<string> temp_specular = Separate( temp[1], ";" );
        const vector<string> temp_emission = Separate( temp[2], ";" );
        color = make_float4(atof(temp_color[0].c_str()), atof(temp_color[1].c_str()), atof(temp_color[2].c_str()), atof(temp_color[3].c_str()) );
        shininess = atof( temp_specular[0].c_str() );
        specular = make_float4( atof(temp_specular[1].c_str()), atof(temp_specular[2].c_str()), atof(temp_specular[3].c_str()), 1.0f );
        emission = make_float4( atof(temp_emission[0].c_str()), atof(temp_emission[1].c_str()), atof(temp_emission[2].c_str()), 1.0f );
    }
};
struct DXTextureFilename : public DXElm {
    
    string fileName;
    
    DXTextureFilename( const string& str ) : DXElm( DX_TextureFilename ) {
        fileName = Separate( str, "\"" )[1];
    }
};
struct DXXSkinMeshHeader : public DXElm {
    
    DXXSkinMeshHeader( const string& str ) : DXElm( DX_XSkinMeshHeader ) {
        // nothing to do.
    }
};
struct DXSkinWeights : public DXElm {
    string boneName;
    vector<int> indexes;
    vector<float> weights;
    ECMatrix offsetMatrix;
    
    DXSkinWeights( const string& str ) : DXElm( DX_SkinWeights ) {
        const vector<string> temp = Separate( str, "\";" );
        boneName = Separate( temp[0], "\"" )[1];
        const int num_of_indexes = atoi( Separate( temp[1], ";" )[0].c_str() );
        const vector<string> indexesStrs = Separate( Separate( temp[1], ";" )[1], "," );
        for( int i=0; i<num_of_indexes; ++i ) {
            indexes.push_back( atoi( indexesStrs[i].c_str() ) );
        }
        const vector<string> weightsStrs = Separate( Separate( temp[1], ";" )[2], "," );
        for( int i=0; i<num_of_indexes; ++i ) {
            weights.push_back( atoi( weightsStrs[i].c_str() ) );
        }
        const vector<string> matrixElm = Separate( Separate( temp[1], ";" )[3], "," );
        ECMatrix tempMatrix;
        for( int i=0; i<matrixElm.size(); ++i ) {
            tempMatrix.m[i] = atof( matrixElm[i].c_str() );
        }
        offsetMatrix = tempMatrix.transposition();
    }
};
struct DXAnimationSet : public DXElm {

    DXAnimationSet( const string& str ) : DXElm( DX_AnimationSet ) {
        
    }
};
struct DXAnimation : public DXElm {
    string animationName;
    
    DXAnimation( const string& str ) : DXElm( DX_Animation ) {
        string::size_type AnimationKey_index = str.find("AnimationKey");
        string::size_type brace_index = str.find("{");
        int index = 1;
        if( AnimationKey_index != string::npos && brace_index != string::npos ) {
            if( AnimationKey_index < brace_index ) {
                index = 2;
            }
        }
        animationName = Separate( Separate( str, "{" )[index], "}" )[0];
    }
};
struct DXAnimationKey : public DXElm {
    int flag; // 0:rotate 1:scale 2:translate 3:matrix
    vector<pair<float,ECMatrix> > keys;
    
    DXAnimationKey( const string& str ) : DXElm( DX_AnimationKey ) {
        const vector<string> temp = Separate( str, ";" );
        flag = atoi( temp[0].c_str() );
        const int num_of_keys = atoi( temp[1].c_str() );
        string keysStr = ""; for(int i=2; i<temp.size(); ++i) { keysStr += temp[i]; keysStr += ";"; }
        const vector<string> tempTimeAndMatrix = Separate( keysStr.c_str(), ";;," );
        for( int i=0; i<num_of_keys; ++i ) {
            const vector<string> tempElm = Separate( tempTimeAndMatrix[i], ";" );
            const float time = atof( tempElm[0].c_str() );
            const int num_of_elms = atoi( tempElm[1].c_str() );
            const vector<string> matStrs = Separate(tempElm[2], ",");
            ECMatrix tempMat;
            for( int i=0; i<matStrs.size(); ++i ) {
                tempMat.m[i] = atof( matStrs[i].c_str() );
            }
            keys.push_back( make_pair( time, tempMat.transposition() ) );
        }
    }
};
struct DXVertexDuplicationIndices : public DXElm {
    
    DXVertexDuplicationIndices( const string& str ) : DXElm( DX_VertexDuplicationIndices ) {
        // nothing to do.
    }
};

static ECSmtPtr<DXElm> MakeDXElm( const string& type, const string& name, const string& content ) {
    ECSmtPtr<DXElm> ret;
    
    if( type == DX_Template ) {
        ret = new DXTemplate( content );
    }
    else if( type == DX_AnimTicksPerSecond ) {
        ret = new DXAnimTicksPerSecond( content );
    }
    else if( type == DX_Frame ) {
        ret = new DXFrame( content );
    }
    else if( type == DX_FrameTransformMatrix ) {
        ret = new DXFrameTransformMatrix( content );
    }
    else if( type == DX_Mesh ) {
        ret = new DXMesh( content );
    }
    else if( type == DX_MeshNormals ) {
        ret = new DXMeshNormals( content );
    }
    else if( type == DX_MeshTextureCoords ) {
        ret = new DXMeshTextureCoords( content );
    }
    else if( type == DX_MeshMaterialList ) {
        ret = new DXMeshMaterialList( content );
    }
    else if( type == DX_Material ) {
        ret = new DXMaterial( content );
    }
    else if( type == DX_TextureFilename ) {
        ret = new DXTextureFilename( content );
    }
    else if( type == DX_XSkinMeshHeader ) {
        ret = new DXXSkinMeshHeader( content );
    }
    else if( type == DX_SkinWeights ) {
        ret = new DXSkinWeights( content );
    }
    else if( type == DX_AnimationSet ) {
        ret = new DXAnimationSet( content );
    }
    else if( type == DX_Animation ) {
        ret = new DXAnimation( content );
    }
    else if( type == DX_AnimationKey ) {
        ret = new DXAnimationKey( content );
    }
    else if( type == DX_VertexDuplicationIndices ) {
        ret = new DXVertexDuplicationIndices( content );
    }
    else {
        cout << "Not exist function parses \"" << type << "\"" << endl;
    }
    
    if( !ret.isNull() ) {
        ret->name = name;
    }
    
    return ret;
}

static vector<ECSmtPtr<DXElm> > ParseToDXElms( const string& str ) {
    vector<ECSmtPtr<DXElm> > ret;
    
    const int size = str.size();
    string word = "";
    int count = 0;
    while ( count < size ) {
        const char c = str[count];
        if( !( c==' ' || c=='\n' || c=='{' || c== '}' ) ) { word += c; }
        else {
            if(
               word == DX_Template ||
               word == DX_AnimTicksPerSecond ||
               word == DX_Frame ||
               word == DX_FrameTransformMatrix ||
               word == DX_Mesh ||
               word == DX_MeshNormals ||
               word == DX_MeshTextureCoords ||
               word == DX_MeshMaterialList ||
               word == DX_Material ||
               word == DX_TextureFilename ||
               word == DX_XSkinMeshHeader ||
               word == DX_SkinWeights ||
               word == DX_AnimationSet ||
               word == DX_Animation ||
               word == DX_AnimationKey ||
               word == DX_VertexDuplicationIndices
               ) {
                ++count;
                string name = "";
                while ( str[count] != '{' ) {
                    if( str[count] != '\n' ) { name += str[count]; }
                    ++count; if( str.size() < count ) { break; }
                } ++count;
                name = ReduceSpaceChar( name );
                int braceCount = 1;
                string content = "";
                bool isPrevNewLineChar = false;
                while ( braceCount > 0 ) {
                    char cc = str[count];
                    if( cc == '{' ) { ++braceCount; }
                    else if( cc == '}' ) {
                        --braceCount;
                        if( braceCount <= 0 ) { break; }
                    }
                    content += cc;
                    ++count; if( str.size() < count ) { break; }
                }
                
                ECSmtPtr<DXElm> currentElm = MakeDXElm( word, name, DeleteNewLineChar(content) );
                currentElm->children = ParseToDXElms( content );
                ret.push_back( currentElm );
            }
            
            word = "";
        }
        ++count;
    }
    
    return ret;
}


static void RecFindDXElmByType(vector<ECSmtPtr<DXElm> >& elm,
                               vector<ECSmtPtr<DXElm> >& dest,
                               const string& target,
                               const bool& print=false,                               
                               const int& level=0
                               ) {
    vector<ECSmtPtr<DXElm> >::iterator it = elm.begin();
    while ( it != elm.end() ) {
        ECSmtPtr<DXElm> theElm = *it;
        if( print ) {
            string space = ""; for( int i=0; i<level; ++i ) { space += "  "; }
            cout << space << theElm->type << " " << theElm->name << endl;
        }
        if( theElm->type == target ) {
            dest.push_back( theElm );
        }
        RecFindDXElmByType( theElm->children, dest, target, print, level+1 );
        
        ++it;
    }
}
static vector<DXFrameTree> RecMakeDXFrameTree( vector<ECSmtPtr<DXElm> >& elm ) {
    vector<DXFrameTree> ret;
    vector<ECSmtPtr<DXElm> >::iterator it = elm.begin();
    while ( it != elm.end() ) {
        ECSmtPtr<DXElm> theElm = *it;
        if( theElm->type == DX_Frame ) {
            ECMatrix tfM;
            vector<ECSmtPtr<DXElm> >::iterator it2 = theElm->children.begin();
            while ( it2 != theElm->children.end() ) {
                if( (*it2)->type == DX_FrameTransformMatrix ) {
                    ECSmtPtr<DXFrameTransformMatrix> dxftM = *it2;
                    tfM = dxftM->transformMatrix;
                }
                ++it2;
            }
            DXFrameTree ft;
            ft.frameName = theElm->name;
            ft.transformMatrix = tfM;
            vector<DXFrameTree> ch = RecMakeDXFrameTree( theElm->children );
            vector<DXFrameTree>::iterator it3 = ch.begin();
            while ( it3 != ch.end() ) {
                ft.children.push_back( *it3 );
                ++it3;
            }
            ret.push_back( ft );
        }
        ++it;
    }
    return ret;
}
static float4 ECVector3DToFloat4( const ECVector3D& v3 ) {
    return make_float4( v3.x, v3.y, v3.z, 1.0f );
}
static float4 ECVector2DToFloat4( const ECVector2D& v2, const float& z, const float& w ) {
    return make_float4( v2.x, v2.y, z, w );
}
void ECDirectXModel::load( string filePath ) {
    
    maxTime = 0.0f;
    
    string resource = "";
    ReadAndcutSpace( filePath, &resource );
    
    vector<ECSmtPtr<DXElm> > rootElms = ParseToDXElms( resource );
    
    ECSmtPtr<DXMesh> meshElm;
    ECSmtPtr<DXMeshNormals> normalElm;
    ECSmtPtr<DXMeshTextureCoords> uvElm;
    map<string, ECSmtPtr<DXMaterial> > materialElms;
    map<string, ECSmtPtr<DXTextureFilename> > textureFileNameElms;
    ECSmtPtr<DXMeshMaterialList> materialListElm;
    
    vector<ECSmtPtr<DXElm> > mesh;
    RecFindDXElmByType( rootElms, mesh, DX_Mesh );
    vector<ECSmtPtr<DXElm> >::iterator mesh_it = mesh.begin();
    while ( mesh_it != mesh.end() ) {
        meshElm = *mesh_it;
        
        vector<ECSmtPtr<DXElm> > normal;
        RecFindDXElmByType( meshElm->children, normal, DX_MeshNormals );
        vector<ECSmtPtr<DXElm> >::iterator normal_it = normal.begin();
        while ( normal_it != normal.end() ) {
            normalElm = *normal_it;
            ++normal_it;
        }
        
        vector<ECSmtPtr<DXElm> > uv;
        RecFindDXElmByType( meshElm->children, uv, DX_MeshTextureCoords );
        vector<ECSmtPtr<DXElm> >::iterator uv_it = uv.begin();
        while ( uv_it != uv.end() ) {
            uvElm = *uv_it;
            ++uv_it;
        }
        
        ++mesh_it;
    }
    
    vector<ECSmtPtr<DXElm> > material;
    RecFindDXElmByType( rootElms, material, DX_Material );
    vector<ECSmtPtr<DXElm> >::iterator material_it = material.begin();
    while ( material_it != material.end() ) {
        ECSmtPtr<DXMaterial> materialElm = *material_it;
        materialElms.insert( make_pair(materialElm->name,materialElm) );
        vector<ECSmtPtr<DXElm> > textureFileName;
        RecFindDXElmByType( materialElm->children, textureFileName, DX_TextureFilename );
        vector<ECSmtPtr<DXElm> >::iterator textureFileName_it = textureFileName.begin();
        while ( textureFileName_it != textureFileName.end() ) {
            ECSmtPtr<DXTextureFilename> textureFileNameElm = *textureFileName_it;
            textureFileNameElms.insert( make_pair(textureFileNameElm->name,textureFileNameElm) );
            ++textureFileName_it;
        }
        ++material_it;
    }
    
    vector<ECSmtPtr<DXElm> > materialList;
    RecFindDXElmByType( rootElms, materialList, DX_MeshMaterialList );
    vector< ECSmtPtr<DXElm> >::iterator materialList_it = materialList.begin();
    while ( materialList_it != materialList.end()  ) {
        materialListElm = *materialList_it;
        ++materialList_it;
    }
    
    vector<string> materialNames;
    map<string, ECSmtPtr<DXMaterial> >::iterator materialElms_it = materialElms.begin();
    while ( materialElms_it != materialElms.end() ) {
        string key = (*materialElms_it).first;
        materialNames.push_back( key );
        ++materialElms_it;
    }
    
    vector<string>::iterator names_it = materialNames.begin();
    while ( names_it != materialNames.end() ) {
        string theName = *names_it;
        {
            int num_of_indexes = meshElm->indexes.size();
            int hostVertexesSize = sizeof(ECDevVertex)*num_of_indexes*3;
            ECDevVertex *hostVertexes = (ECDevVertex*)malloc( hostVertexesSize );
            for( int i=0; i<num_of_indexes; ++i ) {
                hostVertexes[i*3+0].position = ECVector3DToFloat4( meshElm->vertexes[meshElm->indexes[i].index1] );
                hostVertexes[i*3+1].position = ECVector3DToFloat4( meshElm->vertexes[meshElm->indexes[i].index2] );
                hostVertexes[i*3+2].position = ECVector3DToFloat4( meshElm->vertexes[meshElm->indexes[i].index3] );
                hostVertexes[i*3+0].normal = ECVector3DToFloat4( normalElm->normals[normalElm->indexes[i].index1] );
                hostVertexes[i*3+1].normal = ECVector3DToFloat4( normalElm->normals[normalElm->indexes[i].index2] );
                hostVertexes[i*3+2].normal = ECVector3DToFloat4( normalElm->normals[normalElm->indexes[i].index3] );
                hostVertexes[i*3+0].uv = ECVector2DToFloat4( uvElm->uvs[meshElm->indexes[i].index1], 1.0f, 1.0f );
                hostVertexes[i*3+1].uv = ECVector2DToFloat4( uvElm->uvs[meshElm->indexes[i].index2], 1.0f, 1.0f );
                hostVertexes[i*3+2].uv = ECVector2DToFloat4( uvElm->uvs[meshElm->indexes[i].index3], 1.0f, 1.0f );
                const float r = 1.0f;
                const float g = 1.0f;
                const float b = 1.0f;
                const float a = 1.0f;
                hostVertexes[i*3+0].color = make_float4( r,g,b,a );
                hostVertexes[i*3+1].color = make_float4( r,g,b,a );
                hostVertexes[i*3+2].color = make_float4( r,g,b,a );
            }
            srcVertexes[theName]  = ECSmtDevPtr<ECDevVertex>( hostVertexes, hostVertexesSize );
            destVertexes[theName] = ECSmtDevPtr<ECDevVertex>( hostVertexes, hostVertexesSize );
            boneAnimedVertexes[theName] = ECSmtDevPtr<ECDevVertex>( hostVertexes, hostVertexesSize );
            free( hostVertexes );
        }
        {
            ECSmtPtr<DXMaterial> theMaterial = materialElms[theName];
            ECDevMaterial *hostMaterial = (ECDevMaterial*)malloc( sizeof(ECDevMaterial) );
            hostMaterial->ambient = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
            hostMaterial->diffuse = make_float4(1.0, 1.0, 1.0, 1.0f);
            hostMaterial->specular = theMaterial->specular;
            hostMaterial->shininess = make_float4( theMaterial->shininess, theMaterial->shininess, theMaterial->shininess, 1.0f);
            hostMaterial->emission = theMaterial->emission;
            {
                string dir = filePath;
                dir = dir.substr(0,dir.find_last_of("/"));
                string imgFilePath = dir + "/" + textureFileNameElms[theName]->fileName;
                int texW = 0;
                int texH = 0;
                textures[theName] = new ECModelTextures();
                textures[theName]->colorTexture = loadTexture( imgFilePath, &texW, &texH );
                textures[theName]->colorTextureWidth = texW;
                textures[theName]->colorTextureHeight = texH;
                textures[theName]->hasColorTexture = true;
                hostMaterial->colorTextureSize.x = texW;
                hostMaterial->colorTextureSize.y = texH;
            }
            materials[theName] = ECSmtDevPtr<ECDevMaterial>( hostMaterial, sizeof(ECDevMaterial) );
            free( hostMaterial );
        }
        ++names_it;
    }
    
    // animation ----------------------------------------------------------------------------
    vector<ECSmtPtr<DXElm> > animationSet;
    RecFindDXElmByType( rootElms, animationSet, DX_AnimationSet );
    vector<ECSmtPtr<DXElm> >::iterator animationSet_it = animationSet.begin();
    while ( animationSet_it != animationSet.end() ) {
        string name = (*animationSet_it)->name;
        if( name.size() == 0 ) { name = "default"; }
        animationNames.push_back( name );
        ++animationSet_it;
    }
    
    map<string, ECSmtPtr<DXAnimationKey> > animKeys;
    vector<ECSmtPtr<DXElm> > animations;
    RecFindDXElmByType( rootElms, animations, DX_Animation );
    vector<ECSmtPtr<DXElm> >::iterator animations_it = animations.begin();
    while ( animations_it != animations.end() ) {
        ECSmtPtr<DXAnimation> anim = *animations_it;
        vector<ECSmtPtr<DXElm> >::iterator animKey_it = anim->children.begin();
        ECSmtPtr<DXAnimationKey> animationKey;
        while ( animKey_it != anim->children.end() ) {
            if( (*animKey_it)->type == DX_AnimationKey ) {
                animationKey = *animKey_it;
                ++animKey_it;
            }
        }
        animKeys.insert( make_pair( anim->animationName, animationKey ) );
        
        ++animations_it;
    }
    
    map<string, ECSmtPtr<DXAnimationKey> >::iterator animationKeyMap_it = animKeys.begin();
    while ( animationKeyMap_it != animKeys.end() ) {
        string animName = ReduceSpaceChar( (*animationKeyMap_it).first );
        ECSmtPtr<DXAnimationKey> animKey = (*animationKeyMap_it).second;
        animationKeys.insert( make_pair(animName,animKey->keys) );
        
        vector<pair<float,ECMatrix> >::iterator key_it = animKey->keys.begin();
        while ( key_it != animKey->keys.end() ) {
            const float time = (*key_it).first;
            if( maxTime < time ) {
                maxTime = time;
            }
            ++key_it;
        }
        
        ++animationKeyMap_it;
    }
    
    vector<ECSmtPtr<DXElm> >::iterator frame_it = rootElms.begin();
    frameTree = RecMakeDXFrameTree( rootElms );
    
    vector<ECSmtPtr<DXSkinWeights> > skinWeightsElm;
    vector<ECSmtPtr<DXElm> > sw;
    RecFindDXElmByType( rootElms, sw, DX_SkinWeights );
    vector<ECSmtPtr<DXElm> >::iterator skinWeights_it = sw.begin();
    while ( skinWeights_it != sw.end() ) {
        skinWeightsElm.push_back( *skinWeights_it );
        ++skinWeights_it;
    }
    vector<ECSmtPtr<DXSkinWeights> >::iterator skinWeightsElm_it = skinWeightsElm.begin();
    while ( skinWeightsElm_it != skinWeightsElm.end() ) {
        ECSmtPtr<DXSkinWeights> sWeights = *skinWeightsElm_it;
        int num_of_indexes = sWeights->indexes.size();
        DXDevWeight* hostWeight = (DXDevWeight*)malloc( sizeof(DXDevWeight)*num_of_indexes ); {
            for( int i=0; i<num_of_indexes; ++i ) {
                hostWeight[i].index = sWeights->indexes[i];
                hostWeight[i].weight = sWeights->weights[i];
            }
            skinWeights[sWeights->boneName] = ECSmtDevPtr<DXDevWeight>( hostWeight, sizeof(DXDevWeight)*num_of_indexes );
        } free( hostWeight );
        devSetMatrix( offsetMatrixes[sWeights->boneName], sWeights->offsetMatrix );
        ++skinWeightsElm_it;
    }
    
    vector<string> animNames = getAnimationNames();
    vector<string>::iterator animNames_it = animNames.begin();
    while ( animNames_it != animNames.end() ) {
        string animName = *animNames_it;
        ECMatrix hostMatrix;
        devSetMatrix( currentBoneMatrixes[animName], hostMatrix );
        ++animNames_it;
    }
}

static ECMatrix DXInterpolateMatrix(const float& time,
                                    const ECMatrix& mat1, const float& time_1,
                                    const ECMatrix& mat2, const float& time_2 ) {
    ////////////////////////////////////////// TODO: implement
    return mat1;
    //////////////////////////////////////////
}
static void DXAnimate(map<string,ECMatrix>& dest,
                      const float& time,
                      DXFrameTree& tree,
                      map<string, vector<pair<float,ECMatrix> > >& animationKeys,
                      ECMatrix& currentAnimMatrix
                      ) {
    
    string boneName = tree.frameName;
    
    vector<pair<float,ECMatrix> >& currentAnimKey = animationKeys[boneName];
    float time_1 = 0;
    float time_2 = 0;
    ECMatrix mat1 = ECMatrix();
    ECMatrix mat2 = ECMatrix();
    int limit = currentAnimKey.size()-1;
    for( int i=0; i<limit; ++i ) {
        const float t1 = currentAnimKey[i].first;
        const float t2 = currentAnimKey[i+1].first;
        if( t1 <= time && time <= t2 ) {
            time_1 = t1;
            time_2 = t2;
            mat1 = currentAnimKey[i].second;
            mat2 = currentAnimKey[i+1].second;
        }
    }
    ECMatrix animMatrix = DXInterpolateMatrix( time, mat1, time_1, mat2, time_2 );
    currentAnimMatrix *= animMatrix;
    dest[boneName] = currentAnimMatrix;
    
    vector<DXFrameTree>::iterator ch_it = tree.children.begin();
    while ( ch_it != tree.children.end() ) {
        DXAnimate( dest, time, *ch_it, animationKeys, currentAnimMatrix );
        ++ch_it;
    }
}

void ECDirectXModel::animate( const float& f0_f1 ) {
    
    map<string,ECMatrix> dest;
    
    vector<DXFrameTree>::iterator frameTree_it = frameTree.begin();
    while ( frameTree_it != frameTree.end() ) {
        DXFrameTree ft = *frameTree_it;
        ECMatrix rootMat = ECMatrix();
        DXAnimate( dest, maxTime*f0_f1, ft, animationKeys, rootMat );
        ++frameTree_it;
    }

    vector<string> animNames = getAnimationNames();
    vector<string>::iterator animNames_it = animNames.begin();
    while ( animNames_it != animNames.end() ) {
        string animName = *animNames_it;
        devCopyMatrix( currentBoneMatrixes[animName], dest[animName] );
        ++animNames_it;
    }
    
}


