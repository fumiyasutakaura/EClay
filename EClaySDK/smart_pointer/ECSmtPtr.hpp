#ifndef EC_SMTPTR_HPP
#define EC_SMTPTR_HPP


template <typename T>
class ECSmtPtr {
private:
    T* ptr;
    int* refCount;
    
    void delPtr() {
        if( ptr!=(T*)0 ) {
            delete ptr;
            ptr = (T*)0;
        }
    }
    void delRefCountPtr() {
        if( refCount!=(int*)0 ) {
            delete refCount;
            refCount = (int*)0;
        }
    }
    void reduceCount() {
        if( refCount!=(int*)0 ) {
            --(*refCount);
            if( *refCount <= 0 ) {
                delPtr();
                delRefCountPtr();
            }
        }
        else {
            delPtr();
        }
    }
    
public:
    ~ECSmtPtr() {
        reduceCount();
    }
    ECSmtPtr() {
        ptr = (T*)0;
        refCount = new int(1);
    }
    ECSmtPtr( const T* obj ) {
        ptr = const_cast<T*>(obj);
        refCount = new int(1);
    }
    // > constructer with ECSmtPtr<type>& --------
    template <typename U>
    ECSmtPtr( const ECSmtPtr<U>& obj ) {
        ptr = (T*)obj.getPtr();
        refCount = obj.getRefCount();
        ++(*refCount);
    }
    ECSmtPtr( const ECSmtPtr<T>& obj ) {  // specialization
        ptr = obj.getPtr();
        refCount = obj.getRefCount();
        ++(*refCount);
    }
    // < constructer with ECSmtPtr<type>& --------
    const ECSmtPtr<T>& operator=( const T* obj ) {
        delPtr();
        ptr = const_cast<T*>(obj);
        refCount = new int(1);
        return *this;
    }
    const ECSmtPtr<T>& operator=( const ECSmtPtr<T>& obj ) {
        if( ptr == obj.getPtr() ) { return *this; }
        reduceCount();
        ptr = obj.getPtr();
        refCount = obj.getRefCount();
        ++(*refCount);
        return *this;
    }
    
    bool isNull() const {
        if( ptr == (T*)0 ) { return true; }
        return false;
    }
    
    bool operator==( const ECSmtPtr<T>& obj ) const {
        return ptr == obj.getPtr();
    }
    
    T* operator->() const {
        return this->ptr;
    }
    
    T& operator[]( const int index ) const {
        return ptr[index];
    }
    
    T* getPtr() const {
        return this->ptr;
    }
    
    int* getRefCount() const {
        return this->refCount;
    }
    
};


extern "C" {
    void gpuMalloc( void*& dev_ptr, int size );
    void gpuFree( void*& dev_ptr );
    void gpuMemcpyDeviceToHost( void*& host_ptr, void*& dev_ptr, int size );
    void gpuMemcpyHostToDevice( void*& dev_ptr, void*& host_ptr, int size );
    void gpuMemcpyDeviceToDevice( void*& to, void*& from, int size );
}
template <typename T>
class ECSmtDevPtr {
private:
    T* ptr;
    int* refCount;
    int size;
    
    void delPtr() {
        if( ptr!=(T*)0 ) {
            gpuFree( (void*&)ptr );
            ptr = (T*)0;
        }
    }
    void delRefCountPtr() {
        if( refCount!=(int*)0 ) {
            delete refCount;
            refCount = (int*)0;
        }
    }
    void reduceCount() {
        if( refCount!=(int*)0 ) {
            --(*refCount);
            if( *refCount <= 0 ) {
                delPtr();
                delRefCountPtr();
            }
        }
        else {
            delPtr();
        }
    }
    
public:
    
    int getSize() const {
        return size;
    }
    
    void copyDeviceToHost( T* host_ptr ) {
        gpuMemcpyDeviceToHost( (void*&)host_ptr, (void*&)ptr, size );
    }
    void copyHostToDevice( T* host_ptr ) {
        gpuMemcpyHostToDevice( (void*&)ptr, (void*&)host_ptr, size );
    }
    void copyDeviceToDevice( T* dev_ptr ) {
        gpuMemcpyDeviceToDevice( (void*&)ptr, (void*&)dev_ptr, size );
    }
    
    ~ECSmtDevPtr() {
        reduceCount();
    }
    
    ECSmtDevPtr() {
        size = 0;
        ptr = (T*)0;
        refCount = new int(1);
    }
    ECSmtDevPtr( const int s ) {
        size = s;
        gpuMalloc( (void*&)ptr, size );
        refCount = new int(1);
    }
    ECSmtDevPtr( const T* host_ptr, const int s ) {
        size = s;
        gpuMalloc( (void*&)ptr, size );
        gpuMemcpyHostToDevice( (void*&)ptr, (void*&)host_ptr, size );
        refCount = new int(1);
    }
    ECSmtDevPtr( const ECSmtDevPtr<T>& argDev ) {
        ptr = argDev.getPtr();
        refCount = argDev.getRefCount();
        size = argDev.size;
        ++(*refCount);
    }
    const ECSmtDevPtr<T>& operator=( const ECSmtDevPtr<T>& argDev ) {
        if( ptr == argDev.getPtr() ) { return *this; }
        reduceCount();
        size = argDev.size;
        ptr = argDev.getPtr();
        refCount = argDev.getRefCount();
        ++(*refCount);
        return *this;
    }
    
    bool isNull() const {
        if( ptr == (T*)0 ) { return true; }
        return false;
    }
    
    bool operator==( const ECSmtDevPtr<T>& obj ) const {
        return ptr == obj.getPtr();
    }
    
    T* operator->() const {
        return this->ptr;
    }
        
    T* getPtr() const {
        return this->ptr;
    }
    
    int* getRefCount() const {
        return this->refCount;
    }
        
};





#endif
