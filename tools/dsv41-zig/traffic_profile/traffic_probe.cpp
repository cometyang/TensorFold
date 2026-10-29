// Build with g++ and the CUDA/CUPTI include and library directories; no nvcc needed.
// --query [comma-separated substrings] launches no GPU workload or allocations.
// --smoke explicitly allocates memory and launches a small CUDA graph: use only
// during the maintenance window. Environment contract is in traffic_profile.h.
#include "traffic_profile.h"
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void check(CUresult r) {
    if(r==CUDA_SUCCESS)return; const char *s=nullptr;cuGetErrorString(r,&s);
    std::fprintf(stderr,"CUDA: %s\n",s?s:"error");std::exit(2);
}
int main(int argc,char **argv) {
    if(argc>=3 && !std::strcmp(argv[1],"--chip"))
        return tf_traffic_query_chip(argv[2],argc>3?argv[3]:"dram,sysmem,c2c,lts__d_sectors");
    bool large=argc>1 && !std::strcmp(argv[1],"--smoke64MiB");
    if(argc<2 || (std::strcmp(argv[1],"--query") && std::strcmp(argv[1],"--smoke") && !large)) {
        std::fprintf(stderr,"usage: traffic_probe --query [filters] | --chip NAME [filters] | --smoke | --smoke64MiB\n"); return 2;
    }
    check(cuInit(0)); CUdevice device;check(cuDeviceGet(&device,0));
    CUcontext context;check(cuDevicePrimaryCtxRetain(&context,device));check(cuCtxSetCurrent(context));
    if(!std::strcmp(argv[1],"--query")) {
        int result=tf_traffic_query(argc>2?argv[2]:"dram,sysmem,c2c,lts__d_sectors");
        check(cuDevicePrimaryCtxRelease(device));return result;
    }
    static const char ptx[]=R"(.version 8.0
.target sm_80
.address_size 64
.visible .entry touch(.param .u64 ptr) {
 .reg .b32 %i,%x,%b,%n;
 .reg .b64 %p,%o;
 ld.param.u64 %p,[ptr]; mov.u32 %i,%tid.x; mov.u32 %b,%ctaid.x;
 mov.u32 %n,%ntid.x; mad.lo.u32 %i,%b,%n,%i;
 mul.wide.u32 %o,%i,4; add.u64 %p,%p,%o;
 ld.global.u32 %x,[%p]; add.u32 %x,%x,1; st.global.u32 [%p],%x; ret;
})";
    CUmodule module;CUfunction kernel;check(cuModuleLoadData(&module,ptx));check(cuModuleGetFunction(&kernel,module,"touch"));
    size_t bytes=large?64u<<20:1u<<20;
    CUdeviceptr buffer;check(cuMemAlloc(&buffer,bytes));check(cuMemsetD32(buffer,0,bytes/4));
    CUstream stream;check(cuStreamCreate(&stream,CU_STREAM_NON_BLOCKING));
    void *args[]={&buffer};CUgraph graph;CUgraphExec executable;
    check(cuStreamBeginCapture(stream,CU_STREAM_CAPTURE_MODE_GLOBAL));
    check(cuLaunchKernel(kernel,bytes/(256*4),1,1,256,1,1,0,stream,args,nullptr));
    check(cuStreamEndCapture(stream,&graph));check(cuGraphInstantiateWithFlags(&executable,graph,0));
    check(cuGraphLaunch(executable,stream));check(cuStreamSynchronize(stream));
    int selected=0, iterations=large?6:12;
    for(int i=0;i<iterations;++i) {
        int begin=tf_traffic_begin(2048+i,2,0); if(begin<0)return 3; selected+=begin;
        check(cuGraphLaunch(executable,stream));check(cuStreamSynchronize(stream));
        if(tf_traffic_end()<0)return 3;
    }
    tf_traffic_finish();unsigned value=0;check(cuMemcpyDtoH(&value,buffer,sizeof(value)));
    std::printf("{\"type\":\"smoke\",\"selected\":%d,\"buffer_bytes\":%zu,\"value\":%u,\"expected\":%d}\n",selected,bytes,value,iterations+1);
    check(cuGraphExecDestroy(executable));check(cuGraphDestroy(graph));check(cuStreamDestroy(stream));
    check(cuMemFree(buffer));check(cuModuleUnload(module));check(cuDevicePrimaryCtxRelease(device));
    return value==unsigned(iterations+1)?0:4;
}
