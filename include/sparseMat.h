#if defined(_HAS_CUDA_)
#include "cusparse.h"
#include "magma.h"
#endif

namespace magmadnn {

struct sparseMat {
    enum {
        PLAIN_CSR, 
        #if defined(_HAS_CUDA_)

    }
};

}  // namespace magmadnn