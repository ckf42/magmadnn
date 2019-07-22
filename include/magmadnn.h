/**
 * @file magmadnn.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-07
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

/* include all magmadnn header files */

#if defined(USE_GPU)
#define _HAS_CUDA_
#endif

#if defined(_HAS_CUDA_)
#define USE_GPU
#endif

#include <cuda_runtime_api.h>

//  can only use one of the cusparse api, prefer new one
#if defined(USE_CUSPARSE_NEW_API)
#undef USE_CUSPARSE_OLD_API
#elif defined(USE_CUSPARSE_OLD_API)
#undef USE_CUSPARSE_NEW_API
#endif

//  not specified which api to use
#if !defined(USE_CUSPARSE_NEW_API) && !defined(USE_CUSPARSE_OLD_API)
//  determine according to cuda version
#if (CUDART_VERSION >= 10010)
#define USE_CUSPARSE_NEW_API
#else
#define USE_CUSPARSE_OLD_API
#endif
#endif

#include "init_finalize.h"
#include "types.h"
#include "utilities_internal.h"

#include "memory/memorymanager.h"
#include "tensor/tensor.h"
#include "tensor/tensor_io.h"

#include "math/tensor_math.h"

#include "compute/gradients.h"
#include "compute/tensor_operations.h"
#include "compute/variable.h"

#include "layer/layers.h"

#include "model/models.h"
#include "optimizer/optimizers.h"

#include "dataloader/dataloaders.h"

#include "sparseMatrix/sparseMatrix.h"
