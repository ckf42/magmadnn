/**
 * @file tensor_math.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-06-06
 *
 * @copyright Copyright (c) 2019
 */

#pragma once

#include "math/add.h"
#include "math/argmax.h"
#include "math/concat.h"
#include "math/dot.h"
#include "math/dropout.h"
#include "math/matmul.h"
#include "math/pooling.h"
#include "math/pow.h"
#include "math/relu.h"
#include "math/scalar_tensor_product.h"
#include "math/sum.h"
#include "math/tile.h"
#include "reduce_sum.h"
#include "math/abs.h"

#include "math/optimizer_math/adagrad.h"
#include "math/optimizer_math/adam.h"
#include "math/optimizer_math/rmsprop.h"
#include "math/optimizer_math/sgd_momentum.h"