/**
 * @file meansquarederror.h
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-07-03
 *
 * @copyright Copyright (c) 2019
 *
 */
#pragma once

#include "compute/add/addop.h"
#include "compute/negative/negativeop.h"
#include "compute/operation.h"
#include "compute/pow/powop.h"
#include "compute/reducesum/reducesumop.h"
#include "compute/scalarproduct/scalarproductop.h"

namespace magmadnn {
namespace op {

template <typename T>
Operation<T> *meansquarederror(Operation<T> *ground_truth, Operation<T> *prediction);
}
}  // namespace magmadnn