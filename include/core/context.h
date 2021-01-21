#ifndef CORE_CONTEXT_H
#define CORE_CONTEXT_H

#include "mlir/IR/Builders.h"

namespace morph {
namespace core {

class MorphContext {
 public:
 	mlir::MLIRContext* GetMLIRContext() { return &context_; }

 private:
    mlir::MLIRContext context_;
};

}  // namespace core	
}  // namespace morph

#endif  // CORE_CONTEXT_H
