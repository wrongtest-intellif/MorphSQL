#ifndef SQL_SQL_PARSER_H
#define SQL_SQL_PARSER_H

#include <string>

#include "core/context.h"
#include "logical/logical_op.h"

namespace morph {
namespace sql {

class SQLParser {
 public:
 	SQLParser(core::MorphContext* morph_ctx);
 	logical::LogicalPlan Parse(const std::string& sql);

 private:
 	core::MorphContext* morph_ctx_;
};

}  // namespace sql
}  // namespace morph
#endif  // SQL_SQL_PARSER_H
