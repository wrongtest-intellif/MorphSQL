#ifndef SQL_SQL_PARSER_H
#define SQL_SQL_PARSER_H

#include <string>

namespace morph {
namespace sql {

class SQLParser {
 public:
 	void Parse(const std::string& sql);

};

}  // namespace sql
}  // namespace morph
#endif  // SQL_SQL_PARSER_H
