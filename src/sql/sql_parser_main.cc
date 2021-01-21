#include "sql/sql_parser.h"

int main() {
	mlir::registerDialect<morph::logical::LogicalDialect>();
	morph::core::MorphContext morph_ctx;
	morph::sql::SQLParser parser(&morph_ctx);
	parser.Parse("SELECT X, \"11234sdasf\" FROM T;");
	return 0;
}
