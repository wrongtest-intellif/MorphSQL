#include "sql/sql_parser.h"

int main() {
	morph::sql::SQLParser parser;
	parser.Parse("SELECT X, \"11234sdasf\" FROM T;");
	return 0;
}
