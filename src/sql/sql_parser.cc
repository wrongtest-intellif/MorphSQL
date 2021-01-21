#include "sql/sql_parser.h"

#include <sstream>

#include "antlr4-runtime/antlr4-runtime.h"
#include "SqlBaseLexer.h"
#include "SqlBaseParser.h"
#include "SqlBaseBaseVisitor.h"

#define REPORT_ERROR(...)

/**
  * Just return on sql parse failure.
  */
#define RETURN_ON_ERROR(stmt, ctx, ...) \
	while (1) {                         \
		stmt;                           \
		REPORT_ERROR(...);              \
        break;                          \
	}                                   \

#define CHECK_NON_NULL(obj, ctx, ...)


namespace morph {
namespace sql {

class QueryScopeGuard {

};

class SQLVisitor : public SqlBaseBaseVisitor {
 public:
 	// query
    antlrcpp::Any visitQuery(SqlBaseParser::QueryContext *ctx) override {
 		if (ctx->ctes()) {
 			RETURN_ON_ERROR(ctx->ctes()->accept(this), ctx);
 		}
 		CHECK_NON_NULL(ctx->queryOrganization(), ctx);
 		CHECK_NON_NULL(ctx->queryTerm(), ctx);
 		RETURN_ON_ERROR(ctx->queryOrganization()->accept(this), ctx);
    	return ctx->queryTerm()->accept(this);
    }

    // ctes
    antlrcpp::Any visitCtes(SqlBaseParser::CtesContext *ctx) override {
    	auto clauses = ctx->namedQuery();
    	for (SqlBaseParser::NamedQueryContext* named_ctx : clauses) {
    		QueryScopeGuard with_scope_guard;
    		RETURN_ON_ERROR(named_ctx->accept(this), ctx);
    	}
    	return false;
    }

    // namedQuery
    antlrcpp::Any visitNamedQuery(SqlBaseParser::NamedQueryContext *ctx) override {
    	CHECK_NON_NULL(ctx->query(), ctx);
    	CHECK_NON_NULL(ctx->identifierList(), ctx, "Anonymous with clause");

    	const auto& names = visitIdentifierList(ctx->identifierList()).as<std::vector<std::string>>();
    	for (const std::string& name : names) {
    		AddCurrentQueryName(name);
    	}
    	RETURN_ON_ERROR(ctx->query()->accept(this), ctx);
   	}

    // queryOrganization
    antlrcpp::Any visitQueryOrganization(SqlBaseParser::QueryOrganizationContext *context) override {

    }

    // queryTerm
    antlrcpp::Any visitQueryTermDefault(SqlBaseParser::QueryTermDefaultContext *context) override {

    }

 private:
 	void SetCurrentQuery();
 	void AddCurrentQueryName(const std::string& name);

};

void SQLParser::Parse(const std::string& sql) {
	std::stringstream ss(sql);
    antlr4::ANTLRInputStream input(ss);
    SqlBaseLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);
    SqlBaseParser parser(&tokens);

    SQLVisitor visitor;
    parser.singleStatement()->accept(&visitor);
}

}  // namespace sql
}  // namespace morph

