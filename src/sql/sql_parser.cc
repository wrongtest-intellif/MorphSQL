#include "sql/sql_parser.h"

#include <sstream>

#include "antlr4-runtime/antlr4-runtime.h"
#include "SqlBaseLexer.h"
#include "SqlBaseParser.h"
#include "SqlBaseBaseVisitor.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"

#include "core/context.h"
#include "logical/logical_op.h"

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

using morph::core::MorphContext;

class QueryScopeGuard {

};

class SQLVisitor : public SqlBaseBaseVisitor {
 public:
 	SQLVisitor(mlir::OpBuilder* op_builder)
 		: builder_(op_builder) {}

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
    antlrcpp::Any visitQueryOrganization(SqlBaseParser::QueryOrganizationContext *ctx) override {
    	return 0;
    }

    // queryTerm
    antlrcpp::Any visitQueryTermDefault(SqlBaseParser::QueryTermDefaultContext *ctx) override {
    	return 0;
    }

    // queryPrimary
    antlrcpp::Any visitQueryPrimaryDefault(SqlBaseParser::QueryPrimaryDefaultContext *ctx) override {
    	return 0;
    }

    // regularQuerySpecification
    antlrcpp::Any visitRegularQuerySpecification(SqlBaseParser::RegularQuerySpecificationContext *ctx) override {
    	return 0;
    }

    // selectClause
    antlrcpp::Any visitSelectClause(SqlBaseParser::SelectClauseContext *ctx) override {
    	return 0;
    }

    antlrcpp::Any visitFromClause(SqlBaseParser::FromClauseContext *context) override {
    	return 0;
    }

    // simple table relation => op
    antlrcpp::Any visitTableName(SqlBaseParser::TableNameContext *ctx) override {
    	logical::TableOp op = builder_->create<logical::TableOp>(builder_->getUnknownLoc());
    	mlir::Value value(op);
    }

    antlrcpp::Any visitPivotClause(SqlBaseParser::PivotClauseContext *context) override {
    	return 0;
    }

    // relation => op
    antlrcpp::Any visitRelation(SqlBaseParser::RelationContext *context) override {
    	return 0;
    }

    antlrcpp::Any visitJoinRelation(SqlBaseParser::JoinRelationContext *context) override {
    	return 0;
    }

    antlrcpp::Any visitWhereClause(SqlBaseParser::WhereClauseContext *context) override {
    	return 0;
    }

    antlrcpp::Any visitHavingClause(SqlBaseParser::HavingClauseContext *context) override {
    	return 0;
    }

    antlrcpp::Any visitWindowClause(SqlBaseParser::WindowClauseContext *context) override {
    	return 0;
    }

    antlrcpp::Any visitAggregationClause(SqlBaseParser::AggregationClauseContext *context) override {
    	return 0;
    }

    antlrcpp::Any visitLateralView(SqlBaseParser::LateralViewContext *context) override {
    	return 0;
    }

 private:
 	void SetCurrentQuery() {}
 	void AddCurrentQueryName(const std::string& name) {}

 	mlir::OpBuilder* builder_;
};

SQLParser::SQLParser(core::MorphContext* morph_ctx): morph_ctx_(morph_ctx) {}


logical::LogicalPlan SQLParser::Parse(const std::string& sql) {
	// initialize one mlir module for sql logical plan
	mlir::MLIRContext* mlir_ctx = morph_ctx_->GetMLIRContext();
	mlir::OpBuilder logical_builder(mlir_ctx);
	auto dummy_loc = logical_builder.getUnknownLoc();
	mlir::ModuleOp logical_module = mlir::ModuleOp::create(dummy_loc);

	// initialize a logical func return query output table
	auto res_ty = mlir::detail::TypeUniquer::get<logical::NNType>(mlir_ctx, logical::LogicalResultTypes::Table);
	res_ty.print(llvm::outs());


	// auto res_ty = mlir::FloatType::getF32(mlir_ctx); //logical::LogicalTableType::get(mlir_ctx);
	auto func_ty = mlir::FunctionType::get({}, {res_ty}, mlir_ctx);
	auto func_op = mlir::FuncOp::create(dummy_loc, "main", func_ty);
	logical_builder.setInsertionPointToStart(func_op.addEntryBlock());

	std::stringstream ss(sql);
    antlr4::ANTLRInputStream input(ss);
    SqlBaseLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);
    SqlBaseParser parser(&tokens);

    SQLVisitor visitor(&logical_builder);
    parser.singleStatement()->accept(&visitor);
    logical::TableOp op = logical_builder.create<logical::TableOp>(dummy_loc);

    logical_module.push_back(func_op);
    logical_module.print(llvm::outs());
}

}  // namespace sql
}  // namespace morph

