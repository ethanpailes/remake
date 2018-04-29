use regex_syntax;

#[derive(Debug)]
pub enum Expr {
    RegexLiteral(regex_syntax::ast::Ast),
}
