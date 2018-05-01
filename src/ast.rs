use regex_syntax;

#[derive(Debug)]
pub struct Expr {
    kind: ExprKind,
    span: Span,
}

impl Expr {
    pub fn new(kind: ExprKind, span: Span) -> Self {
        Self {
            kind: kind,
            span: span,
        }
    }

    pub fn kind(&self) -> &ExprKind {
        &self.kind
    }
}

#[derive(Debug)]
pub enum ExprKind {
    RegexLiteral(regex_syntax::ast::Ast),
    /// A poison expression is never valid, but it lets us avoid copying
    /// the source string and still please the borrow checker.
    #[doc(hidden)]
    ExprPoison,
}

#[derive(Debug)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}
