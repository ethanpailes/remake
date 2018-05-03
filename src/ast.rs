use regex_syntax;
use regex_syntax::ast::{RepetitionKind, GroupKind};

use error::InternalError;
use operators;
use util::POISON_SPAN;

#[derive(Debug)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

impl Expr {
    pub fn new(kind: ExprKind, span: Span) -> Self {
        Self {
            kind: kind,
            span: span,
        }
    }

    pub fn eval(self) -> Result<Box<regex_syntax::ast::Ast>, InternalError> {
        self.eval_(&mut EvalEnv::new())
    }

    fn eval_(
        self,
        _env: &mut EvalEnv,
    ) -> Result<Box<regex_syntax::ast::Ast>, InternalError> {

        match self.kind {
            ExprKind::RegexLiteral(r) => Ok(r),
            ExprKind::BinOp(lhs, op, rhs) => {
                match op {
                    BOp::Concat =>
                        Ok(operators::concat(lhs.eval()?, rhs.eval()?)),
                    BOp::Alt =>
                        Ok(operators::alt(lhs.eval()?, rhs.eval()?))
                }
            }
            ExprKind::UnaryOp(op, e) => {
                match op {
                    UOp::RepeatZeroOrMore(greedy) => {
                        Ok(Box::new(regex_syntax::ast::Ast::Repetition(
                            regex_syntax::ast::Repetition {
                                span: POISON_SPAN,
                                op: regex_syntax::ast::RepetitionOp {
                                    span: POISON_SPAN,
                                    kind: RepetitionKind::ZeroOrMore,
                                },
                                greedy: greedy,
                                ast: e.eval()?,
                            })))
                    }
                    UOp::RepeatOneOrMore(greedy) => {
                        Ok(Box::new(regex_syntax::ast::Ast::Repetition(
                            regex_syntax::ast::Repetition {
                                span: POISON_SPAN,
                                op: regex_syntax::ast::RepetitionOp {
                                    span: POISON_SPAN,
                                    kind: RepetitionKind::OneOrMore,
                                },
                                greedy: greedy,
                                ast: e.eval()?,
                            })))
                    }
                    UOp::RepeatZeroOrOne(greedy) => {
                        Ok(Box::new(regex_syntax::ast::Ast::Repetition(
                            regex_syntax::ast::Repetition {
                                span: POISON_SPAN,
                                op: regex_syntax::ast::RepetitionOp {
                                    span: POISON_SPAN,
                                    kind: RepetitionKind::ZeroOrOne,
                                },
                                greedy: greedy,
                                ast: e.eval()?,
                            })))
                    }
                    UOp::RepeatRange(greedy, range) => {
                        Ok(Box::new(regex_syntax::ast::Ast::Repetition(
                            regex_syntax::ast::Repetition {
                                span: POISON_SPAN,
                                op: regex_syntax::ast::RepetitionOp {
                                    span: POISON_SPAN,
                                    kind: RepetitionKind::Range(range),
                                },
                                greedy: greedy,
                                ast: e.eval()?,
                            })))
                    }
                }
            }
            ExprKind::Capture(e, name) => {
                Ok(Box::new(regex_syntax::ast::Ast::Group(
                    regex_syntax::ast::Group {
                        span: POISON_SPAN,
                        kind: match name {
                            Some(n) => GroupKind::CaptureName(
                                regex_syntax::ast::CaptureName {
                                    span: POISON_SPAN,
                                    name: n,
                                    index: BOGUS_GROUP_INDEX,
                                }
                            ),
                            None => GroupKind::CaptureIndex(BOGUS_GROUP_INDEX),
                        },
                        ast: e.eval()?,
                    })))
            }
            ExprKind::ExprPoison => panic!("Bug in remake."),
        }
    }
}

/// We don't have to spend any effort assigning indicies to groups because
/// we are going to pretty-print the AST and have regex just parse it.
/// If we passed the AST to the regex crate directly, we would need some
/// way to thread the group index through its parser. This way we can
/// just ignore the whole problem.
const BOGUS_GROUP_INDEX: u32 = 0;

struct EvalEnv {
}
impl EvalEnv {
    fn new() -> Self {
        EvalEnv {}
    }
}

#[derive(Debug)]
pub enum ExprKind {
    BinOp(Box<Expr>, BOp, Box<Expr>),
    UnaryOp(UOp, Box<Expr>),
    Capture(Box<Expr>, Option<String>),
    RegexLiteral(Box<regex_syntax::ast::Ast>),

    /// A poison expression is never valid, but it lets us avoid copying
    /// the source string and still please the borrow checker.
    #[doc(hidden)]
    ExprPoison,
}

#[derive(Debug)]
pub enum BOp {
    Concat,
    Alt,
}

#[derive(Debug)]
pub enum UOp {
    RepeatZeroOrMore(bool),
    RepeatOneOrMore(bool),
    RepeatZeroOrOne(bool),
    RepeatRange(bool, regex_syntax::ast::RepetitionRange),
}

#[derive(Debug)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}
