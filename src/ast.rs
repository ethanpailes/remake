use regex_syntax;
use regex_syntax::ast::{RepetitionKind, GroupKind};

use error::InternalError;
use operators;

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
        env: &mut EvalEnv,
    ) -> Result<Box<regex_syntax::ast::Ast>, InternalError> {

        // TODO(ethan): There are some cases where it is not clear that there
        //              is a sensible value to give a regex ast span. Idk what
        //              to do there.
        let poison_span = regex_syntax::ast::Span::new(
            regex_syntax::ast::Position::new(0, 1, 1),
            regex_syntax::ast::Position::new(0, 1, 1),
        );

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
                                span: poison_span,
                                op: regex_syntax::ast::RepetitionOp {
                                    span: poison_span,
                                    kind: RepetitionKind::ZeroOrMore,
                                },
                                greedy: greedy,
                                ast: e.eval()?,
                            })))
                    }
                    UOp::RepeatOneOrMore(greedy) => {
                        Ok(Box::new(regex_syntax::ast::Ast::Repetition(
                            regex_syntax::ast::Repetition {
                                span: poison_span,
                                op: regex_syntax::ast::RepetitionOp {
                                    span: poison_span,
                                    kind: RepetitionKind::OneOrMore,
                                },
                                greedy: greedy,
                                ast: e.eval()?,
                            })))
                    }
                    UOp::RepeatZeroOrOne(greedy) => {
                        Ok(Box::new(regex_syntax::ast::Ast::Repetition(
                            regex_syntax::ast::Repetition {
                                span: poison_span,
                                op: regex_syntax::ast::RepetitionOp {
                                    span: poison_span,
                                    kind: RepetitionKind::ZeroOrOne,
                                },
                                greedy: greedy,
                                ast: e.eval()?,
                            })))
                    }
                    UOp::RepeatRange(greedy, range) => {
                        Ok(Box::new(regex_syntax::ast::Ast::Repetition(
                            regex_syntax::ast::Repetition {
                                span: poison_span,
                                op: regex_syntax::ast::RepetitionOp {
                                    span: poison_span,
                                    kind: RepetitionKind::Range(range),
                                },
                                greedy: greedy,
                                ast: e.eval()?,
                            })))
                    }
                }
            }
            ExprKind::Capture(e, name) => {
                // TODO(ethan): Think through the semantics a little
                //              more closely here. Should the same expression
                //              always get the same group index, or a new
                //              one every time it is evaluated (as is the
                //              case with this impl). I think this will
                //              be worth revisiting when I look at function
                //              impl.
                let index = env.group_idx;
                env.group_idx += 1;

                Ok(Box::new(regex_syntax::ast::Ast::Group(
                    regex_syntax::ast::Group {
                        span: poison_span,
                        kind: match name {
                            Some(n) => GroupKind::CaptureName(
                                regex_syntax::ast::CaptureName {
                                    span: poison_span,
                                    name: n,
                                    index: index,
                                }
                            ),
                            None => GroupKind::CaptureIndex(index),
                        },
                        ast: e.eval()?,
                    })))
            }
            ExprKind::ExprPoison => panic!("Bug in remake."),
        }
    }
}

struct EvalEnv {
    group_idx: u32,
}
impl EvalEnv {
    fn new() -> Self {
        EvalEnv {
            group_idx: 0
        }
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
