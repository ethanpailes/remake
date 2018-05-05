use std::collections::HashMap;

use regex_syntax;
use regex_syntax::ast::{RepetitionKind, GroupKind};

use error::{InternalError, ErrorKind};
use operators;
use operators::noncapturing_group;
use util::POISON_SPAN;

#[derive(Debug)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

type Value = Box<regex_syntax::ast::Ast>;

impl Expr {
    pub fn new(kind: ExprKind, span: Span) -> Self {
        Self {
            kind: kind,
            span: span,
        }
    }

    pub fn eval(self) -> Result<Value, InternalError> {
        self.eval_(&mut EvalEnv::new())
    }

    fn eval_(
        self,
        env: &mut EvalEnv,
    ) -> Result<Box<regex_syntax::ast::Ast>, InternalError> {

        match self.kind {
            ExprKind::RegexLiteral(r) => Ok(r),
            ExprKind::BinOp(lhs, op, rhs) => {
                match op {
                    BOp::Concat =>
                        Ok(operators::concat(
                            lhs.eval_(env)?, rhs.eval_(env)?)),
                    BOp::Alt =>
                        Ok(operators::alt(
                            lhs.eval_(env)?, rhs.eval_(env)?))
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
                                ast: Box::new(noncapturing_group(e.eval_(env)?)),
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
                                ast: Box::new(noncapturing_group(e.eval_(env)?)),
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
                                ast: Box::new(noncapturing_group(e.eval_(env)?)),
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
                                ast: Box::new(noncapturing_group(e.eval_(env)?)),
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
                        ast: e.eval_(env)?,
                    })))
            }

            ExprKind::Block(statements, value) => {
                env.push_block_env();
                for s in statements {
                    s.eval(env)?;
                }
                let res = value.eval_(env)?;
                env.pop_block_env();

                Ok(res)
            }

            ExprKind::Var(var) => {
                let span = self.span;
                env.lookup(var)
                   .map_err(|e| InternalError::new(e, span))
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
    block_envs: Vec<HashMap<String, Value>>,
}
impl EvalEnv {
    fn new() -> Self {
        EvalEnv {
            block_envs: vec![],
        }
    }

    fn push_block_env(&mut self) {
        self.block_envs.push(HashMap::new());
    }

    fn pop_block_env(&mut self) {
        self.block_envs.pop();
    }

    fn bind(&mut self, var: String, v: Value) {
        let idx = self.block_envs.len() - 1;
        self.block_envs[idx].insert(var, v);
    }

    fn lookup(&self, var: String) -> Result<Value, ErrorKind> {
        for env in self.block_envs.iter().rev() {
            match env.get(&var) {
                None => {},
                // TODO(ethan): drop the clone
                Some(val) => return Ok(val.clone()),
            }
        }

        Err(ErrorKind::NameError { name: var })
    }
}

#[derive(Debug)]
pub enum ExprKind {
    BinOp(Box<Expr>, BOp, Box<Expr>),
    UnaryOp(UOp, Box<Expr>),
    Capture(Box<Expr>, Option<String>),
    RegexLiteral(Box<regex_syntax::ast::Ast>),
    Block(Vec<Statement>, Box<Expr>),
    Var(String),

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
pub struct Statement {
    kind: StatementKind,
    span: Span,
}

impl Statement {
    pub fn new(kind: StatementKind, span: Span) -> Self {
        Statement {
            kind: kind,
            span: span,
        }
    }

    fn eval(self, env: &mut EvalEnv) -> Result<(), InternalError> {
        match self.kind {
            StatementKind::LetBinding(id, e) => {
                let v = e.eval_(env)?;
                env.bind(id.clone(), v);
                Ok(())
            }
        }
    }
}

#[derive(Debug)]
pub enum StatementKind {
    LetBinding(String, Box<Expr>),
}

#[derive(Debug)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}
