///
/// Utilities for manipulating regex ASTs
///
use regex_syntax::ast::{Ast, Concat, Alternation};

// TODO(ethan): use poison span?

//
// The fact that regex_syntax::ast::Ast impls drop makes it really hard
// to avoid copying stuff that I don't think I should have to copy. For
// now I'm going to have to just bite the bullet, but eventually I may
// have to write my own regex AST type that does not come with a custom
// Drop. It may also be worthwhile to talk through the issue with
// @burntsushi.
//

pub fn concat(lhs: Box<Ast>, rhs: Box<Ast>) -> Box<Ast> {
    match (*lhs, *rhs) {
        (Ast::Concat(ref lconcat), Ast::Concat(ref rconcat)) => {
            Box::new(Ast::Concat(Concat {
                span: lconcat.span.with_end(rconcat.span.end),
                asts: {
                    let mut v = lconcat.asts.clone();
                    v.extend(rconcat.asts.clone());
                    v
                },
            }))
        }

        (Ast::Concat(ref lconcat), ref r) => {
            Box::new(Ast::Concat(Concat {
                span: lconcat.span.with_end(r.span().end),
                asts: {
                    let mut v = lconcat.asts.clone();
                    v.push(r.clone());
                    v
                }
            }))
        }

        (ref l, Ast::Concat(ref rconcat)) => {
            Box::new(Ast::Concat(Concat {
                span: l.span().with_end(rconcat.span.end),
                asts: {
                    let mut v = vec![l.clone()];
                    v.extend(rconcat.asts.clone());
                    v
                }
            }))
        }

        (l, r) => {
            Box::new(Ast::Concat(Concat {
                span: l.span().with_end(r.span().end),
                asts: vec![l, r],
            }))
        }
    }
}

pub fn alt(lhs: Box<Ast>, rhs: Box<Ast>) -> Box<Ast> {
    match (*lhs, *rhs) {
        (Ast::Alternation(ref lconcat), Ast::Alternation(ref rconcat)) => {
            Box::new(Ast::Alternation(Alternation {
                span: lconcat.span.with_end(rconcat.span.end),
                asts: {
                    let mut v = lconcat.asts.clone();
                    v.extend(rconcat.asts.clone());
                    v
                },
            }))
        }

        (Ast::Alternation(ref lconcat), ref r) => {
            Box::new(Ast::Alternation(Alternation {
                span: lconcat.span.with_end(r.span().end),
                asts: {
                    let mut v = lconcat.asts.clone();
                    v.push(r.clone());
                    v
                }
            }))
        }

        (ref l, Ast::Alternation(ref rconcat)) => {
            Box::new(Ast::Alternation(Alternation {
                span: l.span().with_end(rconcat.span.end),
                asts: {
                    let mut v = vec![l.clone()];
                    v.extend(rconcat.asts.clone());
                    v
                }
            }))
        }

        (l, r) => {
            Box::new(Ast::Alternation(Alternation {
                span: l.span().with_end(r.span().end),
                asts: vec![l, r],
            }))
        }
    }
}


trait Fuse {
    fn fuse(self, rhs: Self) -> Self;
}
