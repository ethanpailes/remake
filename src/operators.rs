// Copyright 2018 Ethan Pailes.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

///
/// Utilities for manipulating regex ASTs
///
use regex_syntax::ast::{Alternation, Ast, Concat, Flags, Group, GroupKind};
use util::POISON_SPAN;

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

        (Ast::Concat(ref lconcat), ref r) => Box::new(Ast::Concat(Concat {
            span: lconcat.span.with_end(r.span().end),
            asts: {
                let mut v = lconcat.asts.clone();
                v.push(r.clone());
                v
            },
        })),

        (ref l, Ast::Concat(ref rconcat)) => Box::new(Ast::Concat(Concat {
            span: l.span().with_end(rconcat.span.end),
            asts: {
                let mut v = vec![l.clone()];
                v.extend(rconcat.asts.clone());
                v
            },
        })),

        (l, r) => {
            // The regex crate has not public dependency on regex-syntax,
            // which means that we have no way to just pass an AST to
            // the regex engine for compilation. One of the issues with this
            // is that we will lose any structure in the AST which does not
            // have a textual representation. In particular:
            //
            //     concat( 'foo', alt('a', 'bar') )
            //
            // will pretty print as:
            //
            //     fooa|bar
            //
            // then parse as:
            //
            //     alt( 'fooa', 'bar' )
            //
            // which is obviously wrong. The solution is to inject a bunch
            // of non-capturing groups. Sad bois.
            Box::new(Ast::Concat(Concat {
                span: l.span().with_end(r.span().end),
                asts: vec![
                    noncapturing_group(Box::new(l)),
                    noncapturing_group(Box::new(r)),
                ],
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
                },
            }))
        }

        (ref l, Ast::Alternation(ref rconcat)) => {
            Box::new(Ast::Alternation(Alternation {
                span: l.span().with_end(rconcat.span.end),
                asts: {
                    let mut v = vec![l.clone()];
                    v.extend(rconcat.asts.clone());
                    v
                },
            }))
        }

        (l, r) => {
            // There is always a textual indicator of an alternation, so
            // we don't need to wrap the expressions in a non-capturing
            // group.
            Box::new(Ast::Alternation(Alternation {
                span: l.span().with_end(r.span().end),
                asts: vec![l, r],
            }))
        }
    }
}

pub fn noncapturing_group(ast: Box<Ast>) -> Ast {
    Ast::Group(Group {
        span: POISON_SPAN,
        kind: GroupKind::NonCapturing(Flags {
            span: POISON_SPAN,
            items: vec![],
        }),
        ast: ast,
    })
}
