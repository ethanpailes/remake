extern crate regex_syntax;
extern crate regex;
extern crate lalrpop_util;

mod ast;
mod parse;

use regex::Regex;

pub fn remake(remake_src: &str) -> Result<Regex, Error> {
    let expr = match parse::ExprParser::new().parse(remake_src) {
        Ok(expr) => expr,
        Err(err) => return Err(Error::ParseError(format!("{}", err))),
    };

    eval(&*expr)
}

fn eval(expr: &ast::Expr) -> Result<Regex, Error> {
    match expr {
        &ast::Expr::RegexLiteral(ref regex_ast) =>
            Ok(Regex::new(&format!("{}", regex_ast)).unwrap()),
    }
}

// TODO(ethan): use failure?
#[derive(Debug)]
pub enum Error {
    /// A parse error is just parameterized by a string because
    /// zero code is going to be both smart enough to correct the
    /// issue and dumb that it can't parse the human readable error.
    ParseError(String),
}

// TODO(ethan): impl Display for Error
// TODO(ethan): Write all the basic combinators so that we can get an
//              expression language going here.
//              - It is probably worth making a POISON_SPAN the result
//                of regex ASTs that are built up with remake concatination
//                or alternation or whatever.
// TODO(ethan): Write let expressions.
// TODO(ethan): Cleanup and going over error messages to make sure that they
//              are useful.
// TODO(ethan): Docs.
// TODO(ethan): Release!!! (don't talk about it until we have lambdas though).

#[cfg(test)]
mod tests {
    use super::remake;

    macro_rules! mat {
        ($test_name:ident, $remake_src:expr, $input:expr) => {
            #[test]
            fn $test_name() {
                let re = remake($remake_src).unwrap();
                assert!(re.is_match($input),
                    format!("/{:?}/ does not match {:?}.", re, $input));
            }
        }
    }

    macro_rules! noparse {
        ($test_name:ident, $remake_src:expr) => {
            #[test]
            fn $test_name() {
                assert!(!remake($remake_src).ok().is_some(),
                        format!("{:?} parses.", $remake_src));
            }
        }
    }

    mat!(lit_1, r"/a/", "a");
    mat!(lit_2, r"'a'", "a");
    mat!(lit_3, r"/\p{Currency_Symbol}/", r"$");
    mat!(lit_4, r"'\p{Currency_Symbol}'", r"\p{Currency_Symbol}");
    mat!(lit_5, r"'\u{Currency_Symbol}'", r"\u{Currency_Symbol}");

    noparse!(noparse_1, "a");
    noparse!(noparse_2, "'a");
    noparse!(noparse_3, "a/");
    noparse!(noparse_4, "'a/");
}
