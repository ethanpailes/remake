/*!
This crate provides a library for writing maintainable regular expressions.
When regex are small, their terse syntax is nice because it allows you
to say a lot with very little. In some cases regex outgrow the basic
regex syntax. The [regex crate][regexcrate] provides an
[extended mode][emode] which allows users to write regex containing
insignificant whitespace and add comments to their regex. This crate
takes that idea a step further by allowing you to factor your regex
into different named pieces, then recombine the pieces later using
familiar syntax.

The actual rust API surface of this crate is intentionally small.
Most of the functionality provided comes as part of the remake
language.

# Example: Regular Expression Literals

Remake is all about manipulating regular expressions, so regex
literals are one of the key features. In remake, we denote a
regex literal by bracketing a regex in slashes. This is similar
to the approach taken by languages like javascript.

```rust
# fn ex_main() -> Result<(), remake::Error> {
use remake::Remake;
let re = Remake::compile(r" /foo|bar/ ")?;
assert!(re.is_match("foo"));
# Ok(())
# }
```

A common issue when writing a regex is not knowing if a particular
sigil has special meaning in a regex. Even if you know that '+' is
special, it can be easy to forget to escape it as, especially
as your regex grows in complexity. To help with these situations,
remake provides a second type of regex literals using single quotes.
In a single-quote regex literal, there are no special characters.

```rust
# fn ex_main() -> Result<(), remake::Error> {
use remake::Remake;
let re = Remake::compile(r" 'foo|bar' ")?;
assert!(!re.is_match("foo"));
assert!(re.is_match("foo|bar"));
# Ok(())
# }
```

# Combining Regex

The ability to pull regex apart into separate literals is not that
useful without the ability to put them back together. Remake provides
a number of operators for combining regex, corresponding very closely to
ordinary regex operators.

## Example: Composition

We use the `.` char to indicate regex composition, also known as regex
concatenation. There is no syntax for composition in ordinary regex, instead
expressions written next to each other are simply composed automatically.
In remake, we are more explicit.

```rust
# fn ex_main() -> Result<(), remake::Error> {
use remake::Remake;
let re = Remake::compile(r" 'f+oo' . /a*b/ ")?;
assert!(re.is_match("f+oob"));
assert!(re.is_match("f+ooaaaaaaaaaaaaaaab"));
# Ok(())
# }
```

## Example: Choice

Just like in standard regex syntax, we use a pipe char to indicate
choice between two different remake expressions.

```rust
# fn ex_main() -> Result<(), remake::Error> {
use remake::Remake;
let re = Remake::compile(r" 'foo' | 'bar' ")?;
assert!(re.is_match("foo"));
# Ok(())
# }
```

## Example: Kleene Star

Just like in standard regex syntax, we use a unary postfix `*` char to
ask for zero or more repetitions of an expression.

```rust
# fn ex_main() -> Result<(), remake::Error> {
use remake::Remake;
let re = Remake::compile(r" 'a'* 'b' ")?;
assert!(re.is_match("foo"));
# Ok(())
# }
```


TODO(ethan): show off error messages in the examples

[emode]: https://github.com/rust-lang/regex#usage
[regexcrate]: https://github.com/rust-lang/regex
*/

// TODO: add a usage section once this is on crates.io and I can actually
//       explain how to pull it into a project.

extern crate regex_syntax;
extern crate regex;
extern crate lalrpop_util;
extern crate failure;

mod ast;
mod parse;
mod error;
mod operators;
mod util;

//
// I want the eventual interface to look something like:
//
// ```
// let r = Remake::new(r"/foo/")?;
// let wrap_parens = Remake::new(r"fn (re) { '(' + . re . ')' }")?;
// let re: Regex = wrap_parens.apply(r)?.eval()?;
// ```
//
// The idea is that remake expressions can be parsed and then passed
// around within rust as opaque expressions. They can then be combined
// through function application.
//
// This design would work well with an ML-style all-functions-have-
// one-argument sort of approach, but I want remake to semantically
// be basically dynamically typed rust with a strong focus on regular
// expression manipulation. A possible solution is some light trait
// abuse where we define a sealed (using the sealed trait pattern)
// RemakeArgument trait and then impl it for Remake, tuples of RemakeArgument,
// (and Vecs of RemakeArgument, IntoIters of RemakeArgument,
//  and HashMaps of RemakeArgument once corresponding data structures
//  have been added to the language). I think the user would mostly not
// need to know about the RemakeArgument trait.
//

use std::fmt;
use regex::Regex;
use error::InternalError;

/// A remake expression, which can be compiled into a regex.
pub struct Remake {
    /// The parsed remake expression.
    expr: ast::Expr,
    /// The source used to construct this remake expression.
    ///
    /// Required to interpret spans.
    src: String,
}

impl Remake {
    /// Evaluate some remake source to produce a regular expression.
    ///
    /// # Example: A mostly-wrong URI validator
    /// ```rust
    /// # use remake::Remake;
    /// # fn ex_main() -> Result<(), remake::Error> {
    /// let web_uri_re = Remake::compile(r#"
    /// let scheme = /https?:/ . '//';
    /// let domain = /[\w\.-_]/;
    /// let query_body = (/[\w.-_?]/ | '/')*;
    /// let frag_body = query;
    /// scheme . domain . ('?' . query_body)? . ('#' . frag_body)?
    /// "#)?;
    /// assert!(web_uri_re.is_match("https://www.rust-lang.org"));
    /// assert!(web_uri_re.is_match("https://github.com/ethanpailes/remake"));
    /// # Ok(())
    /// # }
    /// ```
    pub fn compile(src: &str) -> Result<Regex, Error> {
        Self::new(String::from(src))?.eval()
    }

    /// Construct a remake expression which can be evaluated
    /// at a later time.
    ///
    /// To go directly from a remake expression to a `Regex`,
    /// see `Remake::compile`.
    ///
    /// # Example:
    /// ```rust
    /// # use remake::Remake;
    /// # fn ex_main() -> Result<(), remake::Error> {
    /// let remake_expr = Remake::new(r" 'a literal' ".to_string())?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(src: String) -> Result<Remake, Error> {
        let mut remake = Remake {
            expr: ast::Expr::new(ast::ExprKind::ExprPoison,
                                 ast::Span { start: 0, end: 0 }),
            src: src,
        };

        remake.expr = match parse::BlockBodyParser::new().parse(&remake.src) {
            Ok(expr) => expr,
            Err(err) => {
                return Err(Error::ParseError(match err {
                    lalrpop_util::ParseError::User { error } =>
                        format!("{}", error.overlay(&remake.src)),

                    lalrpop_util::ParseError::InvalidToken { location } =>
                        format!("{}",
                            InternalError::new(
                                error::ErrorKind::InvalidToken,
                                ast::Span { start: location, end: location + 1 }
                                ).overlay(&remake.src)),

                    lalrpop_util::ParseError::UnrecognizedToken {
                        token: Some((l, tok, r)), expected
                    } => format!(
                        "{}",
                        InternalError::new(
                            error::ErrorKind::UnrecognizedToken {
                                token: tok.to_string(),
                                expected: expected,
                            },
                            ast::Span { start: l, end: r }
                            ).overlay(&remake.src)),

                    err => format!("{}", err),
                }));
            }
        };

        Ok(remake)
    }

    /// Evaluate a remake expression.
    ///
    /// # Example:
    /// ```rust
    /// # use remake::Remake;
    /// # fn ex_main() -> Result<(), remake::Error> {
    /// let remake_expr = Remake::new(r" 'a [li[teral' ".to_string())?;
    /// let re = remake_expr.eval()?;
    ///
    /// assert!(re.is_match("a [li[teral"));
    /// # Ok(())
    /// # }
    /// ```
    pub fn eval(self) -> Result<Regex, Error> {
        match self.expr.eval() {
            Ok(ast) => Ok(Regex::new(&ast.to_string()).unwrap()),
            Err(err) => Err(Error::RuntimeError(
                            format!("{}", err.overlay(&self.src)))),
        }
    }
}

#[derive(Clone)]
pub enum Error {
    /// A parse error occurred.
    ParseError(String),

    /// A runtime error occurred.
    RuntimeError(String),
}

impl failure::Fail for Error {}

// The debug formatter already provides a user-facing error so
// that .unwrap() will result in quick feedback.
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use Error::*;

        match self {
            &ParseError(ref err) => {
                writeln!(f, "\nremake parse error:")?;
                writeln!(f, "{}", err)?;
            }

            &RuntimeError(ref err) => {
                writeln!(f, "\nremake evaluation error:")?;
                writeln!(f, "{}", err)?;
            }
        }

        Ok(())
    }
}

// TODO(ethan): Cleanup and going over error messages to make sure that they
//              are useful.
//              - Code coverage
//              - Refactor tests
//              - Refactor evaluation into its own module
// TODO(ethan): Comment support.
// TODO(ethan): Docs.
//              - Rustdocs
//              - README.md
//                  - Badges
//                  - Point to documentation
//                  - Point to crates.io (chicken/egg here)
// TODO(ethan): Run rustfmt (how do I avoid mangling multi-line strings?)
// TODO(ethan): Release!!! (don't talk about it until we have lambdas though).

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! mat {
        ($test_name:ident, $remake_src:expr, $input:expr) => {
            #[test]
            fn $test_name() {
                let re = Remake::compile($remake_src).unwrap();
                assert!(re.is_match($input),
                    format!("/{:?}/ does not match {:?}.", re, $input));
            }
        }
    }

    macro_rules! captures {
        ($test_name:ident, $remake_src:expr, $input:expr, $( $group:expr ),*) => {
            #[test]
            fn $test_name() {
                let re = Remake::compile($remake_src)
                            .expect("The regex to compile.");
                let expected = vec![$($group),*];
                let actual = re.captures($input)
                               .expect("The regex to match.")
                               .iter()
                               .map(|mat| mat.map(|g| g.as_str()))
                               .collect::<Vec<_>>();

                assert_eq!(&expected[..], &actual[1..],
                    "Captures did not match. re={:?}", re);
            }
        }
    }

    macro_rules! captures_named {
        ($test_name:ident, $remake_src:expr, $input:expr, $( $group:expr ),*) => {
            #[test]
            fn $test_name() {
                use std::str::FromStr;

                let re = Remake::compile($remake_src)
                            .expect("The regex to compile.");
                let expected_caps = vec![$($group),*];
                let num_name_re =
                    Remake::compile("'_' . cap /[0-9]+/").unwrap();

                let caps = re.captures($input).expect("The regex to match.");

                for &(name, result) in expected_caps.iter() {
                    let cap = match num_name_re.captures(name) {
                        Some(c) =>
                            caps.get(usize::from_str(&c[1]).unwrap())
                                .map(|m| m.as_str()),
                        None => caps.name(name).map(|m| m.as_str()),
                    };

                    assert_eq!(cap, result, "Group '{}' did not match", name);
                }
            }
        }
    }

    macro_rules! no_mat {
        ($test_name:ident, $remake_src:expr, $input:expr) => {
            #[test]
            fn $test_name() {
                let re = Remake::compile($remake_src).unwrap();
                assert!(!re.is_match($input),
                    format!("/{:?}/ matches {:?}.", re, $input));
            }
        }
    }

    macro_rules! parse_error {
        ($test_name:ident, $remake_src:expr, $expected_err_str:expr) => {
            #[test]
            fn $test_name() {
                let result = Remake::compile($remake_src);
                match &result {
                    &Ok(_) => panic!("Should not eval to anything."),
                    &Err(Error::ParseError(ref reason)) => {
                        // When copying the right output into the test
                        // case uncommenting this can help debug.
                        //
                        // assert_eq!($expected_err_str, reason);
                        
                        if $expected_err_str != reason {
                            // We unwrap rather than asserting or something
                            // a little more reasonable so that we can see
                            // the error output as the user sees it.
                            result.clone().unwrap();
                        }
                    }
                    _ => panic!("Should not parse."),
                }
            }
        }
    }

    macro_rules! parse_error_pre {
        ($test_name:ident, $remake_src:expr, $expected_err_str:expr) => {
            #[test]
            fn $test_name() {
                let result = Remake::compile($remake_src);
                match &result {
                    &Ok(_) => panic!("Should not eval to anything."),
                    &Err(Error::ParseError(ref reason)) => {
                        let reason = &reason[0..$expected_err_str.len()];
                        // When copying the right output into the test
                        // case uncommenting this can help debug.
                        //
                        // assert_eq!($expected_err_str, reason);
                        
                        if $expected_err_str != reason {
                            // We unwrap rather than asserting or something
                            // a little more reasonable so that we can see
                            // the error output as the user sees it.
                            result.clone().unwrap();
                        }
                    }
                    _ => panic!("Should not parse."),
                }
            }
        }
    }

    macro_rules! runtime_error_pre {
        ($test_name:ident, $remake_src:expr, $expected_err_str:expr) => {
            #[test]
            fn $test_name() {
                let result = Remake::compile($remake_src);
                match &result {
                    &Ok(_) => panic!("Should not eval to anything."),
                    &Err(Error::ParseError(_)) => panic!("Should parse."),
                    &Err(Error::RuntimeError(ref reason)) => {
                        let reason = &reason[0..$expected_err_str.len()];
                        // When copying the right output into the test
                        // case uncommenting this can help debug.
                        //
                        // assert_eq!($expected_err_str, reason);

                        if $expected_err_str != reason {
                            // We unwrap rather than asserting or something
                            // a little more reasonable so that we can see
                            // the error output as the user sees it.
                            result.clone().unwrap();
                        }
                    }
                }
            }
        }
    }

    //
    // remake literals
    //

    mat!(lit_1, r"/a/", "a");
    mat!(lit_2, r"'a'", "a");
    mat!(lit_3, r"/\p{Currency_Symbol}/", r"$");
    mat!(lit_4, r"'\p{Currency_Symbol}'", r"\p{Currency_Symbol}");
    mat!(lit_5, r"'\u{Currency_Symbol}'", r"\u{Currency_Symbol}");

    //
    // remake parse errors
    //
    
    parse_error_pre!(unmatched_tick_1_, r"'a",
        r#"    at line 1, col 1:
    0001 > 'a
           ^
Invalid token."#);

    parse_error_pre!(unmatched_tick_2_, r"a'",
        r#"    at line 1, col 2:
    0001 > a'
            ^
Invalid token."#);

    parse_error!(unmatched_slash_1_, r"/a",
        r#"    at line 1, col 1:
    0001 > /a
           ^
Invalid token.
"#);

    parse_error_pre!(unmatched_slash_2_, r"a/",
        r#"    at line 1, col 2:
    0001 > a/
            ^
Invalid token."#);

    parse_error!(unmatched_tick_slash_1_, r"'a/",
        r#"    at line 1, col 1:
    0001 > 'a/
           ^
Invalid token.
"#);

    parse_error!(unmatched_tick_slash_2_, r"/a'",
        r#"    at line 1, col 1:
    0001 > /a'
           ^
Invalid token.
"#);

    //
    // parse errors that bubble up from the regex crate
    //
    
    parse_error!(re_parse_err_1_, r"/a[/", 
        r#"    at line 1, col 1:
    0001 > /a[/
           ^^^^
Error parsing the regex literal: /a[/
    regex parse error:
        a[
         ^
    error: unclosed character class
"#);

    parse_error!(re_parse_err_2_, r"/a[]/",
        r#"    at line 1, col 1:
    0001 > /a[]/
           ^^^^^
Error parsing the regex literal: /a[]/
    regex parse error:
        a[]
         ^^
    error: unclosed character class
"#);

    parse_error!(re_multiline_parse_err_1_,
        r#"

            /a[/

        "#,
        r#"    at line 3, col 13:
    0002 > 
    0003 >             /a[/
                       ^^^^
    0004 > 
Error parsing the regex literal: /a[/
    regex parse error:
        a[
         ^
    error: unclosed character class
"#);

    parse_error_pre!(unrecognized_token_1_, r"/foo/ /foo/",
        r#"    at line 1, col 7:
    0001 > /foo/ /foo/
                 ^^^^^
Unexpected token '/foo/'. Expected one of:"#);

    parse_error_pre!(binary_plus_1_, r"/foo/ + /bar/",
        r#"    at line 1, col 9:
    0001 > /foo/ + /bar/
                   ^^^^^
Unexpected token '/bar/'. Expected one of:"#);

    parse_error_pre!(binary_plus_multiline_1_, r#"/foo/ +
        /bar
        /"#, r#"    starting at line 2, col 9 and ending at line 3, col 10:
    0001  > /foo/ +
    0002  >         /bar
    start >         ^
    0003  >         /
    end   >         ^
Unexpected token '/bar
        /'. Expected one of:"#);

    parse_error_pre!(binary_plus_multiline_2_, r#"
        /foo/ + /
        
        bar



        /"#, r#"    starting at line 2, col 17 and ending at line 8, col 10:
    0001  > 
    0002  >         /foo/ + /
    start >                 ^
    0003  >         
    ...
    0007  > 
    0008  >         /
    end   >         ^
Unexpected token"#);

    //
    // Remake expressions.
    //

    mat!(concat_1_, r"/foo/ . 'bar'", "foobar");
    mat!(concat_2_, r"/foo/ . 'b[r'", "foob[r");
    mat!(alt_1_, r"/foo/ | /bar/", "foo");

    mat!(greedy_star_1_, r"/a/ *", "aaaaaaa");
    mat!(greedy_star_2_, r"/a/ * . 'b'", "aaaaaaab");
    mat!(greedy_star_3_, r"/a/ * . 'b'", "b");

    mat!(lazy_star_1_, r"/a/ *?", "aaaaaaa");
    mat!(lazy_star_2_, r"/a/ *? . 'b'", "aaaaaaab");
    mat!(lazy_star_3_, r"/a/ *? . 'b'", "b");

    mat!(greedy_plus_1_, r"/a/ +", "aaaaaaa");
    mat!(greedy_plus_2_, r"/a/ + . 'b'", "aaaaaaab");
    no_mat!(greedy_plus_3_, r"/a/ + . 'b'", "b");

    mat!(lazy_plus_1_, r"/a/ +?", "aaaaaaa");
    mat!(lazy_plus_2_, r"/a/ +? . 'b'", "aaaaaaab");
    no_mat!(lazy_plus_3_, r"/a/ +? . 'b'", "b");

    mat!(greedy_question_1_, r"/a/ ?", "a");
    mat!(greedy_question_2_, r"/a/ ? . 'b'", "ab");
    mat!(greedy_question_3_, r"/a/ ? . 'b'", "b");

    mat!(lazy_question_1_, r"/a/ ??", "a");
    mat!(lazy_question_2_, r"/a/ ?? . 'b'", "ab");
    mat!(lazy_question_3_, r"/a/ ?? . 'b'", "b");

    mat!(greedy_exact_1_, r"/a/ {3}", "aaa");
    mat!(greedy_exact_2_, r"/a/ { 3} . 'b'", "aaab");

    // lazyness does not matter for an exact repetition
    no_mat!(lazy_exact_1_, r"/a/ {3}?", "a");
    no_mat!(lazy_exact_2_, r"/a/ {3 }? . 'b'", "ab");
    mat!(lazy_exact_3_, r"/a/ {3}?", "aaa");
    mat!(lazy_exact_4_, r"/a/ {3 }? . 'b'", "aaab");


    mat!(greedy_atleast_1_, r"/a/ {3,}", "aaaaaaaa");
    mat!(greedy_atleast_2_, r"/a/ { 3 , } . 'b'", "aaaaaab");
    no_mat!(greedy_atleast_3_, r"/a/ { 3 , } . 'b'", "ab");

    mat!(lazy_atleast_1_, r"/a/ {3,}?", "aaaaaaaa");
    mat!(lazy_atleast_2_, r"/a/ { 3 , }? . 'b'", "aaaaaab");
    no_mat!(lazy_atleast_3_, r"/a/ { 3 , }? . 'b'", "ab");

    captures!(cap_basic_1_, r"/a(b)a/", "aba",
        Some("b"));

    captures!(cap_remake_1_, r"/a/ . cap /b/ . /a/", "aba",
        Some("b"));
    captures!(cap_remake_2_, r"cap (/a/ . cap /b/ . /a/)", "aba",
        Some("aba"), Some("b"));

    captures!(cap_remake_mixed_1_, r"cap (/a/ . /(b)/ . /a/)", "aba",
        Some("aba"), Some("b"));
    captures!(cap_remake_mixed_2_, r"/(a)/ . cap (/a/ . /(b)/ . /a/)", "aaba",
        Some("a"), Some("aba"), Some("b"));
    captures!(cap_remake_mixed_3_, r"cap /blah/ . (/(a)/ | (/b/ . cap /c/))", 
        "blahbc",
        Some("blah"), None, Some("c"));
    captures!(cap_remake_mixed_4_, r"/(a)(b)(c)/ . cap /blah/", "abcblah",
        Some("a"), Some("b"), Some("c"), Some("blah"));

    captures_named!(cap_remake_named_1_, r"/a/ . cap /b/ as foo . /a/", "aba",
        ("foo", Some("b")));
    captures_named!(cap_remake_named_2_, r"cap cap /foo/ as blah", "foo",
                    ("_1", Some("foo")), ("blah", Some("foo")));

    mat!(block_basic_1_, r"{ /a/ }", "a");

    mat!(block_repeat_1_, r"{ /a/{2} }", "aa");
    no_mat!(block_repeat_2_, r"{ /a/{3,} }", "aa");

    mat!(block_unused_let_1_, r#"{
        let foo = /a/;
        /b/
    }"#, "b");

    mat!(toplevel_unused_let_1_, r#"
        let foo = /a/;
        /b/
    "#, "b");

    mat!(toplevel_let_1_, r#"
        let foo = /a/;
        foo
    "#, "a");

    mat!(toplevel_let_2_, r#"
        let foo = /a/;
        foo . /bar/
    "#, "abar");


    runtime_error_pre!(name_error_1_, r#"
        foo
    "#, r#"    at line 2, col 9:
    0001 > 
    0002 >         foo
                   ^^^
    0003 >     
NameError: unknown variable 'foo'.
"#);

    parse_error!(num_too_long_1_,
        r"'a'{11111111111111111111111111111111111111111111111111111111}",
        r#"    at line 1, col 5:
    0001 > 'a'{11111111111111111111111111111111111111111111111111111111}
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Error parsing 11111111111111111111111111111111111111111111111111111111 as a number. Literal too long.
"#);


    mat!(alt_merge_1_, r"/a|b/ | /c/", "c");
    mat!(alt_merge_2_, r"/c/ | /a|b/", "c");
    mat!(alt_merge_3_, r"/c|d/ | /a|b/", "d");

    mat!(block_1_, r#"
        let gen_delims = ':' | '/' | '?' | '[' | ']' | '@';
        'foo'
    "#,
    "foo");
}
