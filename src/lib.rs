/*!
This crate provides a library for writing maintainable regular expressions.
When regex are small, their terse syntax is nice because it allows you
to say a lot with very little. As regex grow in size, this terse syntax
quickly becomes a liability, interfering with maintainability.
The [regex crate][regexcrate] provides an
[extended mode][emode] which allows users to write regex containing
insignificant whitespace and add comments to their regex. This crate
takes that idea a step further by allowing you to factor your regex
into different named pieces, then recombine the pieces later using
familiar syntax.

The actual rust API surface of this crate is intentionally small.
Most of the functionality provided comes as part of the remake
language.

# Example: A mostly-wrong URI validator
```rust
use remake::Remake;
let web_uri_re = Remake::compile(r#"
    let scheme = /https?:/ . '//';
    let auth = /[\w\.\-_]+/;
    let path = ('/' . /[\w\-_]+/)*;
    let query_body = (/[\w\.\-_?]/ | '/')*;
    let frag_body = cap query_body as frag;

      /^/
    . scheme . auth . path
    . ('?' . query_body)?
    . ('#' . frag_body)?
    . /$/
    "#).unwrap();

assert!(web_uri_re.is_match("https://www.rust-lang.org"));
assert!(web_uri_re.is_match("https://github.com/ethanpailes/remake"));

assert_eq!(
    web_uri_re
        .captures("https://tools.ietf.org/html/rfc3986#section-1.1.3").unwrap()
        .name("frag").unwrap().as_str(),
        "section-1.1.3");
```

# Regular Expression Literals

Remake is all about manipulating regular expressions, so regex
literals are one of the key features. In remake, we denote a
regex literal by bracketing a regex in slashes. This is similar
to the approach taken by languages like javascript.

```rust
use remake::Remake;
let re = Remake::compile(r" /foo|bar/ ").unwrap();

assert!(re.is_match("foo"));
```

If you want to include a forward slash in a regex literal, you will
have to escape it with a backslash.

```rust
use remake::Remake;
let re = Remake::compile(r" /a regex with a \/ slash/ ").unwrap();

assert!(re.is_match("a regex with a / slash"));
```

A common issue when writing a regex is not knowing if a particular
sigil has special meaning in a regex. Even if you know that '+' is
special, it can be easy to forget to escape it, especially
as your regex grows in complexity. To help with these situations,
remake provides a second type of regex literal using single quotes.
In a single-quote regex literal, there are no special characters.

```rust
use remake::Remake;
let re = Remake::compile(r" 'foo|bar' ").unwrap();

assert!(!re.is_match("foo"));
assert!(re.is_match("foo|bar"));
```

# Combining Regex

The ability to pull regex apart into separate literals is not that
useful without the ability to put them back together. Remake provides
a number of operators for combining regex, corresponding very closely to
ordinary regex operators.

## Composition

We use the `.` operator to indicate regex composition, also known as regex
concatenation. There is no syntax for composition in ordinary regex, instead
expressions written next to each other are simply composed automatically.
In remake, we are more explicit.

```rust
use remake::Remake;
let re = Remake::compile(r" 'f+oo' . /a*b/ ").unwrap();

assert!(re.is_match("f+oob"));
assert!(re.is_match("f+ooaaaaaaaaaaaaaaab"));
```

## Choice

Just like in standard regex syntax, we use the pipe operator to indicate
choice between two different remake expressions.

```rust
use remake::Remake;
let re = Remake::compile(r" 'foo' | 'bar' ").unwrap();

assert!(re.is_match("foo"));
```

## Kleene Star

Just like in standard regex syntax, we use a unary postfix `*` operator to
ask for zero or more repetitions of an expression.

```rust
use remake::Remake;
let re = Remake::compile(r" 'a'* . 'b' ").unwrap();

assert!(re.is_match("aaaaab"));
assert!(re.is_match("b"));
```

remake supports lazy Kleene star as well

```rust
use remake::Remake;
let re = Remake::compile(r" 'a'*? ").unwrap();

assert_eq!(re.find("aaaaa").unwrap().as_str(), "");
```

## Kleene Plus

As you might expect, remake also has syntax for repeating
an expression one or more times.

```rust
use remake::Remake;
let re = Remake::compile(r" 'a'+ . 'b' ").unwrap();
assert!(re.is_match("aaaaab"));
assert!(!re.is_match("b"));
```

and there is a lazy variant

```rust
use remake::Remake;
let re = Remake::compile(r" 'a'+? ").unwrap();

assert_eq!(re.find("aaaaa").unwrap().as_str(), "a");
```

## Counted Repetition

[regex][regexcrate] supports a couple of different ways to
ask for a counted repetition. Remake supports them all.

```rust
use remake::Remake;
let re_a_1 = Remake::compile(r" 'a'{1} ").unwrap();
let re_a_1ormore = Remake::compile(r" 'a'{1,} ").unwrap();
let re_between_2_and_5 = Remake::compile(r" 'a'{2,5} ").unwrap();

assert!(re_a_1.is_match("a"));
assert!(re_a_1ormore.is_match("aaaaaa"));
assert!(re_between_2_and_5.is_match("aaaaa"));

let re_a_1_lazy = Remake::compile(r" 'a'{1}? ").unwrap();
let re_a_1ormore_lazy = Remake::compile(r" 'a'{1,}? ").unwrap();
let re_between_2_and_5_lazy = Remake::compile(r" 'a'{2,5}? ").unwrap();

assert!(re_a_1_lazy.is_match("a"));
assert!(re_a_1ormore_lazy.is_match("aaaaaa"));
assert!(re_between_2_and_5_lazy.is_match("aaaaa"));
```

## Capture Groups

Regex can be annotated with parentheses to ask the engine to
take note of where particular sub-expressions occur in a match.
You can access these sub-expressions by invoking the
[`Regex::captures`](struct.Regex.html#method.captures)
method. Remake already uses parentheses to control precedence, so
it would be confusing to also use them as capturing syntax. Instead,
we introduce the `cap` keyword. You can ask for an unnamed capture
group by writing `cap <expr>`

```rust
use remake::Remake;
let re = Remake::compile(r" 'a' . cap 'b' . 'c' ").unwrap();

assert_eq!(&re.captures("abc").unwrap()[1], "b");
```

or give it a name with `cap <expr> as <name>`

```rust
use remake::Remake;
let re = Remake::compile(r" 'a' . cap 'b' as group . 'c' ").unwrap();
assert_eq!(re.captures("abc").unwrap().name("group").unwrap().as_str(), "b");
```

capture groups compose well with capture groups defined in a regex
literal

```rust
use remake::Remake;
let re = Remake::compile(r" /foo(bar)/ . cap 'baz' ").unwrap();

let caps = re.captures("foobarbaz").unwrap();
assert_eq!(&caps[1], "bar");
assert_eq!(&caps[2], "baz");
```

Note that the index of unnamed capture groups depends on their
order in the final expression that is generated, not where they
are defined in a particular piece of remake source. The process of
index assignment is not particularly implicit, so it is good practice
to use named capture groups as regex grow large.

# Block Expressions

Just like rust, remake supports block expressions to introduce
new scopes. A block starts with an open curly brace (`{`), contains
zero or more statements followed by an expression, and ends with
a closing curly brace (`}`).

```rust
use remake::Remake;
let re = Remake::compile(r#"
    {
        let foo_re = 'foo';
        foo_re | /bar/
    }
    "#).unwrap();

assert!(re.is_match("bar"));
```

For convenience, the top level of a piece of remake source is
automatically treated as the inside of a block.

```rust
use remake::Remake;
let re = Remake::compile(r#"
    let foo_re = 'foo';
    foo_re | /bar/
    "#).unwrap();

assert!(re.is_match("bar"));
```

# Let Statements

As shown above, the `let` keyword can be used to bind expressions
to a name in the current scope. 

# Error Messages

Nice error messages are key to developer productivity, so remake
endeavors to provide human friendly error messages. Remake error
messages implement a `Debug` instance written with the expectation
that the most common way to use remake is
`Remake::compile(...).unwrap()`. This means that the error messages
automatically show up with nice formatting in the middle of the
output from `rustc`. Good error messages are important, so if you
find one confusing and have an idea about how to improve it, please
[share your idea](https://github.com/ethanpailes/remake/issues).

## Example: A Bad Literal

The code

```rust,should_panic
use remake::Remake;
let _re = Remake::compile(r" /unclosed literal ").unwrap();
```

will fail with the error message

```text
thread 'main' panicked at 'called `Result::unwrap()` on an `Err` value:
remake parse error:
    at line 1, col 2:
    0001 >  /unclosed literal
            ^
Invalid token.
```

# Comments

remake has C-style comments like rust

```rust
use remake::Remake;
let re = Remake::compile(r#"
    // this is a line comment
    let foo_re = 'foo';
    /* and this is a 
     * block comment
     */
    foo_re | /bar/
    /* just like /* in rust */ block comments can be nested */
    "#).unwrap();

assert!(re.is_match("bar"));
```

[emode]: https://github.com/rust-lang/regex#usage
[regexcrate]: https://github.com/rust-lang/regex
*/

extern crate regex_syntax;
pub extern crate regex;
extern crate lalrpop_util;
extern crate failure;

mod ast;
mod lex;
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
use error::InternalError;

/// A remake expression, which can be compiled into a regex.
#[derive(Clone)]
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
    /// use remake::Remake;
    /// let web_uri_re = Remake::compile(r#"
    ///     let scheme = /https?:/ . '//';
    ///     let auth = /[\w\.\-_]+/;
    ///     let path = ('/' . /[\w\-_]+/)*;
    ///     let query_body = (/[\w\.\-_?]/ | '/')*;
    ///     let frag_body = cap query_body as frag;
    ///
    ///       /^/
    ///     . scheme . auth . path
    ///     . ('?' . query_body)?
    ///     . ('#' . frag_body)?
    ///     . /$/
    ///     "#).unwrap();
    ///
    /// assert!(web_uri_re.is_match("https://www.rust-lang.org"));
    /// assert!(web_uri_re.is_match("https://github.com/ethanpailes/remake"));
    ///
    /// assert_eq!(
    ///     web_uri_re
    ///         .captures("https://tools.ietf.org/html/rfc3986#section-1.1.3").unwrap()
    ///         .name("frag").unwrap().as_str(),
    ///         "section-1.1.3");
    /// ```
    ///
    /// # Errors:
    ///
    /// Calling `compile` on some remake source might result in an
    /// [`Error`](enum.Error.html).
    pub fn compile(src: &str) -> Result<regex::Regex, Error> {
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
    /// # fn main() { ex_main().unwrap() }
    /// ```
    ///
    /// # Errors:
    ///
    /// Calling `new` on some remake source might result in a
    /// [`Error::ParseError`](enum.Error.html).
    pub fn new(src: String) -> Result<Remake, Error> {
        let mut remake = Remake {
            expr: ast::Expr::new(ast::ExprKind::ExprPoison,
                                 ast::Span { start: 0, end: 0 }),
            src: src,
        };

        {
            let parser = parse::BlockBodyParser::new();
            let lexer = lex::Lexer::new(&remake.src);
            remake.expr = match parser.parse(lexer) {
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
        }

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
    /// # fn main() { ex_main().unwrap() }
    /// ```
    ///
    /// # Errors:
    ///
    /// Calling `eval` on a remake expression might result in a
    /// [`Error::RuntimeError`](enum.Error.html).
    pub fn eval(self) -> Result<regex::Regex, Error> {
        match self.expr.eval() {
            Ok(ast) => Ok(regex::Regex::new(&ast.to_string()).unwrap()),
            Err(err) => Err(Error::RuntimeError(
                            format!("{}", err.overlay(&self.src)))),
        }
    }
}

/// A remake error with a descriptive human-readable message explaining
/// what went wrong.
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

// TODO(ethan): Docs.
//      - README.md
//          - Badges
//              - coveralls
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

    macro_rules! error_pre {
        ($test_name:ident, $remake_src:expr, $expected_err_str:expr) => {
            #[test]
            fn $test_name() {
                let result = Remake::compile($remake_src);
                match &result {
                    &Err(ref err) => {
                        let err_msg = format!("{}", err);
                        // When copying the right output into the test
                        // case uncommenting this can help debug.
                        //
                        // assert_eq!($expected_err_str, err_msg);
                        
                        if !err_msg.starts_with($expected_err_str) {
                            // We unwrap rather than asserting or something
                            // a little more reasonable so that we can see
                            // the error output as the user sees it.
                            result.clone().unwrap();
                        }
                    }
                    &Ok(ref res) =>
                        panic!("Should not eval. res={:?}", res),
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
    
    error_pre!(unmatched_tick_1_, r"'a",
        r#"
remake parse error:
    at line 1, col 1:
    0001 > 'a
           ^^
remake lexical error:
Unclosed raw regex literal."#);

    error_pre!(unmatched_tick_2_, r"a'",
        r#"
remake parse error:
    at line 1, col 2:
    0001 > a'
            ^
remake lexical error:
Unclosed raw regex literal."#);

    error_pre!(unmatched_slash_1_, r"/a",
        r#"
remake parse error:
    at line 1, col 1:
    0001 > /a
           ^^
remake lexical error:
Unclosed regex literal."#);

    error_pre!(unmatched_slash_2_, r"a/",
        r#"
remake parse error:
    at line 1, col 2:
    0001 > a/
            ^
remake lexical error:
Unclosed regex literal."#);

    error_pre!(unmatched_tick_slash_1_, r"'a/",
        r#"
remake parse error:
    at line 1, col 1:
    0001 > 'a/
           ^^^
remake lexical error:
Unclosed raw regex literal."#);

    error_pre!(unmatched_tick_slash_2_, r"/a'",
        r#"
remake parse error:
    at line 1, col 1:
    0001 > /a'
           ^^^
remake lexical error:
Unclosed regex literal."#);

    //
    // parse errors that bubble up from the regex crate
    //
    
    error_pre!(re_parse_err_1_, r"/a[/", 
        r#"
remake parse error:
    at line 1, col 1:
    0001 > /a[/
           ^^^^
Error parsing the regex literal: /a[/
    regex parse error:
        a[
         ^
    error: unclosed character class"#);

    error_pre!(re_parse_err_2_, r"/a[]/",
        r#"
remake parse error:
    at line 1, col 1:
    0001 > /a[]/
           ^^^^^
Error parsing the regex literal: /a[]/
    regex parse error:
        a[]
         ^^
    error: unclosed character class"#);

    error_pre!(re_multiline_parse_err_1_,
        r#"

            /a[/

        "#,
        r#"
remake parse error:
    at line 3, col 13:
    0002 > 
    0003 >             /a[/
                       ^^^^
    0004 > 
Error parsing the regex literal: /a[/
    regex parse error:
        a[
         ^
    error: unclosed character class"#);

    error_pre!(unrecognized_token_1_, r"/foo/ /foo/",
        r#"
remake parse error:
    at line 1, col 7:
    0001 > /foo/ /foo/
                 ^^^^^
Unexpected token '/foo/'. Expected one of:"#);

    error_pre!(binary_plus_1_, r"/foo/ + /bar/",
        r#"
remake parse error:
    at line 1, col 9:
    0001 > /foo/ + /bar/
                   ^^^^^
Unexpected token '/bar/'. Expected one of:"#);

    error_pre!(binary_plus_multiline_1_, r#"/foo/ +
        /bar
        /"#, r#"
remake parse error:
    starting at line 2, col 9 and ending at line 3, col 10:
    0001  > /foo/ +
    0002  >         /bar
    start >         ^
    0003  >         /
    end   >         ^
Unexpected token '/bar
        /'. Expected one of:"#);

    error_pre!(binary_plus_multiline_2_, r#"
        /foo/ + /
        
        bar



        /"#, r#"
remake parse error:
    starting at line 2, col 17 and ending at line 8, col 10:
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


    error_pre!(name_error_1_, r#"
        foo
    "#, r#"
remake evaluation error:
    at line 2, col 9:
    0001 > 
    0002 >         foo
                   ^^^
    0003 >     
NameError: unknown variable 'foo'."#);

    error_pre!(num_too_long_1_,
        r"'a'{11111111111111111111111111111111111111111111111111111111}",
        r#"
remake parse error:
    at line 1, col 5:
    0001 > 'a'{11111111111111111111111111111111111111111111111111111111}
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
remake lexical error:
Error parsing '11111111111111111111111111111111111111111111111111111111' as a number:"#);


    mat!(alt_merge_1_, r"/a|b/ | /c/", "c");
    mat!(alt_merge_2_, r"/c/ | /a|b/", "c");
    mat!(alt_merge_3_, r"/c|d/ | /a|b/", "d");

    mat!(block_1_, r#"
        let gen_delims = ':' | '/' | '?' | '[' | ']' | '@';
        'foo'
    "#,
    "foo");

    mat!(repeated_concat_1_, r"/^/ . ('/' . /[\w\-_]+/)* . /$/", "");
    mat!(repeated_concat_2_, r"/^/ . ('/' . /[\w\-_]+/)* . /$/", "/a/b/c");

    mat!(repeated_concat_3_, r"/^/ . ('/' . /[\w\-_]+?/)* . /$/", "");
    mat!(repeated_concat_4_, r"/^/ . ('/' . /[\w\-_]+?/)* . /$/", "/a/b/c");

    mat!(repeated_concat_5_, r"/^/ . ('/' . /[\w\-_]+/)+ . /$/", "/a/b/c");
    mat!(repeated_concat_6_, r"/^/ . ('/' . /[\w\-_]+/)+? . /$/", "/a/b/c");

    mat!(repeated_concat_7_, r"/^/ . ('/' . /[\w\-_]+/){3} . /$/", "/a/b/c");
    mat!(repeated_concat_8_, r"/^/ . ('/' . /[\w\-_]+/){3}? . /$/", "/a/b/c");

    mat!(repeated_concat_9_, r"/^/ . ('/' . /[\w\-_]+/){1,} . /$/", "/a/b/c");
    mat!(repeated_concat_10_, r"/^/ . ('/' . /[\w\-_]+/){1,}? . /$/", "/a/b/c");

    mat!(repeated_concat_11_, r"/^/ . ('/' . /[\w\-_]+/){1,5} . /$/", "/a/b/c");
    mat!(repeated_concat_12_, r"/^/ . ('/' . /[\w\-_]+/){1,5}? . /$/", "/a/b/c");
}
