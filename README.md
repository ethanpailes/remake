
remake
======


A rust library for writing maintainable regex. Regex are wonderful
things, but they lack tools for abstraction and modularity.
When regex are small, this is not a big issue, but as they grow
larger it can become frustrating to maintain them. Remake allows
you to name individual regex and combine them just like you would
combine strings or numbers in a traditional programming language.
When you want to use the terse regex syntax that you know and love,
you still can, but when you want to break things down into bite
sized pieces it is easy.

[![Coverage Status](https://coveralls.io/repos/github/ethanpailes/remake/badge.svg?branch=master)](https://coveralls.io/github/ethanpailes/remake?branch=master)
[![Build Status](https://circleci.com/gh/ethanpailes/remake.svg?style=svg)](https://circleci.com/gh/ethanpailes/remake)
[![](http://meritbadge.herokuapp.com/remake)](https://crates.io/crates/remake)
[![Docs](https://docs.rs/remake/badge.svg)](https://docs.rs/remake)


### Documentation

The [module docs](https://docs.rs/remake) contain a full explanation
of the remake language with inline examples.

### Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
remake = "0.1.0"
```

and this to your crate root:

```rust
extern crate remake;
```

Here is a simple example that builds a toy URI validator

```rust
extern crate remake;

use remake::Remake;

fn main() {
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
            .captures("https://tools.ietf.org/html/rfc3986#section-1.1.3")
                .unwrap().name("frag").unwrap().as_str(),
            "section-1.1.3");
}
```
