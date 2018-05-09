use regex_syntax::ast::{Position, Span};

pub const POISON_SPAN: Span = Span {
    start: Position {
        offset: 0,
        line: 1,
        column: 1,
    },
    end: Position {
        offset: 0,
        line: 1,
        column: 1,
    },
};
