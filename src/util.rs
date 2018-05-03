use regex_syntax::ast::{Span, Position};

// TODO(ethan): There are some cases where it is not clear that there
//              is a sensible value to give a regex ast span. Idk what
//              to do there.
pub const POISON_SPAN: Span = Span {
    start: Position { offset: 0, line: 1, column: 1 },
    end: Position { offset: 0, line: 1, column: 1 },
};
