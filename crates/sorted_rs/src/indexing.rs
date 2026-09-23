pub(super) enum Nb {
    Pos,
    One,
    Zero,
    NegOne,
    Neg,
}
impl From<isize> for Nb {
    fn from(value: isize) -> Self {
        match value {
            0 => Self::Zero,
            1 => Self::One,
            -1 => Self::NegOne,
            _ if value > 1 => Self::Pos,
            _ => Self::Neg,
        }
    }
}
