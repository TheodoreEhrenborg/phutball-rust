#![allow(dead_code)]
const LENGTH: usize = 19;
const WIDTH: usize = 15;
const START: (usize, usize) = (7, 9);
const WHITE_CIRCLE: &str = "○"; // Looks good on GitHub
const BLACK_CIRCLE: &str = "●";

fn main() {
    println!("Hello, world!");
}
// Be sure to check that coordinates of ball are good
enum Side {
    Left,
    Right,
}

#[derive(Copy, Clone)]
enum Piece {
    Man,
    Empty,
    Ball,
}

struct Board {
    side_to_move: Side,
    moves_made: u32,
    array: [[Piece; LENGTH]; WIDTH],
    ball_at: (usize, usize),
}

impl Board {
    /// Creates a fresh board
    pub fn new() -> Self {
        let mut array: [[Piece; LENGTH]; WIDTH] = [[Piece::Empty; LENGTH]; WIDTH];
        array[START.0][START.1] = Piece::Ball;
        return Self {
            side_to_move: Side::Left,
            moves_made: 0,
            array,
            ball_at: START,
        };
    }

     //  /// Returns a pretty string
     //   pub fn pretty_string(&self) -> String {
     // let mut output = String::from("");
     // s1.push_str(s2);
     // }
}
