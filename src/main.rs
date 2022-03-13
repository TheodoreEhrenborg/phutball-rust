#![allow(dead_code)]
const LENGTH: usize = 19;
const WIDTH: usize = 15;
const START: (usize, usize) = (7, 9);
const WHITE_CIRCLE: char = '○'; // Looks good on GitHub
const BLACK_CIRCLE: char = '●';
const UPPERCASE_LETTERS: [char; 26] = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'Y', 'Y', 'Z',
];

fn main() {
    let b = Board::new();
    b.pretty_print_details();
}
// Be sure to check that coordinates of ball are good
#[derive(Debug)]
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
    /// Returns a pretty string with details
    pub fn pretty_string_details(&self) -> String {
        let mut output = String::from("");
        output.push_str("          1111111111\n");
        output.push_str(" 1234567890123456789\n");
        for (i, row) in self.array.iter().enumerate() {
            output.push(UPPERCASE_LETTERS[i]);
            for element in row {
                match element {
                    Piece::Man => output.push(WHITE_CIRCLE),
                    Piece::Ball => output.push(BLACK_CIRCLE),
                    Piece::Empty => output.push('+'),
                }
            }
            output.push('\n');
        }
        output.push_str(
            format!(
                "Side to move: {:?}\nMoves made: {}\nBall at: {:?}\n",
                self.side_to_move, self.moves_made, self.ball_at
            )
            .as_str(),
        );
        output
    }

    /// Returns a pretty string
    pub fn pretty_string(&self) -> String {
        let mut output = String::from("");
        for row in self.array {
            for element in row {
                match element {
                    Piece::Man => output.push(WHITE_CIRCLE),
                    Piece::Ball => output.push(BLACK_CIRCLE),
                    Piece::Empty => output.push('+'),
                }
            }
            output.push('\n')
        }
        output
    }

    /// Prints a pretty string
    pub fn pretty_print(&self) {
        println!("{}", self.pretty_string());
    }

    /// Prints a pretty string with details
    pub fn pretty_print_details(&self) {
        println!("{}", self.pretty_string_details());
    }
}
