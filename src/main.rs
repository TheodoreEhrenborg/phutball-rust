#![allow(dead_code)]

use std::collections::HashMap;
use std::fmt;

const LENGTH: usize = 19;
const WIDTH: usize = 15;
const START_ROW: usize = 7;
const START_COL: usize = 9;
const WHITE_CIRCLE: char = '○';
const BLACK_CIRCLE: char = '●';

fn main() {
    let board = Board::new();
    board.pretty_print_details();

    // Test move generation
    let moves = board.get_all_moves();
    println!("\nAvailable moves: {}", moves.len());
    for (i, (name, _)) in moves.iter().enumerate().take(5) {
        println!("  {}: {}", i + 1, name);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    Left,
    Right,
}

impl Side {
    fn flip(&self) -> Self {
        match self {
            Side::Left => Side::Right,
            Side::Right => Side::Left,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum Piece {
    Man,
    Empty,
    Ball,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Position {
    row: usize,
    col: usize,
}

impl Position {
    fn new(row: usize, col: usize) -> Self {
        Self { row, col }
    }

    fn is_on_board(&self) -> bool {
        self.row < WIDTH && self.col < LENGTH
    }

    /// Adds a direction vector, returning None if out of bounds
    fn checked_add(&self, delta: (i32, i32)) -> Option<Self> {
        let new_row = self.row as i32 + delta.0;
        let new_col = self.col as i32 + delta.1;

        if new_row >= 0 && new_col >= 0 {
            Some(Position::new(new_row as usize, new_col as usize))
        } else {
            None
        }
    }

    /// Converts to chess-like notation (e.g., "A1", "H10")
    fn to_notation(&self) -> String {
        let row_char = (b'A' + self.row as u8) as char;
        let col_num = self.col + 1;
        format!("{}{}", row_char, col_num)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Direction {
    N,
    S,
    E,
    W,
    NE,
    NW,
    SE,
    SW,
}

impl Direction {
    const ALL: [Direction; 8] = [
        Direction::N,
        Direction::S,
        Direction::E,
        Direction::W,
        Direction::NE,
        Direction::NW,
        Direction::SE,
        Direction::SW,
    ];

    fn delta(&self) -> (i32, i32) {
        match self {
            Direction::N => (-1, 0),
            Direction::S => (1, 0),
            Direction::E => (0, 1),
            Direction::W => (0, -1),
            Direction::NE => (-1, 1),
            Direction::NW => (-1, -1),
            Direction::SE => (1, 1),
            Direction::SW => (1, -1),
        }
    }
}

impl fmt::Display for Direction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Clone)]
struct Board {
    side_to_move: Side,
    moves_made: u32,
    array: [[Piece; LENGTH]; WIDTH],
    ball_at: Position,
}

impl Board {
    /// Creates a fresh board
    pub fn new() -> Self {
        let mut array = [[Piece::Empty; LENGTH]; WIDTH];
        array[START_ROW][START_COL] = Piece::Ball;
        Self {
            side_to_move: Side::Left,
            moves_made: 0,
            array,
            ball_at: Position::new(START_ROW, START_COL),
        }
    }

    fn get(&self, pos: Position) -> Piece {
        self.array[pos.row][pos.col]
    }

    fn set(&mut self, pos: Position, piece: Piece) {
        self.array[pos.row][pos.col] = piece;
    }

    /// Returns all possible moves as a map from move notation to resulting board
    pub fn get_all_moves(&self) -> HashMap<String, Board> {
        let mut moves = self.get_man_moves();
        moves.extend(self.get_ball_moves());
        moves
    }

    /// Returns all moves that involve placing a man
    fn get_man_moves(&self) -> HashMap<String, Board> {
        let mut moves = HashMap::new();

        for row in 0..WIDTH {
            for col in 0..LENGTH {
                let pos = Position::new(row, col);
                if self.get(pos) == Piece::Empty {
                    let mut new_board = self.clone();
                    new_board.set(pos, Piece::Man);
                    new_board.increment();
                    moves.insert(pos.to_notation(), new_board);
                }
            }
        }

        moves
    }

    /// Returns all moves that involve jumping the ball
    /// This is recursive to handle multi-jump sequences
    fn get_ball_moves(&self) -> HashMap<String, Board> {
        let mut moves = HashMap::new();
        self.get_ball_moves_recursive(&mut moves, String::new());
        moves
    }

    fn get_ball_moves_recursive(&self, moves: &mut HashMap<String, Board>, prefix: String) {
        // Ball might be off the board (in a goal)
        if !self.ball_at.is_on_board() {
            return;
        }

        for direction in Direction::ALL {
            let delta = direction.delta();
            let mut jump_length = 0;
            let mut end_point;

            // Find how far we can jump in this direction
            loop {
                jump_length += 1;
                end_point = match self.ball_at.checked_add((delta.0 * jump_length, delta.1 * jump_length)) {
                    Some(p) => p,
                    None => break,
                };

                // If we're on the board and hit a non-man, stop
                if end_point.is_on_board() && self.get(end_point) != Piece::Man {
                    break;
                }

                // If we went off the board, stop
                if !end_point.is_on_board() {
                    break;
                }
            }

            // Need at least one man to jump over
            if jump_length <= 1 {
                continue;
            }

            // Check if we jumped off the board illegally
            // Legal only if we jumped over a man on the goal line (col 0 or LENGTH-1)
            if !end_point.is_on_board() {
                let prev_point = self.ball_at.checked_add((delta.0 * (jump_length - 1), delta.1 * (jump_length - 1))).unwrap();
                if prev_point.col != 0 && prev_point.col != LENGTH - 1 {
                    continue;
                }
            }

            // Make the jump
            let mut new_board = self.clone();

            // Clear jumped-over men
            for i in 0..jump_length {
                if let Some(pos) = self.ball_at.checked_add((delta.0 * i, delta.1 * i)) {
                    new_board.set(pos, Piece::Empty);
                }
            }

            // Place ball at destination (if on board)
            if end_point.is_on_board() {
                new_board.set(end_point, Piece::Ball);
            }
            new_board.ball_at = end_point;

            // Add this move
            let move_name = format!("{}{} ", prefix, direction);
            let final_board = {
                let mut b = new_board.clone();
                b.increment();
                b
            };
            moves.insert(move_name.clone(), final_board);

            // Recursively find continuation jumps
            new_board.get_ball_moves_recursive(moves, move_name);
        }
    }

    fn increment(&mut self) {
        self.side_to_move = self.side_to_move.flip();
        self.moves_made += 1;
    }

    /// Returns a pretty string with details
    pub fn pretty_string_details(&self) -> String {
        let mut output = String::new();
        output.push_str("          1111111111\n");
        output.push_str(" 1234567890123456789\n");

        for row in 0..WIDTH {
            output.push((b'A' + row as u8) as char);
            for col in 0..LENGTH {
                let pos = Position::new(row, col);
                match self.get(pos) {
                    Piece::Man => output.push(WHITE_CIRCLE),
                    Piece::Ball => output.push(BLACK_CIRCLE),
                    Piece::Empty => output.push('+'),
                }
            }
            output.push('\n');
        }

        output.push_str(&format!(
            "Side to move: {:?}\nMoves made: {}\nBall at: {}\n",
            self.side_to_move,
            self.moves_made,
            self.ball_at.to_notation()
        ));
        output
    }

    /// Returns a pretty string
    pub fn pretty_string(&self) -> String {
        let mut output = String::new();
        for row in 0..WIDTH {
            for col in 0..LENGTH {
                let pos = Position::new(row, col);
                match self.get(pos) {
                    Piece::Man => output.push(WHITE_CIRCLE),
                    Piece::Ball => output.push(BLACK_CIRCLE),
                    Piece::Empty => output.push('+'),
                }
            }
            output.push('\n');
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_board() {
        let board = Board::new();
        assert_eq!(board.side_to_move, Side::Left);
        assert_eq!(board.moves_made, 0);
        assert_eq!(board.ball_at.row, START_ROW);
        assert_eq!(board.ball_at.col, START_COL);
        assert_eq!(board.get(board.ball_at), Piece::Ball);
    }

    #[test]
    fn test_position_notation() {
        assert_eq!(Position::new(0, 0).to_notation(), "A1");
        assert_eq!(Position::new(7, 9).to_notation(), "H10");
        assert_eq!(Position::new(14, 18).to_notation(), "O19");
    }

    #[test]
    fn test_side_flip() {
        assert_eq!(Side::Left.flip(), Side::Right);
        assert_eq!(Side::Right.flip(), Side::Left);
    }

    #[test]
    fn test_position_checked_add() {
        let pos = Position::new(5, 5);
        assert_eq!(pos.checked_add((1, 1)), Some(Position::new(6, 6)));
        assert_eq!(pos.checked_add((-1, -1)), Some(Position::new(4, 4)));
        assert_eq!(pos.checked_add((-10, 0)), None); // Would go negative
    }

    #[test]
    fn test_position_is_on_board() {
        assert!(Position::new(0, 0).is_on_board());
        assert!(Position::new(WIDTH - 1, LENGTH - 1).is_on_board());
        assert!(!Position::new(WIDTH, LENGTH).is_on_board());
    }

    #[test]
    fn test_initial_man_moves() {
        let board = Board::new();
        let moves = board.get_man_moves();
        // Should be able to place a man on every empty square
        // That's WIDTH * LENGTH - 1 (minus the ball position)
        assert_eq!(moves.len(), WIDTH * LENGTH - 1);

        // Verify a specific move
        let moved_board = moves.get("A1").expect("A1 should be a valid move");
        assert_eq!(moved_board.get(Position::new(0, 0)), Piece::Man);
        assert_eq!(moved_board.side_to_move, Side::Right);
        assert_eq!(moved_board.moves_made, 1);
    }

    #[test]
    fn test_no_ball_moves_initially() {
        let board = Board::new();
        let ball_moves = board.get_ball_moves();
        // No men on the board initially, so no jumps possible
        assert_eq!(ball_moves.len(), 0);
    }

    #[test]
    fn test_simple_jump() {
        // Create a board with a man to the east of the ball
        let mut board = Board::new();
        let man_pos = Position::new(START_ROW, START_COL + 1);
        board.set(man_pos, Piece::Man);

        let ball_moves = board.get_ball_moves();
        // Should have one move: jump east
        assert!(ball_moves.contains_key("E "));

        let jumped_board = &ball_moves["E "];
        // Ball should be at START_COL + 2
        assert_eq!(jumped_board.ball_at.col, START_COL + 2);
        // Man should be removed
        assert_eq!(jumped_board.get(man_pos), Piece::Empty);
        // Ball square should have ball
        assert_eq!(jumped_board.get(jumped_board.ball_at), Piece::Ball);
    }

    #[test]
    fn test_multi_jump() {
        // Create a board with two men to the east
        let mut board = Board::new();
        board.set(Position::new(START_ROW, START_COL + 1), Piece::Man);
        board.set(Position::new(START_ROW, START_COL + 2), Piece::Man);

        let ball_moves = board.get_ball_moves();
        // Should have jumps: "E " (jump one) and potentially "E E " (jump both)
        assert!(ball_moves.contains_key("E "));

        // Check that jumping over both men is possible
        let jumped_once = &ball_moves["E "];
        assert_eq!(jumped_once.ball_at.col, START_COL + 2);
    }

    #[test]
    fn test_direction_deltas() {
        assert_eq!(Direction::N.delta(), (-1, 0));
        assert_eq!(Direction::S.delta(), (1, 0));
        assert_eq!(Direction::E.delta(), (0, 1));
        assert_eq!(Direction::W.delta(), (0, -1));
        assert_eq!(Direction::NE.delta(), (-1, 1));
        assert_eq!(Direction::NW.delta(), (-1, -1));
        assert_eq!(Direction::SE.delta(), (1, 1));
        assert_eq!(Direction::SW.delta(), (1, -1));
    }

    #[test]
    fn test_get_all_moves() {
        let board = Board::new();
        let all_moves = board.get_all_moves();
        let man_moves = board.get_man_moves();
        let ball_moves = board.get_ball_moves();

        // All moves should be the union of man moves and ball moves
        assert_eq!(all_moves.len(), man_moves.len() + ball_moves.len());
    }

    #[test]
    fn test_diagonal_jump() {
        // Test a northeast jump
        let mut board = Board::new();
        let man_pos = Position::new(START_ROW - 1, START_COL + 1);
        board.set(man_pos, Piece::Man);

        let ball_moves = board.get_ball_moves();
        assert!(ball_moves.contains_key("NE "));

        let jumped = &ball_moves["NE "];
        assert_eq!(jumped.ball_at.row, START_ROW - 2);
        assert_eq!(jumped.ball_at.col, START_COL + 2);
        assert_eq!(jumped.get(man_pos), Piece::Empty);
    }

    #[test]
    fn test_jump_sequence() {
        // Test that we can jump multiple times in one turn
        let mut board = Board::new();
        // Place men for a jump east then north
        board.set(Position::new(START_ROW, START_COL + 1), Piece::Man);
        board.set(Position::new(START_ROW - 1, START_COL + 2), Piece::Man);

        let ball_moves = board.get_ball_moves();
        // Should have "E " and "E NE " among others
        assert!(ball_moves.contains_key("E "));
        assert!(ball_moves.contains_key("E NE "));

        let double_jump = &ball_moves["E NE "];
        assert_eq!(double_jump.ball_at.row, START_ROW - 2);
        assert_eq!(double_jump.ball_at.col, START_COL + 3);
    }
}
