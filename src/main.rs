#![allow(dead_code)]

use clap::Parser;
use std::collections::HashMap;
use std::fmt;
use std::io::{self, Write};

const LENGTH: usize = 19;
const WIDTH: usize = 15;
const START_ROW: usize = 7;
const START_COL: usize = 9;
const WHITE_CIRCLE: char = '○';
const BLACK_CIRCLE: char = '●';

/// Phutball game - play against AI or other players
#[derive(Parser, Debug)]
#[command(name = "phutball-rust")]
#[command(about = "Play Phutball (Philosopher's Football)", long_about = None)]
struct Args {
    /// Left player type: human, minimax[:depth], or plodding
    #[arg(value_name = "LEFT_PLAYER")]
    left: String,

    /// Right player type: human, minimax[:depth], or plodding
    #[arg(value_name = "RIGHT_PLAYER")]
    right: String,
}

fn main() {
    let args = Args::parse();

    let left_player = parse_player(&args.left);
    let right_player = parse_player(&args.right);

    run_game(left_player, right_player);
}

fn parse_player(spec: &str) -> Box<dyn Player> {
    let parts: Vec<&str> = spec.split(':').collect();
    let player_type = parts[0];

    match player_type {
        "human" => Box::new(HumanPlayer),
        "minimax" => {
            if parts.len() < 2 {
                eprintln!("Error: minimax requires depth specification (e.g., minimax:3)");
                std::process::exit(1);
            }
            let depth = match parts[1].parse::<u32>() {
                Ok(d) if d > 0 => d,
                _ => {
                    eprintln!("Error: minimax depth must be a positive integer");
                    std::process::exit(1);
                }
            };
            Box::new(MinimaxPlayer::new(depth))
        }
        "plodding" => Box::new(PloddingPlayer),
        _ => {
            eprintln!("Unknown player type: {}", player_type);
            eprintln!("Valid types: human, minimax:DEPTH, plodding");
            std::process::exit(1);
        }
    }
}

fn run_game(left_player: Box<dyn Player>, right_player: Box<dyn Player>) {
    let mut board = Board::new();
    let players = [left_player, right_player];

    // Print game setup before first move
    println!("=== Phutball Game ===");
    println!("Left tries to get the ball to the RIGHT (column 19)");
    println!("Right tries to get the ball to the LEFT (column 1)");
    println!("Left plays first.\n");

    loop {
        board.pretty_print_details();

        // Check for win condition
        if let Some(winner) = board.check_winner() {
            println!("\n{:?} has won!", winner);
            break;
        }

        let player_idx = board.moves_made as usize % 2;
        let current_side = board.side_to_move;

        let player_move = players[player_idx].make_move(&board);

        // Validate and execute move
        let moves = board.get_all_moves();
        match moves.get(&player_move) {
            Some(new_board) => {
                println!("\n>>> {:?} played: {}\n", current_side, player_move.trim());
                board = new_board.clone();
            }
            None => {
                println!("Invalid move: {}", player_move);
                continue;
            }
        }
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

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Piece {
    Man,
    Empty,
    Ball,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

    /// Returns moves with man placements restricted to nearby squares (within 2 squares of any piece)
    pub fn get_all_nearby_moves(&self) -> HashMap<String, Board> {
        let mut moves = self.get_nearby_man_moves();
        moves.extend(self.get_ball_moves());
        moves
    }

    /// Returns man placement moves restricted to nearby squares
    fn get_nearby_man_moves(&self) -> HashMap<String, Board> {
        let mut moves = HashMap::new();
        let mut used = std::collections::HashSet::new();

        for row in 0..WIDTH {
            for col in 0..LENGTH {
                let pos = Position::new(row, col);
                if self.get(pos) != Piece::Empty {
                    // Found a piece, add all empty squares within 2 steps
                    for dr in -2..=2 {
                        for dc in -2..=2 {
                            if let Some(nearby) = pos.checked_add((dr, dc)) {
                                if nearby.is_on_board()
                                    && self.get(nearby) == Piece::Empty
                                    && !used.contains(&nearby) {
                                    let mut new_board = self.clone();
                                    new_board.set(nearby, Piece::Man);
                                    new_board.increment();
                                    moves.insert(nearby.to_notation(), new_board);
                                    used.insert(nearby);
                                }
                            }
                        }
                    }
                }
            }
        }

        moves
    }

    /// Check if the game has been won
    pub fn check_winner(&self) -> Option<Side> {
        if self.ball_at.col <= 0 {
            Some(Side::Right)
        } else if self.ball_at.col >= LENGTH - 1 {
            Some(Side::Left)
        } else {
            None
        }
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

            // First, check if there's at least one man to jump over
            let first_pos = match self.ball_at.checked_add(delta) {
                Some(p) => p,
                None => continue,
            };

            if !first_pos.is_on_board() || self.get(first_pos) != Piece::Man {
                continue; // No man to jump over
            }

            // Find how far we can jump - keep going while we see men
            let mut jump_length = 1;
            let (end_row, end_col): (i32, i32) = loop {
                let new_row = self.ball_at.row as i32 + delta.0 * (jump_length + 1);
                let new_col = self.ball_at.col as i32 + delta.1 * (jump_length + 1);

                // Check if we're still in valid positive coordinates to check for more men
                if new_row >= 0 && new_col >= 0 {
                    let next_pos = Position::new(new_row as usize, new_col as usize);
                    if next_pos.is_on_board() && self.get(next_pos) == Piece::Man {
                        // Continue jumping over consecutive men
                        jump_length += 1;
                        continue;
                    }
                }

                // We've either gone off the edge or found an empty/ball square
                break (new_row, new_col);
            };

            // Check if we jumped off the board via col (into goal)
            let jumped_off_col = end_col < 0 || end_col >= LENGTH as i32;

            // Check if we jumped off the board via row
            let jumped_off_row = end_row < 0 || end_row >= WIDTH as i32;

            // If we jumped off via row only (not col), that's illegal
            // But if we jumped off via col (with or without row), that's a goal
            if jumped_off_row && !jumped_off_col {
                continue;
            }

            let jumped_into_goal = jumped_off_col;

            let end_point = Position::new(
                end_row.max(0) as usize,
                end_col.max(0) as usize
            );

            // Make the jump
            let mut new_board = self.clone();

            // Clear jumped-over men (from position 1 to jump_length)
            for i in 1..=jump_length {
                if let Some(pos) = self.ball_at.checked_add((delta.0 * i, delta.1 * i)) {
                    if pos.is_on_board() {
                        new_board.set(pos, Piece::Empty);
                    }
                }
            }

            // Clear the starting position
            new_board.set(self.ball_at, Piece::Empty);

            // Place ball at destination (if still on board, not jumped into goal)
            if !jumped_into_goal {
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

            // Recursively find continuation jumps (but not if we landed on a goal col - that's a victory)
            let landed_on_goal_col = end_point.col == 0 || end_point.col == LENGTH - 1;
            if !jumped_into_goal && !landed_on_goal_col {
                new_board.get_ball_moves_recursive(moves, move_name);
            }
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

// ============================================================================
// Player Trait and Implementations
// ============================================================================

trait Player {
    fn make_move(&self, board: &Board) -> String;
}

// ----------------------------------------------------------------------------
// Human Player
// ----------------------------------------------------------------------------

struct HumanPlayer;

impl Player for HumanPlayer {
    fn make_move(&self, board: &Board) -> String {
        let moves = board.get_all_moves();

        // Show a sample jump move if available
        if let Some((jump_move, _)) = moves.iter().find(|(k, _)| k.contains(' ')) {
            println!("Example jump move: {}", jump_move.trim());
        }

        // Show a sample placement move
        if let Some((place_move, _)) = moves.iter().find(|(k, _)| !k.contains(' ')) {
            println!("Example placement move: {}", place_move);
        }

        loop {
            print!("Enter your move: ");
            io::stdout().flush().unwrap();

            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            let input = input.trim();

            // Try the input as-is first, then try with trailing space
            if moves.contains_key(input) {
                return input.to_string();
            }

            let with_space = format!("{} ", input);
            if moves.contains_key(&with_space) {
                return with_space;
            }

            println!("Invalid move. Try again.");
        }
    }
}

// ----------------------------------------------------------------------------
// Plodding Player
// ----------------------------------------------------------------------------

struct PloddingPlayer;

impl Player for PloddingPlayer {
    fn make_move(&self, board: &Board) -> String {
        let moves = board.get_all_moves();

        // Try to jump in the direction of our goal
        let preferred_dir = match board.side_to_move {
            Side::Left => "E ",
            Side::Right => "W ",
        };

        if moves.contains_key(preferred_dir) {
            return preferred_dir.to_string();
        }

        // Otherwise, place a man next to the ball in the direction of our goal
        let offset = match board.side_to_move {
            Side::Left => 1,
            Side::Right => -1,
        };

        if let Some(target) = board.ball_at.checked_add((0, offset)) {
            if target.is_on_board() {
                let move_name = target.to_notation();
                if moves.contains_key(&move_name) {
                    return move_name;
                }
            }
        }

        // Fallback: return any move
        moves.keys().next().unwrap().clone()
    }
}

// ----------------------------------------------------------------------------
// Minimax Player
// ----------------------------------------------------------------------------

struct MinimaxPlayer {
    depth: u32,
}

impl MinimaxPlayer {
    fn new(depth: u32) -> Self {
        Self { depth }
    }

    /// Negamax scoring function - returns score from perspective of player to move
    fn negamax(&self, board: &Board, depth: u32) -> f64 {
        // Check for terminal positions
        if let Some(winner) = board.check_winner() {
            return if winner == board.side_to_move {
                1.0
            } else {
                0.0
            };
        }

        // Base case: use static evaluator
        if depth == 0 {
            return LocationEvaluator::score(board);
        }

        // Recursive case
        let moves = board.get_all_nearby_moves();
        let mut max_score = 0.0;

        for (_, next_board) in moves {
            let score = 1.0 - self.negamax(&next_board, depth - 1);
            if score > max_score {
                max_score = score;
            }
        }

        max_score
    }
}

impl Player for MinimaxPlayer {
    fn make_move(&self, board: &Board) -> String {
        println!("Minimax thinking (depth {})...", self.depth);

        let moves = board.get_all_nearby_moves();
        let mut best_move = moves.keys().next().unwrap().clone();
        let mut best_score = 0.0;

        for (move_name, next_board) in moves {
            let score = 1.0 - self.negamax(&next_board, self.depth - 1);

            if score > best_score {
                best_score = score;
                best_move = move_name;
            }
        }

        println!("Minimax chose: {} (score: {:.3})", best_move.trim(), best_score);
        best_move
    }
}

// ----------------------------------------------------------------------------
// Location Evaluator
// ----------------------------------------------------------------------------

struct LocationEvaluator;

impl LocationEvaluator {
    /// Returns normalized position of ball (0.0 to 1.0)
    /// Score is close to 1 if ball is near the goal of the player to move
    fn score(board: &Board) -> f64 {
        let mut location = board.ball_at.col as i32;

        // Clamp to board edges
        if location <= 0 {
            location = 0;
        }
        if location >= LENGTH as i32 - 1 {
            location = LENGTH as i32 - 1;
        }

        let value = location as f64 / (LENGTH - 1) as f64;

        // value is near 1 if we're to the east (Right's goal)
        match board.side_to_move {
            Side::Left => value,
            Side::Right => 1.0 - value,
        }
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
        // Create a board with two consecutive men to the east
        let mut board = Board::new();
        board.set(Position::new(START_ROW, START_COL + 1), Piece::Man);
        board.set(Position::new(START_ROW, START_COL + 2), Piece::Man);

        let ball_moves = board.get_ball_moves();
        // Should have one jump: "E " that jumps over both men at once
        assert!(ball_moves.contains_key("E "));

        // Ball should land at START_COL + 3 (after both men)
        let jumped = &ball_moves["E "];
        assert_eq!(jumped.ball_at.col, START_COL + 3);
        // Both men should be removed
        assert_eq!(jumped.get(Position::new(START_ROW, START_COL + 1)), Piece::Empty);
        assert_eq!(jumped.get(Position::new(START_ROW, START_COL + 2)), Piece::Empty);
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
        // Test that we can jump multiple times in one turn (change directions)
        let mut board = Board::new();
        // Place men for a jump east, then after landing, can jump northeast
        board.set(Position::new(START_ROW, START_COL + 1), Piece::Man);
        // After jumping east, ball lands at START_COL + 2
        // Place a man northeast of that landing position
        board.set(Position::new(START_ROW - 1, START_COL + 3), Piece::Man);

        let ball_moves = board.get_ball_moves();
        // Should have "E " (single jump)
        assert!(ball_moves.contains_key("E "));
        // Should have "E NE " (jump east, then northeast)
        assert!(ball_moves.contains_key("E NE "));

        let double_jump = &ball_moves["E NE "];
        assert_eq!(double_jump.ball_at.row, START_ROW - 2);
        assert_eq!(double_jump.ball_at.col, START_COL + 4);
    }

    #[test]
    fn test_jump_into_goal() {
        // Ball at A2 (row 0, col 1), man at A1 (row 0, col 0)
        let mut board = Board::new();
        // Clear the starting position
        board.set(board.ball_at, Piece::Empty);

        // Place ball at A2
        let ball_pos = Position::new(0, 1);
        board.ball_at = ball_pos;
        board.set(ball_pos, Piece::Ball);

        // Place man at A1 (in the left goal line)
        let man_pos = Position::new(0, 0);
        board.set(man_pos, Piece::Man);

        let ball_moves = board.get_ball_moves();

        // Should be able to jump west over the man into the goal
        assert!(ball_moves.contains_key("W "), "Should have W jump");

        // After jumping west, ball should be off the board (col -1 or 0)
        let jumped = &ball_moves["W "];
        assert!(jumped.ball_at.col <= 0 || !jumped.ball_at.is_on_board());

        // The man should be cleared
        assert_eq!(jumped.get(man_pos), Piece::Empty);

        // Should not be able to jump in other directions (no men there)
        assert!(!ball_moves.contains_key("E "), "Should not have E jump");
        assert!(!ball_moves.contains_key("N "), "Should not have N jump");
        assert!(!ball_moves.contains_key("S "), "Should not have S jump");
        assert!(!ball_moves.contains_key("NE "), "Should not have NE jump");
        assert!(!ball_moves.contains_key("NW "), "Should not have NW jump");
        assert!(!ball_moves.contains_key("SE "), "Should not have SE jump");
        assert!(!ball_moves.contains_key("SW "), "Should not have SW jump");

        // Total: should only have the one jump west
        assert_eq!(ball_moves.len(), 1, "Should only have 1 jump move");
    }

    #[test]
    fn test_multijump_stops_at_goal_col() {
        // If a multijump would land on a goal column (col 0 or LENGTH-1),
        // the move should stop there - no further jumps allowed
        let mut board = Board::new();
        board.set(board.ball_at, Piece::Empty);

        // Ball at col 3, men at col 2 and col 0 (goal line)
        // After first jump W, ball lands at col 1
        // After second jump W, ball lands at col 0 (goal line) - this should be a victory, no more jumps
        let ball_pos = Position::new(7, 3);
        board.ball_at = ball_pos;
        board.set(ball_pos, Piece::Ball);

        // Man to jump over to reach col 1
        board.set(Position::new(7, 2), Piece::Man);
        // Man at col 0 to potentially jump over (but shouldn't - col 0 is goal)
        board.set(Position::new(7, 0), Piece::Man);

        let ball_moves = board.get_ball_moves();

        // Should have "W " move that lands at col 1
        assert!(ball_moves.contains_key("W "), "Should have W jump");
        let after_w = &ball_moves["W "];
        assert_eq!(after_w.ball_at.col, 1);

        // Should NOT have "W W " move - can't continue jumping after landing at col 1
        // because the next jump would land on col 0 (goal), and that's the end
        // Actually, we CAN jump to col 0, but we can't jump AWAY from col 0

        // Let's set up differently: ball at col 2, man at col 1, man at col 0
        // Jump W lands at col 0 (goal) - game over, no more jumps
        let mut board2 = Board::new();
        board2.set(board2.ball_at, Piece::Empty);

        let ball_pos2 = Position::new(7, 2);
        board2.ball_at = ball_pos2;
        board2.set(ball_pos2, Piece::Ball);

        board2.set(Position::new(7, 1), Piece::Man);
        // Man that could be jumped if we could continue from col 0
        board2.set(Position::new(6, 0), Piece::Man);

        let ball_moves2 = board2.get_ball_moves();

        // Should have "W " that lands at col 0
        assert!(ball_moves2.contains_key("W "), "Should have W jump to goal col");
        let after_w2 = &ball_moves2["W "];
        assert_eq!(after_w2.ball_at.col, 0, "Should land on goal col");

        // Should NOT have any continuation jumps from col 0 (like "W NE " or "W SE ")
        assert!(!ball_moves2.contains_key("W NE "), "Should not continue from goal col");
        assert!(!ball_moves2.contains_key("W SE "), "Should not continue from goal col");
        assert!(!ball_moves2.contains_key("W NW "), "Should not continue from goal col");
        assert!(!ball_moves2.contains_key("W SW "), "Should not continue from goal col");
    }

    #[test]
    fn test_diagonal_jump_into_goal() {
        // Jumping diagonally off both row and col boundaries should be legal (a goal)
        let mut board = Board::new();
        board.set(board.ball_at, Piece::Empty);

        // Ball at row 0, col 1. Man at row 0, col 0 (corner).
        // Jumping SW would go to row 1, col -1 (off col = goal)
        // But wait, SW is (1, -1) so from (0, 1) we'd go to (1, 0) then check (2, -1)
        // Actually let's think more carefully...

        // Let's put ball at corner: row 0, col 1
        // Man at row 0, col 0
        // Jump W goes to row 0, col -1 (goal, but row stays valid)

        // For a true diagonal-off-both test:
        // Ball at row 1, col 1
        // Man at row 0, col 0
        // Jump NW: delta is (-1, -1), so end would be row -1, col -1
        // This should be a legal goal (col is off, row is also off but that's ok)

        let ball_pos = Position::new(1, 1);
        board.ball_at = ball_pos;
        board.set(ball_pos, Piece::Ball);

        // Man at corner (0, 0)
        board.set(Position::new(0, 0), Piece::Man);

        let ball_moves = board.get_ball_moves();

        // Should be able to jump NW into the goal
        assert!(ball_moves.contains_key("NW "), "Should have NW diagonal jump into goal");

        let jumped = &ball_moves["NW "];
        // Ball should be off-board (clamped to 0,0 but conceptually off)
        assert!(jumped.ball_at.col == 0, "Ball col should be clamped to 0");

        // The man should be cleared
        assert_eq!(jumped.get(Position::new(0, 0)), Piece::Empty, "Man should be cleared");
    }

    #[test]
    fn test_no_jump_off_row_only() {
        // Jumping off the board via row only (not col) should be illegal
        let mut board = Board::new();
        board.set(board.ball_at, Piece::Empty);

        // Ball at row 1, col 5 (middle of board horizontally)
        // Man at row 0, col 5
        // Jumping N would go to row -1, col 5 - illegal (off row, not off col)
        let ball_pos = Position::new(1, 5);
        board.ball_at = ball_pos;
        board.set(ball_pos, Piece::Ball);

        board.set(Position::new(0, 5), Piece::Man);

        let ball_moves = board.get_ball_moves();

        // Should NOT be able to jump north off the board
        assert!(!ball_moves.contains_key("N "), "Should not be able to jump off north edge");
    }

    #[test]
    fn test_no_jump_sequence_through_goal() {
        // A single jump that passes through a goal line position should stop there
        // This tests the "can't jump off board and back on" rule
        // In practice, this can't happen with a normal board (you can't be beyond the goal),
        // but we verify the consecutive-men jump logic doesn't somehow skip past the goal

        let mut board = Board::new();
        board.set(board.ball_at, Piece::Empty);

        // Ball at col 2, men at col 1 AND col 0 (consecutive)
        // Jumping W over both should land at col -1 (off board = goal)
        // We should NOT somehow continue and land back on the board
        let ball_pos = Position::new(7, 2);
        board.ball_at = ball_pos;
        board.set(ball_pos, Piece::Ball);

        board.set(Position::new(7, 1), Piece::Man);
        board.set(Position::new(7, 0), Piece::Man);

        let ball_moves = board.get_ball_moves();

        // Should have W jump that clears both men and goes to goal
        assert!(ball_moves.contains_key("W "), "Should have W jump");
        let after_w = &ball_moves["W "];

        // Ball should be at col 0 (clamped from -1)
        assert_eq!(after_w.ball_at.col, 0, "Ball should be at goal");

        // Both men should be cleared
        assert_eq!(after_w.get(Position::new(7, 1)), Piece::Empty, "First man cleared");
        assert_eq!(after_w.get(Position::new(7, 0)), Piece::Empty, "Second man cleared");

        // Ball should NOT be placed on the board (it's in the goal)
        assert_eq!(after_w.get(Position::new(7, 0)), Piece::Empty, "Ball not placed at col 0");
    }
}
