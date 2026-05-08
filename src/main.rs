#![allow(dead_code)]

use clap::{Parser, Subcommand};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt;
use std::io::{self, Write};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const LENGTH: usize = 19;
const WIDTH: usize = 15;
const START_ROW: usize = 7;
const START_COL: usize = 9;
const WHITE_CIRCLE: char = '○';
const BLACK_CIRCLE: char = '●';

const WEIGHT_PROGRESS:    f64 = 0.6;
const WEIGHT_MEN_NEAR:    f64 = 0.3;
const WEIGHT_DIRECTIONAL: f64 = 0.1;

// ============================================================================
// CLI
// ============================================================================

#[derive(Parser, Debug)]
#[command(name = "phutball-rust", about = "Play Phutball (Philosopher's Football)")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Play a single game between two players
    Play {
        /// Left player: human, minimax:D, alphabeta:D, parallel:D, timed[:ms], plodding
        left: String,
        /// Right player: human, minimax:D, alphabeta:D, parallel:D, timed[:ms], plodding
        right: String,
    },
    /// Run a tournament between two engines using Beta posterior stopping
    Tournament {
        /// First engine spec (e.g. alphabeta:3 or timed:1000)
        engine1: String,
        /// Second engine spec
        engine2: String,
        /// Stop when P(e1 > e2) exceeds this confidence threshold
        #[arg(long, default_value_t = 0.9)]
        confidence: f64,
        /// Number of men to place randomly at game start
        #[arg(long, default_value_t = 4)]
        start_men: usize,
        /// Maximum number of games to play (0 = unlimited)
        #[arg(long, default_value_t = 0)]
        max_games: u32,
    },
    /// Generate imitation learning data from engine-vs-engine games
    GenerateData {
        /// Number of games to play
        #[arg(long, default_value_t = 200)]
        games: usize,
        /// Engine spec to use as teacher (e.g. eval:1000)
        #[arg(long, default_value = "eval:1000")]
        engine: String,
        /// Output file path
        #[arg(long, default_value = "imitation.dat")]
        out: String,
    },
    /// Pre-train network on imitation data to warm-start self-play
    Pretrain {
        /// Input data file
        #[arg(long, default_value = "imitation.dat")]
        data: String,
        /// Number of training epochs
        #[arg(long, default_value_t = 20)]
        epochs: usize,
        /// Output weights file
        #[arg(long, default_value = "weights.bin")]
        save: String,
    },
    /// Self-play training loop (fine-tune from existing weights)
    Train {
        /// Number of self-play iterations
        #[arg(long, default_value_t = 10)]
        iterations: usize,
        /// Number of games per iteration
        #[arg(long, default_value_t = 20)]
        games: usize,
        /// Weights file to load and save
        #[arg(long, default_value = "weights.bin")]
        save: String,
        /// Optional imitation data to mix in each iteration (prevents mode collapse)
        #[arg(long)]
        replay: Option<String>,
    },
    /// Generate NNUE value-network training data from engine-vs-engine games
    NnueGenData {
        /// Number of games to play
        #[arg(long, default_value_t = 100)]
        games: usize,
        /// Engine spec to use as teacher (e.g. eval5:1000)
        #[arg(long, default_value = "eval5:1000")]
        engine: String,
        /// Output data file
        #[arg(long, default_value = "nnue.dat")]
        out: String,
    },
    /// Train NNUE value network on labeled game outcome data
    NnueTrain {
        /// Input data file produced by nnue-gen-data
        #[arg(long, default_value = "nnue.dat")]
        data: String,
        /// Number of training epochs
        #[arg(long, default_value_t = 50)]
        epochs: usize,
        /// Output weights file
        #[arg(long, default_value = "nnue.bin")]
        save: String,
    },
    /// Generate NNUE training data from random board positions labeled by engine
    NnueGenRandom {
        /// Number of random positions to generate
        #[arg(long, default_value_t = 5000)]
        positions: usize,
        /// Maximum number of men to place on each random board
        #[arg(long, default_value_t = 5)]
        max_men: usize,
        /// Engine spec used to label each position (e.g. eval5:200)
        #[arg(long, default_value = "eval5:200")]
        engine: String,
        /// Output data file
        #[arg(long, default_value = "nnue_random.dat")]
        out: String,
    },
    /// Merge multiple NNUE data files into one combined file
    NnueMerge {
        /// Comma-separated list of input .dat files
        #[arg(long, value_delimiter = ',')]
        inputs: Vec<String>,
        /// Output data file
        #[arg(long, default_value = "nnue_combined.dat")]
        out: String,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Play { left, right } => {
            let left_player = parse_player(&left, true);
            let right_player = parse_player(&right, true);
            run_game(left_player, right_player);
        }
        Command::Tournament { engine1, engine2, confidence, start_men, max_games } => {
            run_tournament(&engine1, &engine2, confidence, start_men, max_games);
        }
        Command::GenerateData { games, engine, out } => {
            run_generate_data(games, &engine, &out);
        }
        Command::Pretrain { data, epochs, save } => {
            run_pretrain(&data, epochs, &save);
        }
        Command::Train { iterations, games, save, replay } => {
            run_train(iterations, games, &save, replay.as_deref());
        }
        Command::NnueGenData { games, engine, out } => {
            run_nnue_gen_data(games, &engine, &out);
        }
        Command::NnueTrain { data, epochs, save } => {
            run_nnue_train(&data, epochs, &save);
        }
        Command::NnueGenRandom { positions, max_men, engine, out } => {
            run_nnue_gen_random(positions, max_men, &engine, &out);
        }
        Command::NnueMerge { inputs, out } => {
            run_nnue_merge(&inputs, &out);
        }
    }
}

fn parse_depth(s: Option<&str>, player_type: &str) -> u32 {
    match s {
        Some(ds) => match ds.parse::<u32>() {
            Ok(d) if d > 0 => d,
            _ => {
                eprintln!("Error: {} requires a positive integer depth", player_type);
                std::process::exit(1);
            }
        },
        None => {
            eprintln!("Error: {} requires depth (e.g., {}:3)", player_type, player_type);
            std::process::exit(1);
        }
    }
}

fn parse_player(spec: &str, verbose: bool) -> Box<dyn Player> {
    let parts: Vec<&str> = spec.split(':').collect();
    match parts[0] {
        "human" => Box::new(HumanPlayer),
        "minimax" => {
            let depth = parse_depth(parts.get(1).copied(), "minimax");
            Box::new(MinimaxPlayer::new(depth, verbose))
        }
        "alphabeta" => {
            let depth = parse_depth(parts.get(1).copied(), "alphabeta");
            Box::new(AlphaBetaPlayer::new(depth, verbose))
        }
        "parallel" => {
            let depth = parse_depth(parts.get(1).copied(), "parallel");
            Box::new(ParallelAlphaBetaPlayer::new(depth))
        }
        "timed" => {
            let ms = parts.get(1).and_then(|s| s.parse::<u64>().ok()).unwrap_or(1000);
            Box::new(TimedPlayer::new(ms, verbose))
        }
        "eval" => {
            let ms = parts.get(1).and_then(|s| s.parse::<u64>().ok()).unwrap_or(1000);
            Box::new(TimedPlayer::with_eval(ms, verbose, RichEvaluator::score))
        }
        "eval2" => {
            let ms = parts.get(1).and_then(|s| s.parse::<u64>().ok()).unwrap_or(1000);
            Box::new(TimedPlayer2::new(ms, verbose))
        }
        "eval4" => {
            let ms = parts.get(1).and_then(|s| s.parse::<u64>().ok()).unwrap_or(1000);
            Box::new(TimedPlayer2::with_eval(ms, verbose, Eval4Evaluator::score))
        }
        "eval5" => {
            let ms = parts.get(1).and_then(|s| s.parse::<u64>().ok()).unwrap_or(1000);
            Box::new(TimedPlayer5::with_eval(ms, verbose, Eval4Evaluator::score))
        }
        "eval5q" => {
            let ms = parts.get(1).and_then(|s| s.parse::<u64>().ok()).unwrap_or(1000);
            Box::new(TimedPlayer5Q::new(ms, verbose))
        }
        "eval6" => {
            let ms = parts.get(1).and_then(|s| s.parse::<u64>().ok()).unwrap_or(1000);
            Box::new(TimedPlayer6::with_eval(ms, verbose, Eval4Evaluator::score))
        }
        "mcts" => {
            let ms = parts.get(1).and_then(|s| s.parse::<u64>().ok()).unwrap_or(1000);
            Box::new(MctsPlayer::new(ms))
        }
        "mcts-eval" => {
            let ms = parts.get(1).and_then(|s| s.parse::<u64>().ok()).unwrap_or(1000);
            Box::new(MctsEvalPlayer::new(ms))
        }
        "mcts2" => {
            let ms = parts.get(1).and_then(|s| s.parse::<u64>().ok()).unwrap_or(1000);
            Box::new(Mcts2Player::new(ms))
        }
        "beam-mcts" => {
            let ms = parts.get(1).and_then(|s| s.parse::<u64>().ok()).unwrap_or(1000);
            Box::new(BeamMctsPlayer::new(ms))
        }
        "azero" => {
            let ms = parts.get(1).and_then(|s| s.parse::<u64>().ok()).unwrap_or(1000);
            Box::new(AzeroPlayer::new(ms))
        }
        "net-eval" => {
            let ms = parts.get(1).and_then(|s| s.parse::<u64>().ok()).unwrap_or(1000);
            Box::new(NetTimedPlayer::new(ms))
        }
        "nnue-eval" => {
            let ms = parts.get(1).and_then(|s| s.parse::<u64>().ok()).unwrap_or(1000);
            Box::new(TimedPlayer5::with_eval(ms, verbose, nnue_eval))
        }
        "plodding" => Box::new(PloddingPlayer),
        _ => {
            eprintln!("Unknown player type: {}", spec);
            eprintln!("Valid types: human, minimax:D, alphabeta:D, parallel:D, timed[:ms], eval[:ms], eval2[:ms], eval4[:ms], eval5[:ms], eval5q[:ms], eval6[:ms], mcts[:ms], mcts-eval[:ms], mcts2[:ms], beam-mcts[:ms], azero[:ms], net-eval[:ms], nnue-eval[:ms], plodding");
            std::process::exit(1);
        }
    }
}

fn run_game(left_player: Box<dyn Player>, right_player: Box<dyn Player>) {
    let mut board = Board::new();
    let players = [left_player, right_player];

    println!("=== Phutball Game ===");
    println!("Left tries to get the ball to the RIGHT (column 19)");
    println!("Right tries to get the ball to the LEFT (column 1)");
    println!("Left plays first.\n");

    loop {
        board.pretty_print_details();

        if let Some(winner) = board.check_winner() {
            println!("\n{:?} has won!", winner);
            break;
        }

        let player_idx = board.moves_made as usize % 2;
        let current_side = board.side_to_move;

        let player_move = players[player_idx].make_move(&board);

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

// ============================================================================
// Tournament
// ============================================================================

struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new() -> Self {
        let t = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        let seed = t.as_secs().wrapping_mul(1_000_000_007)
            .wrapping_add(t.subsec_nanos() as u64);
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    fn new_with_seed(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

fn run_tournament(spec1: &str, spec2: &str, confidence: f64, start_men: usize, max_games: u32) {
    let mut rng = Xorshift64::new();
    let mut w1: u32 = 0;
    let mut w2: u32 = 0;
    let mut ties: u32 = 0;
    let mut game_num: u32 = 0;

    println!("Tournament: {} vs {} (confidence={}, start_men={})", spec1, spec2, confidence, start_men);
    println!();

    loop {
        game_num += 1;
        match play_tournament_game(spec1, spec2, &mut rng, start_men) {
            Some(true)  => w1 += 1,
            Some(false) => w2 += 1,
            None        => ties += 1,
        }

        let p = prob_engine1_better(w1, w2);
        println!(
            "Game {:4}: {} wins={} {} wins={} ties={} | P(e1>e2)={:.3}",
            game_num, spec1, w1, spec2, w2, ties, p
        );

        if max_games > 0 && game_num >= max_games {
            println!("\nResult after {} games: {} wins={} {} wins={} ties={} | P(e1>e2)={:.3}",
                game_num, spec1, w1, spec2, w2, ties, p);
            return;
        }

        if p > confidence {
            println!("\nResult: {} is stronger (P={:.3} > {})", spec1, p, confidence);
            return;
        }
        if p < 1.0 - confidence {
            println!("\nResult: {} is stronger (P={:.3} < 1-{})", spec2, 1.0 - p, confidence);
            return;
        }
    }
}

/// Returns Some(true) if engine1 won, Some(false) if engine2 won, None if tie.
fn play_tournament_game(spec1: &str, spec2: &str, rng: &mut Xorshift64, start_men: usize) -> Option<bool> {
    let engine1_side = if rng.next() % 2 == 0 { Side::Left } else { Side::Right };

    let (left_spec, right_spec) = match engine1_side {
        Side::Left  => (spec1, spec2),
        Side::Right => (spec2, spec1),
    };

    let left_player  = parse_player(left_spec,  false);
    let right_player = parse_player(right_spec, false);

    let mut board = Board::new();
    let mut placed = 0;
    while placed < start_men {
        let row = (rng.next() % WIDTH as u64) as usize;
        let col = (rng.next() % LENGTH as u64) as usize;
        let pos = Position::new(row, col);
        if board.get(pos) == Piece::Empty {
            board.set(pos, Piece::Man);
            placed += 1;
        }
    }

    let players: [Box<dyn Player>; 2] = [left_player, right_player];

    for _ in 0..100 {
        if let Some(game_winner) = board.check_winner() {
            return Some(game_winner == engine1_side);
        }
        let idx = board.moves_made as usize % 2;
        let mv = players[idx].make_move(&board);
        let all_moves = board.get_all_moves();
        board = match all_moves.get(&mv) {
            Some(b) => b.clone(),
            None    => match all_moves.values().next() {
                Some(b) => b.clone(),
                None    => return None,
            },
        };
    }

    board.check_winner().map(|w| w == engine1_side)
}

fn prob_engine1_better(w1: u32, w2: u32) -> f64 {
    // P(θ > 0.5) where θ ~ Beta(w1+1, w2+1)
    let a = w1 as f64 + 1.0;
    let b = w2 as f64 + 1.0;
    1.0 - ibeta(0.5, a, b)
}

// ============================================================================
// Math: lgamma, incomplete beta (Numerical Recipes continued-fraction method)
// ============================================================================

fn lgamma(z: f64) -> f64 {
    if z < 0.5 {
        (std::f64::consts::PI / (std::f64::consts::PI * z).sin()).ln() - lgamma(1.0 - z)
    } else {
        let z = z - 1.0;
        let c: [f64; 9] = [
            0.999_999_999_999_809_93,
            676.520_368_121_885_1,
            -1259.139_216_722_402_8,
            771.323_428_777_653_13,
            -176.615_029_162_140_59,
            12.507_343_278_686_905,
            -0.138_571_095_265_720_12,
            9.984_369_578_019_571_6e-6,
            1.505_632_735_149_311_6e-7,
        ];
        let mut sum = c[0];
        for (i, &ci) in c[1..].iter().enumerate() {
            sum += ci / (z + (i + 1) as f64);
        }
        let t = z + 7.5; // g = 7, so z + g + 0.5 = z + 7.5
        0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + sum.ln()
    }
}

/// Lentz continued-fraction evaluation for the incomplete beta function.
/// Computes the CF term used by ibeta when x < (a+1)/(a+b+2).
fn ibeta_cf(a: f64, b: f64, x: f64) -> f64 {
    const MAX_ITER: usize = 200;
    const EPS: f64 = 3.0e-7;
    const FP_MIN: f64 = 1.0e-30;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0_f64;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FP_MIN { d = FP_MIN; }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAX_ITER {
        let mf = m as f64;
        let m2 = 2.0 * mf;

        // Even step
        let aa = mf * (b - mf) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d; if d.abs() < FP_MIN { d = FP_MIN; }
        c = 1.0 + aa / c; if c.abs() < FP_MIN { c = FP_MIN; }
        d = 1.0 / d;
        h *= d * c;

        // Odd step
        let aa = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d; if d.abs() < FP_MIN { d = FP_MIN; }
        c = 1.0 + aa / c; if c.abs() < FP_MIN { c = FP_MIN; }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < EPS { break; }
    }

    h
}

/// Regularized incomplete beta function I_x(a, b).
fn ibeta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    if x >= 1.0 { return 1.0; }

    let bt = ((lgamma(a + b) - lgamma(a) - lgamma(b))
        + a * x.ln() + b * (1.0 - x).ln()).exp();

    if x < (a + 1.0) / (a + b + 2.0) {
        bt * ibeta_cf(a, b, x) / a
    } else {
        1.0 - bt * ibeta_cf(b, a, 1.0 - x) / b
    }
}

// ============================================================================
// Board
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    Left,
    Right,
}

impl Side {
    fn flip(&self) -> Self {
        match self {
            Side::Left  => Side::Right,
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

    fn checked_add(&self, delta: (i32, i32)) -> Option<Self> {
        let new_row = self.row as i32 + delta.0;
        let new_col = self.col as i32 + delta.1;
        if new_row >= 0 && new_col >= 0 {
            Some(Position::new(new_row as usize, new_col as usize))
        } else {
            None
        }
    }

    fn to_notation(&self) -> String {
        let row_char = (b'A' + self.row as u8) as char;
        format!("{}{}", row_char, self.col + 1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Direction {
    N, S, E, W, NE, NW, SE, SW,
}

impl Direction {
    const ALL: [Direction; 8] = [
        Direction::N, Direction::S, Direction::E, Direction::W,
        Direction::NE, Direction::NW, Direction::SE, Direction::SW,
    ];

    fn delta(&self) -> (i32, i32) {
        match self {
            Direction::N  => (-1,  0),
            Direction::S  => ( 1,  0),
            Direction::E  => ( 0,  1),
            Direction::W  => ( 0, -1),
            Direction::NE => (-1,  1),
            Direction::NW => (-1, -1),
            Direction::SE => ( 1,  1),
            Direction::SW => ( 1, -1),
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

    pub fn get_all_moves(&self) -> HashMap<String, Board> {
        let mut moves = self.get_man_moves();
        moves.extend(self.get_ball_moves());
        moves
    }

    pub fn get_all_nearby_moves(&self) -> HashMap<String, Board> {
        let mut moves = self.get_nearby_man_moves();
        moves.extend(self.get_ball_moves());
        moves
    }

    fn get_nearby_man_moves(&self) -> HashMap<String, Board> {
        let mut moves = HashMap::new();
        let mut used = std::collections::HashSet::new();

        for row in 0..WIDTH {
            for col in 0..LENGTH {
                let pos = Position::new(row, col);
                if self.get(pos) != Piece::Empty {
                    for dr in -2..=2i32 {
                        for dc in -2..=2i32 {
                            if let Some(nearby) = pos.checked_add((dr, dc)) {
                                if nearby.is_on_board()
                                    && self.get(nearby) == Piece::Empty
                                    && !used.contains(&nearby)
                                {
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

    pub fn check_winner(&self) -> Option<Side> {
        if self.ball_at.col <= 0 {
            Some(Side::Right)
        } else if self.ball_at.col >= LENGTH - 1 {
            Some(Side::Left)
        } else {
            None
        }
    }

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

    fn get_ball_moves(&self) -> HashMap<String, Board> {
        let mut moves = HashMap::new();
        self.get_ball_moves_recursive(&mut moves, String::new());
        moves
    }

    fn get_ball_boards(&self) -> Vec<Board> {
        let mut boards = Vec::new();
        self.get_ball_boards_recursive(&mut boards);
        boards
    }

    fn get_ball_boards_recursive(&self, boards: &mut Vec<Board>) {
        if !self.ball_at.is_on_board() {
            return;
        }
        for direction in Direction::ALL {
            let delta = direction.delta();
            let first_pos = match self.ball_at.checked_add(delta) {
                Some(p) => p,
                None    => continue,
            };
            if !first_pos.is_on_board() || self.get(first_pos) != Piece::Man {
                continue;
            }
            let mut jump_length = 1;
            let (end_row, end_col): (i32, i32) = loop {
                let new_row = self.ball_at.row as i32 + delta.0 * (jump_length + 1);
                let new_col = self.ball_at.col as i32 + delta.1 * (jump_length + 1);
                if new_row >= 0 && new_col >= 0 {
                    let next_pos = Position::new(new_row as usize, new_col as usize);
                    if next_pos.is_on_board() && self.get(next_pos) == Piece::Man {
                        jump_length += 1;
                        continue;
                    }
                }
                break (new_row, new_col);
            };
            let jumped_off_col = end_col < 0 || end_col >= LENGTH as i32;
            let jumped_off_row = end_row < 0 || end_row >= WIDTH as i32;
            if jumped_off_row && !jumped_off_col {
                continue;
            }
            let jumped_into_goal = jumped_off_col;
            let end_point = Position::new(end_row.max(0) as usize, end_col.max(0) as usize);
            let mut new_board = self.clone();
            for i in 1..=jump_length {
                if let Some(pos) = self.ball_at.checked_add((delta.0 * i, delta.1 * i)) {
                    if pos.is_on_board() {
                        new_board.set(pos, Piece::Empty);
                    }
                }
            }
            new_board.set(self.ball_at, Piece::Empty);
            if !jumped_into_goal {
                new_board.set(end_point, Piece::Ball);
            }
            new_board.ball_at = end_point;
            let landed_on_goal_col = end_point.col == 0 || end_point.col == LENGTH - 1;
            if !jumped_into_goal && !landed_on_goal_col {
                new_board.get_ball_boards_recursive(boards);
            }
            let mut final_board = new_board;
            final_board.increment();
            boards.push(final_board);
        }
    }

    fn get_nearby_boards(&self) -> Vec<Board> {
        let mut boards = Vec::new();
        let mut used = [[false; LENGTH]; WIDTH];
        for row in 0..WIDTH {
            for col in 0..LENGTH {
                let pos = Position::new(row, col);
                if self.get(pos) != Piece::Empty {
                    for dr in -2i32..=2 {
                        for dc in -2i32..=2 {
                            let nr = pos.row as i32 + dr;
                            let nc = pos.col as i32 + dc;
                            if nr >= 0 && nc >= 0 {
                                let nr = nr as usize;
                                let nc = nc as usize;
                                if nr < WIDTH && nc < LENGTH
                                    && self.array[nr][nc] == Piece::Empty
                                    && !used[nr][nc]
                                {
                                    let mut new_board = self.clone();
                                    new_board.array[nr][nc] = Piece::Man;
                                    new_board.increment();
                                    boards.push(new_board);
                                    used[nr][nc] = true;
                                }
                            }
                        }
                    }
                }
            }
        }
        boards.extend(self.get_ball_boards());
        boards
    }

    fn get_ball_moves_recursive(&self, moves: &mut HashMap<String, Board>, prefix: String) {
        if !self.ball_at.is_on_board() {
            return;
        }

        for direction in Direction::ALL {
            let delta = direction.delta();

            let first_pos = match self.ball_at.checked_add(delta) {
                Some(p) => p,
                None    => continue,
            };

            if !first_pos.is_on_board() || self.get(first_pos) != Piece::Man {
                continue;
            }

            let mut jump_length = 1;
            let (end_row, end_col): (i32, i32) = loop {
                let new_row = self.ball_at.row as i32 + delta.0 * (jump_length + 1);
                let new_col = self.ball_at.col as i32 + delta.1 * (jump_length + 1);

                if new_row >= 0 && new_col >= 0 {
                    let next_pos = Position::new(new_row as usize, new_col as usize);
                    if next_pos.is_on_board() && self.get(next_pos) == Piece::Man {
                        jump_length += 1;
                        continue;
                    }
                }
                break (new_row, new_col);
            };

            let jumped_off_col = end_col < 0 || end_col >= LENGTH as i32;
            let jumped_off_row = end_row < 0 || end_row >= WIDTH as i32;

            if jumped_off_row && !jumped_off_col {
                continue;
            }

            let jumped_into_goal = jumped_off_col;
            let end_point = Position::new(
                end_row.max(0) as usize,
                end_col.max(0) as usize,
            );

            let mut new_board = self.clone();

            for i in 1..=jump_length {
                if let Some(pos) = self.ball_at.checked_add((delta.0 * i, delta.1 * i)) {
                    if pos.is_on_board() {
                        new_board.set(pos, Piece::Empty);
                    }
                }
            }

            new_board.set(self.ball_at, Piece::Empty);

            if !jumped_into_goal {
                new_board.set(end_point, Piece::Ball);
            }
            new_board.ball_at = end_point;

            let move_name = format!("{}{} ", prefix, direction);
            let final_board = {
                let mut b = new_board.clone();
                b.increment();
                b
            };
            moves.insert(move_name.clone(), final_board);

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

    fn clone_with_side_flip(&self) -> Board {
        let mut b = self.clone();
        b.side_to_move = b.side_to_move.flip();
        b
    }

    pub fn pretty_string_details(&self) -> String {
        let mut output = String::new();
        output.push_str("          1111111111\n");
        output.push_str(" 1234567890123456789\n");

        for row in 0..WIDTH {
            output.push((b'A' + row as u8) as char);
            for col in 0..LENGTH {
                let pos = Position::new(row, col);
                match self.get(pos) {
                    Piece::Man   => output.push(WHITE_CIRCLE),
                    Piece::Ball  => output.push(BLACK_CIRCLE),
                    Piece::Empty => output.push('+'),
                }
            }
            output.push('\n');
        }

        output.push_str(&format!(
            "Side to move: {:?}\nMoves made: {}\nBall at: {}\n",
            self.side_to_move, self.moves_made, self.ball_at.to_notation()
        ));
        output
    }

    pub fn pretty_string(&self) -> String {
        let mut output = String::new();
        for row in 0..WIDTH {
            for col in 0..LENGTH {
                let pos = Position::new(row, col);
                match self.get(pos) {
                    Piece::Man   => output.push(WHITE_CIRCLE),
                    Piece::Ball  => output.push(BLACK_CIRCLE),
                    Piece::Empty => output.push('+'),
                }
            }
            output.push('\n');
        }
        output
    }

    pub fn pretty_print(&self) {
        println!("{}", self.pretty_string());
    }

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

        if let Some((jump_move, _)) = moves.iter().find(|(k, _)| k.contains(' ')) {
            println!("Example jump move: {}", jump_move.trim());
        }
        if let Some((place_move, _)) = moves.iter().find(|(k, _)| !k.contains(' ')) {
            println!("Example placement move: {}", place_move);
        }

        loop {
            print!("Enter your move: ");
            io::stdout().flush().unwrap();

            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            let input = input.trim();

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

        let preferred_dir = match board.side_to_move {
            Side::Left  => "E ",
            Side::Right => "W ",
        };
        if moves.contains_key(preferred_dir) {
            return preferred_dir.to_string();
        }

        let offset = match board.side_to_move {
            Side::Left  =>  1,
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

        moves.keys().next().unwrap().clone()
    }
}

// ----------------------------------------------------------------------------
// Minimax Player
// ----------------------------------------------------------------------------

struct MinimaxPlayer {
    depth:   u32,
    verbose: bool,
}

impl MinimaxPlayer {
    fn new(depth: u32, verbose: bool) -> Self {
        Self { depth, verbose }
    }

    pub fn evaluate(&self, board: &Board) -> f64 {
        1.0 - self.negamax(board, self.depth - 1)
    }

    fn negamax(&self, board: &Board, depth: u32) -> f64 {
        if let Some(winner) = board.check_winner() {
            return if winner == board.side_to_move { 1.0 } else { 0.0 };
        }
        if depth == 0 {
            return LocationEvaluator::score(board);
        }
        let moves = board.get_all_nearby_moves();
        let mut max_score = 0.0_f64;
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
        if self.verbose {
            println!("Minimax thinking (depth {})...", self.depth);
        }

        let moves = board.get_all_nearby_moves();
        let mut best_move  = moves.keys().next().unwrap().clone();
        let mut best_score = 0.0_f64;

        for (move_name, next_board) in moves {
            let score = 1.0 - self.negamax(&next_board, self.depth - 1);
            if score > best_score || (score == best_score && move_name < best_move) {
                best_score = score;
                best_move  = move_name;
            }
        }

        if self.verbose {
            println!("Minimax chose: {} (score: {:.3})", best_move.trim(), best_score);
        }
        best_move
    }
}

// ----------------------------------------------------------------------------
// Alpha-Beta Player
// ----------------------------------------------------------------------------

struct AlphaBetaPlayer {
    depth:   u32,
    verbose: bool,
}

impl AlphaBetaPlayer {
    fn new(depth: u32, verbose: bool) -> Self {
        Self { depth, verbose }
    }

    pub fn evaluate(&self, board: &Board) -> f64 {
        1.0 - self.negamax_ab(board, self.depth - 1, 0.0, 1.0)
    }

    fn negamax_ab(&self, board: &Board, depth: u32, mut alpha: f64, beta: f64) -> f64 {
        if let Some(winner) = board.check_winner() {
            return if winner == board.side_to_move { 1.0 } else { 0.0 };
        }
        if depth == 0 {
            return LocationEvaluator::score(board);
        }
        let moves = board.get_all_nearby_moves();
        let mut max_score = 0.0_f64;
        for (_, next_board) in moves {
            let score = 1.0 - self.negamax_ab(&next_board, depth - 1, 1.0 - beta, 1.0 - alpha);
            if score > max_score {
                max_score = score;
            }
            alpha = alpha.max(score);
            if alpha >= beta {
                break;
            }
        }
        max_score
    }
}

impl Player for AlphaBetaPlayer {
    fn make_move(&self, board: &Board) -> String {
        if self.verbose {
            println!("Alpha-Beta thinking (depth {})...", self.depth);
        }

        let moves = board.get_all_nearby_moves();
        let mut best_move  = moves.keys().next().unwrap().clone();
        let mut best_score = 0.0_f64;

        for (move_name, next_board) in moves {
            let score = 1.0 - self.negamax_ab(&next_board, self.depth - 1, 0.0, 1.0);
            if score > best_score || (score == best_score && move_name < best_move) {
                best_score = score;
                best_move  = move_name;
            }
        }

        if self.verbose {
            println!("Alpha-Beta chose: {} (score: {:.3})", best_move.trim(), best_score);
        }
        best_move
    }
}

// ----------------------------------------------------------------------------
// Parallel Alpha-Beta Player
// ----------------------------------------------------------------------------

struct ParallelAlphaBetaPlayer {
    depth: u32,
}

impl ParallelAlphaBetaPlayer {
    fn new(depth: u32) -> Self {
        Self { depth }
    }

    pub fn evaluate(&self, board: &Board) -> f64 {
        1.0 - self.negamax_ab(board, self.depth - 1, 0.0, 1.0)
    }

    fn negamax_ab(&self, board: &Board, depth: u32, mut alpha: f64, beta: f64) -> f64 {
        if let Some(winner) = board.check_winner() {
            return if winner == board.side_to_move { 1.0 } else { 0.0 };
        }
        if depth == 0 {
            return LocationEvaluator::score(board);
        }
        let moves = board.get_all_nearby_moves();
        let mut max_score = 0.0_f64;
        for (_, next_board) in moves {
            let score = 1.0 - self.negamax_ab(&next_board, depth - 1, 1.0 - beta, 1.0 - alpha);
            if score > max_score {
                max_score = score;
            }
            alpha = alpha.max(score);
            if alpha >= beta {
                break;
            }
        }
        max_score
    }
}

impl Player for ParallelAlphaBetaPlayer {
    fn make_move(&self, board: &Board) -> String {
        println!("Parallel Alpha-Beta thinking (depth {})...", self.depth);

        let moves = board.get_all_nearby_moves();

        let mut move_scores: Vec<(String, f64)> = moves
            .par_iter()
            .map(|(move_name, next_board)| {
                let score = 1.0 - self.negamax_ab(next_board, self.depth - 1, 0.0, 1.0);
                (move_name.clone(), score)
            })
            .collect();

        move_scores.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap().then_with(|| a.0.cmp(&b.0))
        });

        let (best_move, best_score) = &move_scores[0];
        println!("Parallel Alpha-Beta chose: {} (score: {:.3})", best_move.trim(), best_score);
        best_move.clone()
    }
}

// ----------------------------------------------------------------------------
// Timed Player (iterative-deepening alpha-beta with wall-clock deadline)
// ----------------------------------------------------------------------------

struct TimedPlayer {
    ms:      u64,
    verbose: bool,
    eval_fn: fn(&Board) -> f64,
}

impl TimedPlayer {
    fn new(ms: u64, verbose: bool) -> Self {
        Self { ms, verbose, eval_fn: LocationEvaluator::score }
    }

    fn with_eval(ms: u64, verbose: bool, eval_fn: fn(&Board) -> f64) -> Self {
        Self { ms, verbose, eval_fn }
    }

    /// Returns None on timeout, otherwise the negamax score from board.side_to_move's perspective.
    fn negamax_timed(
        &self,
        board:    &Board,
        depth:    u32,
        mut alpha: f64,
        beta:     f64,
        deadline: Instant,
        nodes:    &mut u32,
    ) -> Option<f64> {
        *nodes += 1;
        // Check deadline every 1000 nodes to amortize Instant::now() cost.
        if *nodes % 1000 == 0 && Instant::now() >= deadline {
            return None;
        }

        if let Some(winner) = board.check_winner() {
            return Some(if winner == board.side_to_move { 1.0 } else { 0.0 });
        }
        if depth == 0 {
            return Some((self.eval_fn)(board));
        }

        let moves = board.get_nearby_boards();
        let mut max_score = 0.0_f64;

        for next_board in &moves {
            let score = match self.negamax_timed(
                next_board, depth - 1,
                1.0 - beta, 1.0 - alpha,
                deadline, nodes,
            ) {
                Some(s) => 1.0 - s,
                None    => return None,
            };
            if score > max_score {
                max_score = score;
            }
            alpha = alpha.max(score);
            if alpha >= beta {
                break;
            }
        }

        Some(max_score)
    }
}

impl Player for TimedPlayer {
    fn make_move(&self, board: &Board) -> String {
        let deadline = Instant::now() + Duration::from_millis(self.ms);
        let moves = board.get_all_nearby_moves();

        if moves.is_empty() {
            return String::new();
        }

        // Safe fallback in case depth-1 times out immediately.
        let mut best_move = moves.keys().next().unwrap().clone();
        let mut depth = 1u32;

        loop {
            let mut nodes: u32 = 0;
            let mut depth_best_move  = String::new();
            let mut depth_best_score = f64::NEG_INFINITY;
            let mut timed_out = false;
            let depth_start = Instant::now();

            for (move_name, next_board) in &moves {
                match self.negamax_timed(
                    next_board, depth - 1,
                    0.0, 1.0,
                    deadline, &mut nodes,
                ) {
                    Some(s) => {
                        let score = 1.0 - s;
                        if score > depth_best_score
                            || (score == depth_best_score && *move_name < depth_best_move)
                        {
                            depth_best_score = score;
                            depth_best_move  = move_name.clone();
                        }
                    }
                    None => {
                        timed_out = true;
                        break;
                    }
                }
            }

            if !timed_out {
                best_move = depth_best_move;
                if self.verbose {
                    eprintln!(
                        "timed: depth={} in {}ms (nodes={}, score={:.3})",
                        depth, depth_start.elapsed().as_millis(), nodes, depth_best_score
                    );
                }
                depth += 1;
            } else {
                break;
            }

            if Instant::now() >= deadline {
                break;
            }
        }

        best_move
    }
}

// ============================================================================
// Zobrist Hashing + Transposition Table
// ============================================================================

struct ZobristTable {
    man_at:    [[u64; LENGTH]; WIDTH],
    ball_at:   [[u64; LENGTH]; WIDTH],
    side_left: u64,
}

impl ZobristTable {
    fn new() -> Self {
        let mut rng = Xorshift64::new_with_seed(0x9e3779b97f4a7c15);
        let mut man_at  = [[0u64; LENGTH]; WIDTH];
        let mut ball_at = [[0u64; LENGTH]; WIDTH];
        for row in 0..WIDTH {
            for col in 0..LENGTH {
                man_at[row][col]  = rng.next();
                ball_at[row][col] = rng.next();
            }
        }
        Self { man_at, ball_at, side_left: rng.next() }
    }
}

fn zobrist_hash(board: &Board, zt: &ZobristTable) -> u64 {
    let mut h = 0u64;
    for row in 0..WIDTH {
        for col in 0..LENGTH {
            match board.array[row][col] {
                Piece::Man  => h ^= zt.man_at[row][col],
                Piece::Ball => h ^= zt.ball_at[row][col],
                Piece::Empty => {}
            }
        }
    }
    if board.side_to_move == Side::Left {
        h ^= zt.side_left;
    }
    h
}

static ZOBRIST: std::sync::OnceLock<ZobristTable> = std::sync::OnceLock::new();

fn get_zobrist() -> &'static ZobristTable {
    ZOBRIST.get_or_init(ZobristTable::new)
}

const TT_SIZE: usize = 1 << 17;

#[derive(Clone, Copy)]
enum NodeType { Exact, LowerBound, UpperBound }

#[derive(Clone)]
struct TTEntry {
    hash:           u64,
    depth:          u32,
    score:          f64,
    node_type:      NodeType,
    best_next_hash: u64,
}

struct TTable {
    entries: Vec<Option<TTEntry>>,
}

impl TTable {
    fn new() -> Self {
        Self { entries: vec![None; TT_SIZE] }
    }

    fn probe(&self, hash: u64) -> Option<&TTEntry> {
        let idx = (hash as usize) & (TT_SIZE - 1);
        self.entries[idx].as_ref().filter(|e| e.hash == hash)
    }

    fn store(&mut self, hash: u64, depth: u32, score: f64, node_type: NodeType, best_next_hash: u64) {
        let idx = (hash as usize) & (TT_SIZE - 1);
        self.entries[idx] = Some(TTEntry { hash, depth, score, node_type, best_next_hash });
    }
}

// ============================================================================
// Eval2: TimedPlayer with TTable + move ordering
// ============================================================================

struct TimedPlayer2 {
    ms:      u64,
    verbose: bool,
    eval_fn: fn(&Board) -> f64,
}

impl TimedPlayer2 {
    fn new(ms: u64, verbose: bool) -> Self {
        Self { ms, verbose, eval_fn: RichEvaluator::score }
    }

    fn with_eval(ms: u64, verbose: bool, eval_fn: fn(&Board) -> f64) -> Self {
        Self { ms, verbose, eval_fn }
    }

    /// Negamax with alpha-beta and transposition table.
    /// Hash is computed once per node (not per child) to avoid O(285*children) overhead.
    fn negamax2(
        &self,
        board:     &Board,
        depth:     u32,
        mut alpha: f64,
        mut beta:  f64,
        deadline:  Instant,
        nodes:     &mut u64,
        tt:        &mut TTable,
        zt:        &ZobristTable,
    ) -> Option<f64> {
        *nodes += 1;
        if *nodes % 1000 == 0 && Instant::now() >= deadline {
            return None;
        }

        if let Some(winner) = board.check_winner() {
            return Some(if winner == board.side_to_move { 1.0 } else { 0.0 });
        }
        if depth == 0 {
            return Some((self.eval_fn)(board));
        }

        let hash = zobrist_hash(board, zt);
        let alpha_orig = alpha;

        if let Some(entry) = tt.probe(hash) {
            if entry.depth >= depth {
                match entry.node_type {
                    NodeType::Exact      => return Some(entry.score),
                    NodeType::LowerBound => alpha = alpha.max(entry.score),
                    NodeType::UpperBound => beta  = beta.min(entry.score),
                }
                if alpha >= beta { return Some(entry.score); }
            }
        }

        // Generate children; try ball-jump moves before man placements.
        let ball_pos = board.ball_at;
        let all_children = board.get_nearby_boards();
        let (jumps, placements): (Vec<Board>, Vec<Board>) =
            all_children.into_iter().partition(|b| b.ball_at != ball_pos);

        let ordered: Vec<&Board> = jumps.iter().chain(placements.iter()).collect();
        let mut max_score = 0.0_f64;
        let mut found_any = false;
        let mut best_idx: usize = 0;

        for (i, next_board) in ordered.iter().enumerate() {
            let score = match self.negamax2(
                next_board, depth - 1,
                1.0 - beta, 1.0 - alpha,
                deadline, nodes, tt, zt,
            ) {
                Some(s) => 1.0 - s,
                None    => return None,
            };
            if !found_any || score > max_score {
                max_score = score;
                found_any = true;
                best_idx = i;
            }
            alpha = alpha.max(score);
            if alpha >= beta {
                break;
            }
        }

        let best_next_hash = if found_any { zobrist_hash(ordered[best_idx], zt) } else { 0 };
        let node_type = if max_score <= alpha_orig {
            NodeType::UpperBound
        } else if alpha >= beta {
            NodeType::LowerBound
        } else {
            NodeType::Exact
        };
        tt.store(hash, depth, max_score, node_type, best_next_hash);

        Some(max_score)
    }
}

impl Player for TimedPlayer2 {
    fn make_move(&self, board: &Board) -> String {
        let deadline = Instant::now() + Duration::from_millis(self.ms);
        let zt = get_zobrist();
        let moves_map = board.get_all_nearby_moves();

        if moves_map.is_empty() {
            return String::new();
        }

        // Collect into Vec; best_move from depth D goes first in depth D+1.
        let mut moves: Vec<(String, Board)> =
            moves_map.into_iter().collect();
        let mut best_move = moves[0].0.clone();
        let mut depth = 1u32;
        let mut tt = TTable::new();

        loop {
            let mut nodes: u64 = 0;
            let mut depth_best_idx   = 0usize;
            let mut depth_best_score = f64::NEG_INFINITY;
            let mut timed_out = false;
            let depth_start = Instant::now();

            for (idx, (move_name, next_board)) in moves.iter().enumerate() {
                match self.negamax2(
                    next_board, depth - 1,
                    0.0, 1.0,
                    deadline, &mut nodes, &mut tt, zt,
                ) {
                    Some(s) => {
                        let score = 1.0 - s;
                        if score > depth_best_score
                            || (score == depth_best_score && *move_name < moves[depth_best_idx].0)
                        {
                            depth_best_score = score;
                            depth_best_idx   = idx;
                        }
                    }
                    None => {
                        timed_out = true;
                        break;
                    }
                }
            }

            if !timed_out {
                best_move = moves[depth_best_idx].0.clone();
                // Promote best move to front for better ordering next iteration.
                moves.swap(0, depth_best_idx);
                if self.verbose {
                    eprintln!(
                        "eval2: depth={} in {}ms (nodes={}, score={:.3})",
                        depth, depth_start.elapsed().as_millis(), nodes, depth_best_score
                    );
                }
                depth += 1;
            } else {
                break;
            }

            if Instant::now() >= deadline {
                break;
            }
        }

        best_move
    }
}

// ============================================================================
// Eval5: TimedPlayer2 with aspiration windows
// ============================================================================

struct TimedPlayer5 {
    ms:      u64,
    verbose: bool,
    eval_fn: fn(&Board) -> f64,
}

impl TimedPlayer5 {
    fn with_eval(ms: u64, verbose: bool, eval_fn: fn(&Board) -> f64) -> Self {
        Self { ms, verbose, eval_fn }
    }

    fn negamax(
        &self,
        board:     &Board,
        depth:     u32,
        mut alpha: f64,
        mut beta:  f64,
        deadline:  Instant,
        nodes:     &mut u64,
        tt:        &mut TTable,
        zt:        &ZobristTable,
    ) -> Option<f64> {
        *nodes += 1;
        if *nodes % 1000 == 0 && Instant::now() >= deadline {
            return None;
        }

        if let Some(winner) = board.check_winner() {
            return Some(if winner == board.side_to_move { 1.0 } else { 0.0 });
        }
        if depth == 0 {
            return Some((self.eval_fn)(board));
        }

        let hash = zobrist_hash(board, zt);
        let alpha_orig = alpha;

        if let Some(entry) = tt.probe(hash) {
            if entry.depth >= depth {
                match entry.node_type {
                    NodeType::Exact      => return Some(entry.score),
                    NodeType::LowerBound => alpha = alpha.max(entry.score),
                    NodeType::UpperBound => beta  = beta.min(entry.score),
                }
                if alpha >= beta { return Some(entry.score); }
            }
        }

        let ball_pos = board.ball_at;
        let all_children = board.get_nearby_boards();
        let (jumps, placements): (Vec<Board>, Vec<Board>) =
            all_children.into_iter().partition(|b| b.ball_at != ball_pos);

        let ordered: Vec<&Board> = jumps.iter().chain(placements.iter()).collect();
        let mut max_score = 0.0_f64;
        let mut found_any = false;
        let mut best_idx: usize = 0;

        for (i, next_board) in ordered.iter().enumerate() {
            let score = match self.negamax(
                next_board, depth - 1,
                1.0 - beta, 1.0 - alpha,
                deadline, nodes, tt, zt,
            ) {
                Some(s) => 1.0 - s,
                None    => return None,
            };
            if !found_any || score > max_score {
                max_score = score;
                found_any = true;
                best_idx = i;
            }
            alpha = alpha.max(score);
            if alpha >= beta {
                break;
            }
        }

        let best_next_hash = if found_any { zobrist_hash(ordered[best_idx], zt) } else { 0 };
        let node_type = if max_score <= alpha_orig {
            NodeType::UpperBound
        } else if alpha >= beta {
            NodeType::LowerBound
        } else {
            NodeType::Exact
        };
        tt.store(hash, depth, max_score, node_type, best_next_hash);

        Some(max_score)
    }
}

impl Player for TimedPlayer5 {
    fn make_move(&self, board: &Board) -> String {
        let deadline = Instant::now() + Duration::from_millis(self.ms);
        let zt = get_zobrist();
        let moves_map = board.get_all_nearby_moves();

        if moves_map.is_empty() {
            return String::new();
        }

        let mut moves: Vec<(String, Board)> = moves_map.into_iter().collect();
        let mut best_move = moves[0].0.clone();
        let mut depth = 1u32;
        let mut tt = TTable::new();
        let mut prev_score = 0.5_f64;
        const DELTA: f64 = 0.15;

        'depth: loop {
            let (mut alpha, mut beta) = if depth < 2 {
                (0.0_f64, 1.0_f64)
            } else {
                ((prev_score - DELTA).max(0.0), (prev_score + DELTA).min(1.0))
            };

            loop {
                let mut nodes: u64 = 0;
                let mut depth_best_idx   = 0usize;
                let mut depth_best_score = f64::NEG_INFINITY;
                let mut timed_out = false;
                let depth_start = Instant::now();

                for (idx, (_move_name, next_board)) in moves.iter().enumerate() {
                    match self.negamax(
                        next_board, depth - 1,
                        1.0 - beta, 1.0 - alpha,
                        deadline, &mut nodes, &mut tt, zt,
                    ) {
                        Some(s) => {
                            let score = 1.0 - s;
                            if score > depth_best_score
                                || (score == depth_best_score
                                    && moves[idx].0 < moves[depth_best_idx].0)
                            {
                                depth_best_score = score;
                                depth_best_idx   = idx;
                            }
                        }
                        None => { timed_out = true; break; }
                    }
                }

                if timed_out { break 'depth; }

                let at_full_window = alpha <= 0.0 && beta >= 1.0;
                if depth_best_score <= alpha && !at_full_window {
                    alpha = (alpha - DELTA).max(0.0);
                } else if depth_best_score >= beta && !at_full_window {
                    beta = (beta + DELTA).min(1.0);
                } else {
                    best_move = moves[depth_best_idx].0.clone();
                    moves.swap(0, depth_best_idx);
                    prev_score = depth_best_score;
                    if self.verbose {
                        eprintln!(
                            "eval5: depth={} in {}ms (nodes={}, score={:.3})",
                            depth,
                            depth_start.elapsed().as_millis(),
                            nodes,
                            depth_best_score,
                        );
                    }
                    depth += 1;
                    break;
                }

                if Instant::now() >= deadline { break 'depth; }
            }

            if Instant::now() >= deadline { break; }
        }

        best_move
    }
}

impl TimedPlayer5 {
    fn score_position(&self, board: &Board) -> f64 {
        let deadline = Instant::now() + Duration::from_millis(self.ms);
        let zt = get_zobrist();
        let mut tt = TTable::new();
        let mut prev_score = 0.5_f64;
        for depth in 1..=20u32 {
            let mut nodes: u64 = 0;
            match self.negamax(board, depth, 0.0, 1.0, deadline, &mut nodes, &mut tt, zt) {
                Some(s) => { prev_score = s; }
                None => break,
            }
            if Instant::now() >= deadline { break; }
        }
        prev_score
    }
}

// ============================================================================
// Eval5Q: Eval5 + quiescence search for jump sequences
// ============================================================================

const Q_DEPTH: u32 = 3;

struct TimedPlayer5Q {
    ms:      u64,
    verbose: bool,
}

impl TimedPlayer5Q {
    fn new(ms: u64, verbose: bool) -> Self {
        Self { ms, verbose }
    }

    fn qsearch(
        &self,
        board:     &Board,
        mut alpha: f64,
        beta:      f64,
        qdepth:    u32,
        deadline:  Instant,
        nodes:     &mut u64,
    ) -> Option<f64> {
        *nodes += 1;
        if *nodes % 1000 == 0 && Instant::now() >= deadline {
            return None;
        }

        if let Some(winner) = board.check_winner() {
            return Some(if winner == board.side_to_move { 1.0 } else { 0.0 });
        }

        let static_score = JumpChainEvaluator::score(board);
        if qdepth == 0 || static_score >= beta {
            return Some(static_score);
        }
        alpha = alpha.max(static_score);

        for next in board.get_ball_boards() {
            let score = 1.0 - self.qsearch(
                &next, 1.0 - beta, 1.0 - alpha, qdepth - 1, deadline, nodes,
            )?;
            if score > alpha {
                alpha = score;
                if alpha >= beta { break; }
            }
        }
        Some(alpha)
    }

    fn negamax(
        &self,
        board:     &Board,
        depth:     u32,
        mut alpha: f64,
        mut beta:  f64,
        deadline:  Instant,
        nodes:     &mut u64,
        tt:        &mut TTable,
        zt:        &ZobristTable,
    ) -> Option<f64> {
        *nodes += 1;
        if *nodes % 1000 == 0 && Instant::now() >= deadline {
            return None;
        }

        if let Some(winner) = board.check_winner() {
            return Some(if winner == board.side_to_move { 1.0 } else { 0.0 });
        }

        if depth == 0 {
            let static_score = JumpChainEvaluator::score(board);
            let jump_moves = board.get_ball_boards();
            if jump_moves.is_empty() || static_score >= beta {
                return Some(static_score);
            }
            return self.qsearch(board, alpha, beta, Q_DEPTH, deadline, nodes);
        }

        let hash = zobrist_hash(board, zt);
        let alpha_orig = alpha;

        if let Some(entry) = tt.probe(hash) {
            if entry.depth >= depth {
                match entry.node_type {
                    NodeType::Exact      => return Some(entry.score),
                    NodeType::LowerBound => alpha = alpha.max(entry.score),
                    NodeType::UpperBound => beta  = beta.min(entry.score),
                }
                if alpha >= beta { return Some(entry.score); }
            }
        }

        let ball_pos = board.ball_at;
        let all_children = board.get_nearby_boards();
        let (jumps, placements): (Vec<Board>, Vec<Board>) =
            all_children.into_iter().partition(|b| b.ball_at != ball_pos);

        let ordered: Vec<&Board> = jumps.iter().chain(placements.iter()).collect();
        let mut max_score = 0.0_f64;
        let mut found_any = false;
        let mut best_idx: usize = 0;

        for (i, next_board) in ordered.iter().enumerate() {
            let score = match self.negamax(
                next_board, depth - 1,
                1.0 - beta, 1.0 - alpha,
                deadline, nodes, tt, zt,
            ) {
                Some(s) => 1.0 - s,
                None    => return None,
            };
            if !found_any || score > max_score {
                max_score = score;
                found_any = true;
                best_idx = i;
            }
            alpha = alpha.max(score);
            if alpha >= beta { break; }
        }

        let best_next_hash = if found_any { zobrist_hash(ordered[best_idx], zt) } else { 0 };
        let node_type = if max_score <= alpha_orig {
            NodeType::UpperBound
        } else if alpha >= beta {
            NodeType::LowerBound
        } else {
            NodeType::Exact
        };
        tt.store(hash, depth, max_score, node_type, best_next_hash);

        Some(max_score)
    }
}

impl Player for TimedPlayer5Q {
    fn make_move(&self, board: &Board) -> String {
        let deadline = Instant::now() + Duration::from_millis(self.ms);
        let zt = get_zobrist();
        let moves_map = board.get_all_nearby_moves();

        if moves_map.is_empty() {
            return String::new();
        }

        let mut moves: Vec<(String, Board)> = moves_map.into_iter().collect();
        let mut best_move = moves[0].0.clone();
        let mut depth = 1u32;
        let mut tt = TTable::new();
        let mut prev_score = 0.5_f64;
        const DELTA: f64 = 0.15;

        'depth: loop {
            let (mut alpha, mut beta) = if depth < 2 {
                (0.0_f64, 1.0_f64)
            } else {
                ((prev_score - DELTA).max(0.0), (prev_score + DELTA).min(1.0))
            };

            loop {
                let mut nodes: u64 = 0;
                let mut depth_best_idx   = 0usize;
                let mut depth_best_score = f64::NEG_INFINITY;
                let mut timed_out = false;
                let depth_start = Instant::now();

                for (idx, (_move_name, next_board)) in moves.iter().enumerate() {
                    match self.negamax(
                        next_board, depth - 1,
                        1.0 - beta, 1.0 - alpha,
                        deadline, &mut nodes, &mut tt, zt,
                    ) {
                        Some(s) => {
                            let score = 1.0 - s;
                            if score > depth_best_score
                                || (score == depth_best_score
                                    && moves[idx].0 < moves[depth_best_idx].0)
                            {
                                depth_best_score = score;
                                depth_best_idx   = idx;
                            }
                        }
                        None => { timed_out = true; break; }
                    }
                }

                if timed_out { break 'depth; }

                let at_full_window = alpha <= 0.0 && beta >= 1.0;
                if depth_best_score <= alpha && !at_full_window {
                    alpha = (alpha - DELTA).max(0.0);
                } else if depth_best_score >= beta && !at_full_window {
                    beta = (beta + DELTA).min(1.0);
                } else {
                    best_move = moves[depth_best_idx].0.clone();
                    moves.swap(0, depth_best_idx);
                    prev_score = depth_best_score;
                    if self.verbose {
                        eprintln!(
                            "eval5q: depth={} in {}ms (nodes={}, score={:.3})",
                            depth,
                            depth_start.elapsed().as_millis(),
                            nodes,
                            depth_best_score,
                        );
                    }
                    depth += 1;
                    break;
                }

                if Instant::now() >= deadline { break 'depth; }
            }

            if Instant::now() >= deadline { break; }
        }

        best_move
    }
}

// ============================================================================
// Eval6: TimedPlayer5 + beam search (forward pruning)
// ============================================================================

const BEAM_K: usize = 8;

struct TimedPlayer6 {
    ms:      u64,
    verbose: bool,
    eval_fn: fn(&Board) -> f64,
}

impl TimedPlayer6 {
    fn with_eval(ms: u64, verbose: bool, eval_fn: fn(&Board) -> f64) -> Self {
        Self { ms, verbose, eval_fn }
    }

    fn negamax(
        &self,
        board:     &Board,
        depth:     u32,
        mut alpha: f64,
        mut beta:  f64,
        deadline:  Instant,
        nodes:     &mut u64,
        tt:        &mut TTable,
        zt:        &ZobristTable,
    ) -> Option<f64> {
        *nodes += 1;
        if *nodes % 1000 == 0 && Instant::now() >= deadline {
            return None;
        }

        if let Some(winner) = board.check_winner() {
            return Some(if winner == board.side_to_move { 1.0 } else { 0.0 });
        }
        if depth == 0 {
            return Some((self.eval_fn)(board));
        }

        let hash = zobrist_hash(board, zt);
        let alpha_orig = alpha;
        let mut tt_best_hash = 0u64;

        if let Some(entry) = tt.probe(hash) {
            tt_best_hash = entry.best_next_hash;
            if entry.depth >= depth {
                match entry.node_type {
                    NodeType::Exact      => return Some(entry.score),
                    NodeType::LowerBound => alpha = alpha.max(entry.score),
                    NodeType::UpperBound => beta  = beta.min(entry.score),
                }
                if alpha >= beta { return Some(entry.score); }
            }
        }

        let ball_pos = board.ball_at;
        let all_children = board.get_nearby_boards();
        let (jumps, mut placements): (Vec<Board>, Vec<Board>) =
            all_children.into_iter().partition(|b| b.ball_at != ball_pos);

        // Forward pruning: at depth >= 2, prune placements to top BEAM_K by static eval.
        // Always preserve the TT best move and all jumps.
        if depth >= 2 && placements.len() > BEAM_K {
            let tt_best = if tt_best_hash != 0 {
                if let Some(pos) = placements.iter().position(|b| zobrist_hash(b, zt) == tt_best_hash) {
                    Some(placements.swap_remove(pos))
                } else {
                    None
                }
            } else {
                None
            };

            placements.sort_unstable_by(|a, b| {
                JumpChainEvaluator::score(b)
                    .partial_cmp(&JumpChainEvaluator::score(a))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            placements.truncate(BEAM_K);

            if let Some(tt_board) = tt_best {
                placements.push(tt_board);
                let last = placements.len() - 1;
                placements.swap(0, last);
            }
        }

        let ordered: Vec<&Board> = jumps.iter().chain(placements.iter()).collect();
        let mut max_score = 0.0_f64;
        let mut found_any = false;
        let mut best_idx: usize = 0;

        for (i, next_board) in ordered.iter().enumerate() {
            let score = match self.negamax(
                next_board, depth - 1,
                1.0 - beta, 1.0 - alpha,
                deadline, nodes, tt, zt,
            ) {
                Some(s) => 1.0 - s,
                None    => return None,
            };
            if !found_any || score > max_score {
                max_score = score;
                found_any = true;
                best_idx = i;
            }
            alpha = alpha.max(score);
            if alpha >= beta {
                break;
            }
        }

        let best_next_hash = if found_any { zobrist_hash(ordered[best_idx], zt) } else { 0 };
        let node_type = if max_score <= alpha_orig {
            NodeType::UpperBound
        } else if alpha >= beta {
            NodeType::LowerBound
        } else {
            NodeType::Exact
        };
        tt.store(hash, depth, max_score, node_type, best_next_hash);

        Some(max_score)
    }
}

impl Player for TimedPlayer6 {
    fn make_move(&self, board: &Board) -> String {
        let deadline = Instant::now() + Duration::from_millis(self.ms);
        let zt = get_zobrist();
        let moves_map = board.get_all_nearby_moves();

        if moves_map.is_empty() {
            return String::new();
        }

        let mut moves: Vec<(String, Board)> = moves_map.into_iter().collect();
        let mut best_move = moves[0].0.clone();
        let mut depth = 1u32;
        let mut tt = TTable::new();
        let mut prev_score = 0.5_f64;
        const DELTA: f64 = 0.15;

        'depth: loop {
            let (mut alpha, mut beta) = if depth < 2 {
                (0.0_f64, 1.0_f64)
            } else {
                ((prev_score - DELTA).max(0.0), (prev_score + DELTA).min(1.0))
            };

            loop {
                let mut nodes: u64 = 0;
                let mut depth_best_idx   = 0usize;
                let mut depth_best_score = f64::NEG_INFINITY;
                let mut timed_out = false;
                let depth_start = Instant::now();

                for (idx, (_move_name, next_board)) in moves.iter().enumerate() {
                    match self.negamax(
                        next_board, depth - 1,
                        1.0 - beta, 1.0 - alpha,
                        deadline, &mut nodes, &mut tt, zt,
                    ) {
                        Some(s) => {
                            let score = 1.0 - s;
                            if score > depth_best_score
                                || (score == depth_best_score
                                    && moves[idx].0 < moves[depth_best_idx].0)
                            {
                                depth_best_score = score;
                                depth_best_idx   = idx;
                            }
                        }
                        None => { timed_out = true; break; }
                    }
                }

                if timed_out { break 'depth; }

                let at_full_window = alpha <= 0.0 && beta >= 1.0;
                if depth_best_score <= alpha && !at_full_window {
                    alpha = (alpha - DELTA).max(0.0);
                } else if depth_best_score >= beta && !at_full_window {
                    beta = (beta + DELTA).min(1.0);
                } else {
                    best_move = moves[depth_best_idx].0.clone();
                    moves.swap(0, depth_best_idx);
                    prev_score = depth_best_score;
                    if self.verbose {
                        eprintln!(
                            "eval6: depth={} in {}ms (nodes={}, score={:.3})",
                            depth,
                            depth_start.elapsed().as_millis(),
                            nodes,
                            depth_best_score,
                        );
                    }
                    depth += 1;
                    break;
                }

                if Instant::now() >= deadline { break 'depth; }
            }

            if Instant::now() >= deadline { break; }
        }

        best_move
    }
}

// ============================================================================
// Net-guided timed alpha-beta (neural leaf evaluation + policy move ordering)
// ============================================================================

struct NetTimedPlayer {
    net: PhutballNet,
    ms:  u64,
}

impl NetTimedPlayer {
    fn new(ms: u64) -> Self {
        Self { net: PhutballNet::load("weights.bin"), ms }
    }

    fn negamax(
        &self,
        board:    &Board,
        depth:    u32,
        mut alpha: f64,
        beta:     f64,
        deadline: Instant,
        nodes:    &mut u32,
    ) -> Option<f64> {
        *nodes += 1;
        if *nodes % 1000 == 0 && Instant::now() >= deadline {
            return None;
        }
        if let Some(winner) = board.check_winner() {
            return Some(if winner == board.side_to_move { 1.0 } else { 0.0 });
        }
        if depth == 0 {
            let (_, value) = self.net.forward(board);
            return Some(value as f64);
        }
        let moves: Vec<Board> = board.get_nearby_boards()
            .into_iter().take(MAX_MOVES).collect();
        let mut max_score = 0.0_f64;
        for next_board in &moves {
            let score = match self.negamax(next_board, depth - 1, 1.0 - beta, 1.0 - alpha, deadline, nodes) {
                Some(s) => 1.0 - s,
                None    => return None,
            };
            if score > max_score { max_score = score; }
            alpha = alpha.max(score);
            if alpha >= beta { break; }
        }
        Some(max_score)
    }
}

impl Player for NetTimedPlayer {
    fn make_move(&self, board: &Board) -> String {
        let deadline = Instant::now() + Duration::from_millis(self.ms);
        // Get moves sorted by name (matches policy index ordering)
        let mut all_moves: Vec<(String, Board)> = board.get_all_nearby_moves()
            .into_iter().collect();
        all_moves.sort_by(|a, b| a.0.cmp(&b.0));
        all_moves.truncate(MAX_MOVES);
        if all_moves.is_empty() { return String::new(); }
        // Sort by policy prior (higher prior first for better alpha-beta pruning)
        let (policy, _) = self.net.forward(board);
        let n = all_moves.len().min(MAX_MOVES);
        let mut order: Vec<usize> = (0..all_moves.len()).collect();
        order.sort_by(|&i, &j| {
            let pi = if i < n { policy[i] } else { 0.0 };
            let pj = if j < n { policy[j] } else { 0.0 };
            pj.partial_cmp(&pi).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut best_move = all_moves[order[0]].0.clone();
        let mut depth = 1u32;
        loop {
            let mut nodes: u32 = 0;
            let mut best_this_depth = best_move.clone();
            let mut best_score = f64::NEG_INFINITY;
            let mut timed_out = false;
            for &idx in &order {
                let (ref move_name, ref next_board) = all_moves[idx];
                match self.negamax(next_board, depth - 1, 0.0 - 1.0, 0.0, deadline, &mut nodes) {
                    Some(s) => {
                        let score = 1.0 - s;
                        if score > best_score {
                            best_score = score;
                            best_this_depth = move_name.clone();
                        }
                    }
                    None => { timed_out = true; break; }
                }
            }
            if !timed_out {
                best_move = best_this_depth;
                depth += 1;
            } else {
                break;
            }
            if Instant::now() >= deadline { break; }
        }
        best_move
    }
}

// ============================================================================
// MCTS (Monte Carlo Tree Search)
// ============================================================================

fn mcts_rollout(start: &Board, rng: &mut Xorshift64, target: Side) -> f64 {
    let mut board = start.clone();
    for _ in 0..50 {
        if let Some(winner) = board.check_winner() {
            return if winner == target { 1.0 } else { 0.0 };
        }
        let moves = board.get_nearby_boards();
        if moves.is_empty() { return 0.5; }
        let idx = (rng.next() as usize) % moves.len();
        board = moves[idx].clone();
    }
    0.5
}

struct MctsNode {
    board: Board,
    visits: u32,
    wins: f64,              // from perspective of the player WHO MOVED INTO this node
    children: Vec<MctsNode>,
    unexpanded_moves: Vec<Board>,
}

impl MctsNode {
    fn select_uct(&self) -> usize {
        const C: f64 = 1.4142135623730951; // sqrt(2)
        let log_n = (self.visits as f64).ln();
        let mut best = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (i, child) in self.children.iter().enumerate() {
            let score = child.wins / child.visits as f64
                + C * (log_n / child.visits as f64).sqrt();
            if score > best_score {
                best_score = score;
                best = i;
            }
        }
        best
    }

    // Returns result from THIS node's parent's perspective.
    fn iterate(&mut self, rng: &mut Xorshift64) -> f64 {
        if let Some(winner) = self.board.check_winner() {
            let result = if winner == self.board.side_to_move.flip() { 1.0 } else { 0.0 };
            self.wins += result;
            self.visits += 1;
            return result;
        }

        if !self.unexpanded_moves.is_empty() {
            let child_board = self.unexpanded_moves.pop().unwrap();
            let target = self.board.side_to_move;
            let rollout_result = mcts_rollout(&child_board, rng, target);
            let child_unexpanded = if child_board.check_winner().is_none() {
                child_board.get_nearby_boards()
            } else {
                vec![]
            };
            self.children.push(MctsNode {
                board: child_board,
                visits: 1,
                wins: rollout_result,
                children: vec![],
                unexpanded_moves: child_unexpanded,
            });
            let our_result = 1.0 - rollout_result;
            self.wins += our_result;
            self.visits += 1;
            return our_result;
        }

        if self.children.is_empty() {
            self.visits += 1;
            self.wins += 0.5;
            return 0.5;
        }

        let idx = self.select_uct();
        let child_result = self.children[idx].iterate(rng);
        let our_result = 1.0 - child_result;
        self.wins += our_result;
        self.visits += 1;
        our_result
    }
}

// ----------------------------------------------------------------------------
// MCTS Player
// ----------------------------------------------------------------------------

struct MctsPlayer {
    ms: u64,
}

impl MctsPlayer {
    fn new(ms: u64) -> Self {
        Self { ms }
    }
}

impl Player for MctsPlayer {
    fn make_move(&self, board: &Board) -> String {
        let deadline = Instant::now() + Duration::from_millis(self.ms);
        let mut rng = Xorshift64::new();

        let mut named: Vec<(String, Board)> = board.get_all_nearby_moves().into_iter().collect();
        named.sort_by(|a, b| a.0.cmp(&b.0));

        if named.is_empty() {
            return String::new();
        }

        // Reverse so pop() expands named[0] first, keeping children[i] == named[i].
        let mut root = MctsNode {
            board: board.clone(),
            visits: 0,
            wins: 0.0,
            children: vec![],
            unexpanded_moves: named.iter().rev().map(|(_, b)| b.clone()).collect(),
        };

        while Instant::now() < deadline {
            root.iterate(&mut rng);
        }

        let best_idx = root.children.iter()
            .enumerate()
            .max_by_key(|(_, c)| c.visits)
            .map(|(i, _)| i)
            .unwrap_or(0);

        named[best_idx].0.clone()
    }
}

// ----------------------------------------------------------------------------
// MCTS with Eval Rollout (mcts-eval)
// ----------------------------------------------------------------------------

fn mcts_eval_rollout(child_board: &Board, target: Side) -> f64 {
    if let Some(winner) = child_board.check_winner() {
        return if winner == target { 1.0 } else { 0.0 };
    }
    // RichEvaluator scores from child_board.side_to_move's perspective.
    // child_board.side_to_move == target.flip(), so invert to get target's perspective.
    1.0 - RichEvaluator::score(child_board)
}

struct MctsEvalNode {
    board: Board,
    visits: u32,
    wins: f64,
    children: Vec<MctsEvalNode>,
    unexpanded_moves: Vec<Board>,
}

impl MctsEvalNode {
    fn select_uct(&self, c: f64) -> usize {
        let log_n = (self.visits as f64).ln();
        let mut best = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (i, child) in self.children.iter().enumerate() {
            let score = child.wins / child.visits as f64
                + c * (log_n / child.visits as f64).sqrt();
            if score > best_score {
                best_score = score;
                best = i;
            }
        }
        best
    }

    fn iterate(&mut self, c: f64) -> f64 {
        if let Some(winner) = self.board.check_winner() {
            let result = if winner == self.board.side_to_move.flip() { 1.0 } else { 0.0 };
            self.wins += result;
            self.visits += 1;
            return result;
        }

        if !self.unexpanded_moves.is_empty() {
            let child_board = self.unexpanded_moves.pop().unwrap();
            let target = self.board.side_to_move;
            let rollout_result = mcts_eval_rollout(&child_board, target);
            let child_unexpanded = if child_board.check_winner().is_none() {
                child_board.get_nearby_boards()
            } else {
                vec![]
            };
            self.children.push(MctsEvalNode {
                board: child_board,
                visits: 1,
                wins: rollout_result,
                children: vec![],
                unexpanded_moves: child_unexpanded,
            });
            let our_result = 1.0 - rollout_result;
            self.wins += our_result;
            self.visits += 1;
            return our_result;
        }

        if self.children.is_empty() {
            self.visits += 1;
            self.wins += 0.5;
            return 0.5;
        }

        let idx = self.select_uct(c);
        let child_result = self.children[idx].iterate(c);
        let our_result = 1.0 - child_result;
        self.wins += our_result;
        self.visits += 1;
        our_result
    }
}

struct MctsEvalPlayer {
    ms: u64,
    c: f64,
}

impl MctsEvalPlayer {
    fn new(ms: u64) -> Self {
        Self { ms, c: 0.5 }
    }
}

impl Player for MctsEvalPlayer {
    fn make_move(&self, board: &Board) -> String {
        let deadline = Instant::now() + Duration::from_millis(self.ms);

        let mut named: Vec<(String, Board)> = board.get_all_nearby_moves().into_iter().collect();
        named.sort_by(|a, b| a.0.cmp(&b.0));

        if named.is_empty() {
            return String::new();
        }

        let mut root = MctsEvalNode {
            board: board.clone(),
            visits: 0,
            wins: 0.0,
            children: vec![],
            unexpanded_moves: named.iter().rev().map(|(_, b)| b.clone()).collect(),
        };

        while Instant::now() < deadline {
            root.iterate(self.c);
        }

        let best_idx = root.children.iter()
            .enumerate()
            .max_by_key(|(_, c)| c.visits)
            .map(|(i, _)| i)
            .unwrap_or(0);

        named[best_idx].0.clone()
    }
}

// ----------------------------------------------------------------------------
// mcts2: lazy-expansion MCTS with JumpChainEvaluator at leaves
// ----------------------------------------------------------------------------

struct JumpChainEvaluator;

impl JumpChainEvaluator {
    fn count_chain(board: &Board, from: Position, dir: Direction) -> usize {
        let delta = dir.delta();
        let mut count = 0;
        let mut pos = from;
        loop {
            match pos.checked_add(delta) {
                Some(next) if next.is_on_board() => {
                    if board.get(next) == Piece::Man {
                        count += 1;
                        pos = next;
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }
        count
    }

    fn score(board: &Board) -> f64 {
        let mut location = board.ball_at.col as i32;
        if location <= 0 { location = 0; }
        if location >= LENGTH as i32 - 1 { location = LENGTH as i32 - 1; }
        let raw_progress = location as f64 / (LENGTH - 1) as f64;
        let progress = match board.side_to_move {
            Side::Left  => raw_progress,
            Side::Right => 1.0 - raw_progress,
        };

        let mut best_chain = 0usize;
        for dir in Direction::ALL {
            let chain = Self::count_chain(board, board.ball_at, dir);
            if chain > best_chain { best_chain = chain; }
        }
        let chain_score = (best_chain as f64 / 10.0).min(1.0);

        0.5 * progress + 0.5 * chain_score
    }
}

struct Mcts2Node {
    board: Board,
    visits: u32,
    wins: f64,              // from perspective of the player WHO MOVED INTO this node
    children: Vec<Mcts2Node>,
    unexpanded_moves: Vec<Board>,
}

impl Mcts2Node {
    fn select_uct(&self) -> usize {
        const C: f64 = 1.4142135623730951;
        let log_n = (self.visits as f64).ln();
        let mut best = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (i, child) in self.children.iter().enumerate() {
            let score = child.wins / child.visits as f64
                + C * (log_n / child.visits as f64).sqrt();
            if score > best_score {
                best_score = score;
                best = i;
            }
        }
        best
    }

    fn iterate(&mut self) -> f64 {
        if let Some(winner) = self.board.check_winner() {
            let result = if winner == self.board.side_to_move.flip() { 1.0 } else { 0.0 };
            self.wins += result;
            self.visits += 1;
            return result;
        }

        if !self.unexpanded_moves.is_empty() {
            let child_board = self.unexpanded_moves.pop().unwrap();
            let target = self.board.side_to_move;
            let leaf_value = if let Some(winner) = child_board.check_winner() {
                if winner == target { 1.0 } else { 0.0 }
            } else {
                1.0 - JumpChainEvaluator::score(&child_board)
            };
            let child_unexpanded = if child_board.check_winner().is_none() {
                child_board.get_nearby_boards()
            } else {
                vec![]
            };
            self.children.push(Mcts2Node {
                board: child_board,
                visits: 1,
                wins: leaf_value,
                children: vec![],
                unexpanded_moves: child_unexpanded,
            });
            self.wins += leaf_value;
            self.visits += 1;
            return leaf_value;
        }

        if self.children.is_empty() {
            self.visits += 1;
            self.wins += 0.5;
            return 0.5;
        }

        let idx = self.select_uct();
        let child_result = self.children[idx].iterate();
        let our_result = 1.0 - child_result;
        self.wins += our_result;
        self.visits += 1;
        our_result
    }
}

struct Mcts2Player {
    ms: u64,
}

impl Mcts2Player {
    fn new(ms: u64) -> Self {
        Self { ms }
    }
}

impl Player for Mcts2Player {
    fn make_move(&self, board: &Board) -> String {
        let deadline = Instant::now() + Duration::from_millis(self.ms);
        let start = Instant::now();

        let mut named: Vec<(String, Board)> = board.get_all_nearby_moves().into_iter().collect();
        named.sort_by(|a, b| a.0.cmp(&b.0));

        if named.is_empty() {
            return String::new();
        }

        let mut root = Mcts2Node {
            board: board.clone(),
            visits: 0,
            wins: 0.0,
            children: vec![],
            unexpanded_moves: named.iter().rev().map(|(_, b)| b.clone()).collect(),
        };

        while Instant::now() < deadline {
            root.iterate();
        }

        let elapsed = start.elapsed().as_secs_f64();
        let sims = root.visits;
        let sps = if elapsed > 0.0 { sims as f64 / elapsed } else { 0.0 };
        eprintln!("mcts2: {} sims in {:.1}ms = {:.0} sims/sec", sims, elapsed * 1000.0, sps);

        let best_idx = root.children.iter()
            .enumerate()
            .max_by_key(|(_, c)| c.visits)
            .map(|(i, _)| i)
            .unwrap_or(0);

        named[best_idx].0.clone()
    }
}

// ----------------------------------------------------------------------------
// beam-mcts: MCTS with beam selection as policy
// ----------------------------------------------------------------------------

fn beam_filter_boards(parent_ball: Position, boards: Vec<Board>) -> Vec<Board> {
    let (jumps, mut placements): (Vec<Board>, Vec<Board>) =
        boards.into_iter().partition(|b| b.ball_at != parent_ball);

    if placements.len() > BEAM_K {
        placements.sort_unstable_by(|a, b| {
            JumpChainEvaluator::score(b)
                .partial_cmp(&JumpChainEvaluator::score(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        placements.truncate(BEAM_K);
    }

    let mut result = jumps;
    result.extend(placements);
    result
}

struct BeamMctsNode {
    board: Board,
    visits: u32,
    wins: f64,
    children: Vec<BeamMctsNode>,
    unexpanded: Vec<Board>,
}

impl BeamMctsNode {
    fn select_uct(&self) -> usize {
        const C: f64 = 1.4142135623730951;
        let log_n = (self.visits as f64).ln();
        let mut best = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (i, child) in self.children.iter().enumerate() {
            let score = child.wins / child.visits as f64
                + C * (log_n / child.visits as f64).sqrt();
            if score > best_score {
                best_score = score;
                best = i;
            }
        }
        best
    }

    fn iterate(&mut self) -> f64 {
        if let Some(winner) = self.board.check_winner() {
            let result = if winner == self.board.side_to_move.flip() { 1.0 } else { 0.0 };
            self.wins += result;
            self.visits += 1;
            return result;
        }

        if !self.unexpanded.is_empty() {
            let child_board = self.unexpanded.pop().unwrap();
            let target = self.board.side_to_move;
            let leaf_value = if let Some(winner) = child_board.check_winner() {
                if winner == target { 1.0 } else { 0.0 }
            } else {
                1.0 - JumpChainEvaluator::score(&child_board)
            };
            let child_unexpanded = if child_board.check_winner().is_none() {
                let all = child_board.get_nearby_boards();
                beam_filter_boards(child_board.ball_at, all)
            } else {
                vec![]
            };
            self.children.push(BeamMctsNode {
                board: child_board,
                visits: 1,
                wins: leaf_value,
                children: vec![],
                unexpanded: child_unexpanded,
            });
            self.wins += leaf_value;
            self.visits += 1;
            return leaf_value;
        }

        if self.children.is_empty() {
            self.visits += 1;
            self.wins += 0.5;
            return 0.5;
        }

        let idx = self.select_uct();
        let child_result = self.children[idx].iterate();
        let our_result = 1.0 - child_result;
        self.wins += our_result;
        self.visits += 1;
        our_result
    }
}

struct BeamMctsPlayer {
    ms: u64,
}

impl BeamMctsPlayer {
    fn new(ms: u64) -> Self {
        Self { ms }
    }
}

impl Player for BeamMctsPlayer {
    fn make_move(&self, board: &Board) -> String {
        let deadline = Instant::now() + Duration::from_millis(self.ms);
        let start = Instant::now();

        let ball_pos = board.ball_at;
        let all_moves: Vec<(String, Board)> = board.get_all_nearby_moves().into_iter().collect();

        if all_moves.is_empty() {
            return String::new();
        }

        let (jump_moves, mut place_moves): (Vec<(String, Board)>, Vec<(String, Board)>) =
            all_moves.into_iter().partition(|(_, b)| b.ball_at != ball_pos);

        if place_moves.len() > BEAM_K {
            place_moves.sort_unstable_by(|a, b| {
                JumpChainEvaluator::score(&b.1)
                    .partial_cmp(&JumpChainEvaluator::score(&a.1))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            place_moves.truncate(BEAM_K);
        }

        let mut named: Vec<(String, Board)> = jump_moves;
        named.extend(place_moves);
        named.sort_by(|a, b| a.0.cmp(&b.0));

        let mut root = BeamMctsNode {
            board: board.clone(),
            visits: 0,
            wins: 0.0,
            children: vec![],
            unexpanded: named.iter().rev().map(|(_, b)| b.clone()).collect(),
        };

        while Instant::now() < deadline {
            root.iterate();
        }

        let elapsed = start.elapsed().as_secs_f64();
        let sims = root.visits;
        let sps = if elapsed > 0.0 { sims as f64 / elapsed } else { 0.0 };
        eprintln!("beam-mcts: {} sims in {:.1}ms = {:.0} sims/sec", sims, elapsed * 1000.0, sps);

        let best_idx = root.children.iter()
            .enumerate()
            .max_by_key(|(_, c)| c.visits)
            .map(|(i, _)| i)
            .unwrap_or(0);

        named[best_idx].0.clone()
    }
}

// ----------------------------------------------------------------------------
// Location Evaluator
// ----------------------------------------------------------------------------

struct LocationEvaluator;

impl LocationEvaluator {
    fn score(board: &Board) -> f64 {
        let mut location = board.ball_at.col as i32;
        if location <= 0 { location = 0; }
        if location >= LENGTH as i32 - 1 { location = LENGTH as i32 - 1; }

        let value = location as f64 / (LENGTH - 1) as f64;
        match board.side_to_move {
            Side::Left  => value,
            Side::Right => 1.0 - value,
        }
    }
}

// ----------------------------------------------------------------------------
// Rich Evaluator (position + men-near-ball + directional clustering)
// ----------------------------------------------------------------------------

struct RichEvaluator;

impl RichEvaluator {
    fn score(board: &Board) -> f64 {
        // Signal 1: ball column progress (same formula as LocationEvaluator)
        let mut location = board.ball_at.col as i32;
        if location <= 0 { location = 0; }
        if location >= LENGTH as i32 - 1 { location = LENGTH as i32 - 1; }
        let raw_progress = location as f64 / (LENGTH - 1) as f64;
        let progress = match board.side_to_move {
            Side::Left  => raw_progress,
            Side::Right => 1.0 - raw_progress,
        };

        // Signals 2 & 3: single scan over all men
        let ball = board.ball_at;
        let ball_col = ball.col as i32;
        let mut total_men = 0u32;
        let mut men_near  = 0u32;
        let mut forward_men = 0u32;
        for row in 0..WIDTH {
            for col in 0..LENGTH {
                let pos = Position::new(row, col);
                if board.get(pos) == Piece::Man {
                    total_men += 1;
                    let dist = (row as i32 - ball.row as i32).unsigned_abs()
                        + (col as i32 - ball_col).unsigned_abs();
                    if dist <= 3 {
                        men_near += 1;
                    }
                    let is_forward = match board.side_to_move {
                        Side::Left  => (col as i32) > ball_col,
                        Side::Right => (col as i32) < ball_col,
                    };
                    if is_forward {
                        forward_men += 1;
                    }
                }
            }
        }
        let denom = total_men as f64 + 1.0;
        let men_near_score    = men_near   as f64 / denom;
        let directional_score = forward_men as f64 / denom;

        WEIGHT_PROGRESS    * progress
            + WEIGHT_MEN_NEAR    * men_near_score
            + WEIGHT_DIRECTIONAL * directional_score
    }
}

// ----------------------------------------------------------------------------
// Eval4 Evaluator (jump-chain + goal proximity)
// ----------------------------------------------------------------------------

struct Eval4Evaluator;

impl Eval4Evaluator {
    fn count_consecutive_men(board: &Board, from: Position, dir: Direction) -> usize {
        let delta = dir.delta();
        let mut count = 0;
        let mut pos = from;
        loop {
            match pos.checked_add(delta) {
                Some(next) if next.is_on_board() => {
                    if board.get(next) == Piece::Man {
                        count += 1;
                        pos = next;
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }
        count
    }

    fn max_jump_chain(board: &Board) -> usize {
        let mut best = 0;
        for dir in Direction::ALL {
            let chain = Self::count_consecutive_men(board, board.ball_at, dir);
            best = best.max(chain);
        }
        best
    }

    pub fn score(board: &Board) -> f64 {
        // Signal 1: ball column progress
        let mut location = board.ball_at.col as i32;
        if location <= 0 { location = 0; }
        if location >= LENGTH as i32 - 1 { location = LENGTH as i32 - 1; }
        let raw_progress = location as f64 / (LENGTH - 1) as f64;
        let progress = match board.side_to_move {
            Side::Left  => raw_progress,
            Side::Right => 1.0 - raw_progress,
        };

        // Signal 2: max consecutive-men chain (jump chain potential)
        let chain = Self::max_jump_chain(board) as f64;
        let chain_score = (chain / 10.0).min(1.0);

        // Signals 3 & 4: scan all men
        let ball = board.ball_at;
        let ball_col = ball.col as i32;
        let goal_col: i32 = match board.side_to_move {
            Side::Left  => LENGTH as i32 - 1, // col 18: right side
            Side::Right => 0,                  // col 0: left side
        };
        let mut total_men = 0u32;
        let mut men_near  = 0u32;
        let mut goal_side_men = 0u32;
        for row in 0..WIDTH {
            for col in 0..LENGTH {
                let pos = Position::new(row, col);
                if board.get(pos) == Piece::Man {
                    total_men += 1;
                    let dist = (row as i32 - ball.row as i32).unsigned_abs()
                        + (col as i32 - ball_col).unsigned_abs();
                    if dist <= 3 {
                        men_near += 1;
                    }
                    if (col as i32 - goal_col).abs() <= 3 {
                        goal_side_men += 1;
                    }
                }
            }
        }
        let denom = total_men as f64 + 1.0;
        let men_near_score  = men_near      as f64 / denom;
        let goal_prox_score = goal_side_men as f64 / denom;

        0.4 * progress
            + 0.3 * chain_score
            + 0.2 * men_near_score
            + 0.1 * goal_prox_score
    }
}

// ============================================================================
// PhutballNet + AlphaZero MCTS
// ============================================================================

const INPUT_SIZE: usize = WIDTH * LENGTH * 3; // 855 (15×19×3 planes)
const MAX_MOVES: usize = 200;
const HIDDEN1: usize = 64;
const HIDDEN2: usize = 32;

// MLP: INPUT_SIZE -> HIDDEN1 -> HIDDEN2 -> (policy: MAX_MOVES, value: 1)
// Weights stored row-major: w[out_idx * in_size + in_idx]
#[derive(Clone)]
struct PhutballNet {
    w1: Vec<f32>, // HIDDEN1 * INPUT_SIZE
    b1: Vec<f32>, // HIDDEN1
    w2: Vec<f32>, // HIDDEN2 * HIDDEN1
    b2: Vec<f32>, // HIDDEN2
    wp: Vec<f32>, // MAX_MOVES * HIDDEN2
    bp: Vec<f32>, // MAX_MOVES
    wv: Vec<f32>, // HIDDEN2
    bv: f32,
}

// Intermediate activations saved for backprop.
struct ForwardCache {
    h1_pre: Vec<f32>,
    h1:     Vec<f32>,
    h2_pre: Vec<f32>,
    h2:     Vec<f32>,
}

fn xavier_init(rng: &mut Xorshift64, n: usize, fan_in: usize) -> Vec<f32> {
    let scale = (6.0_f32 / fan_in as f32).sqrt();
    (0..n).map(|_| {
        let u = rng.next() as f32 / u64::MAX as f32;
        (u * 2.0 - 1.0) * scale
    }).collect()
}

fn net_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let inv = if sum > 1e-9 { 1.0 / sum } else { 1.0 };
    exps.iter().map(|&e| e * inv).collect()
}

fn net_sigmoid(x: f32) -> f32 {
    if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e = x.exp(); e / (1.0 + e) }
}

impl PhutballNet {
    fn new() -> Self {
        let mut rng = Xorshift64::new();
        Self {
            w1: xavier_init(&mut rng, HIDDEN1 * INPUT_SIZE, INPUT_SIZE),
            b1: vec![0.0; HIDDEN1],
            w2: xavier_init(&mut rng, HIDDEN2 * HIDDEN1, HIDDEN1),
            b2: vec![0.0; HIDDEN2],
            wp: xavier_init(&mut rng, MAX_MOVES * HIDDEN2, HIDDEN2),
            bp: vec![0.0; MAX_MOVES],
            wv: xavier_init(&mut rng, HIDDEN2, HIDDEN2),
            bv: 0.0,
        }
    }

    fn encode_board(board: &Board) -> Vec<f32> {
        let mut input = vec![0.0f32; INPUT_SIZE];
        let side_val = if board.side_to_move == Side::Left { 1.0 } else { 0.0 };
        for row in 0..WIDTH {
            for col in 0..LENGTH {
                let idx = (row * LENGTH + col) * 3;
                let pos = Position::new(row, col);
                match board.get(pos) {
                    Piece::Man   => input[idx]     = 1.0,
                    Piece::Ball  => input[idx + 1] = 1.0,
                    Piece::Empty => {}
                }
                input[idx + 2] = side_val;
            }
        }
        input
    }

    fn forward_input(&self, input: &[f32]) -> (Vec<f32>, f32, ForwardCache) {
        // Layer 1
        let mut h1_pre = vec![0.0f32; HIDDEN1];
        for j in 0..HIDDEN1 {
            let mut s = self.b1[j];
            let base = j * INPUT_SIZE;
            for i in 0..INPUT_SIZE { s += self.w1[base + i] * input[i]; }
            h1_pre[j] = s;
        }
        let h1: Vec<f32> = h1_pre.iter().map(|&x| x.max(0.0)).collect();

        // Layer 2
        let mut h2_pre = vec![0.0f32; HIDDEN2];
        for j in 0..HIDDEN2 {
            let mut s = self.b2[j];
            let base = j * HIDDEN1;
            for i in 0..HIDDEN1 { s += self.w2[base + i] * h1[i]; }
            h2_pre[j] = s;
        }
        let h2: Vec<f32> = h2_pre.iter().map(|&x| x.max(0.0)).collect();

        // Policy head
        let mut policy_logits = vec![0.0f32; MAX_MOVES];
        for j in 0..MAX_MOVES {
            let mut s = self.bp[j];
            let base = j * HIDDEN2;
            for i in 0..HIDDEN2 { s += self.wp[base + i] * h2[i]; }
            policy_logits[j] = s;
        }
        let policy = net_softmax(&policy_logits);

        // Value head
        let mut value_logit = self.bv;
        for i in 0..HIDDEN2 { value_logit += self.wv[i] * h2[i]; }
        let value = net_sigmoid(value_logit);

        (policy, value, ForwardCache { h1_pre, h1, h2_pre, h2 })
    }

    fn forward(&self, board: &Board) -> (Vec<f32>, f32) {
        let input = Self::encode_board(board);
        let (policy, value, _) = self.forward_input(&input);
        (policy, value)
    }

    // SGD step: update weights toward (move_idx, outcome) target.
    fn train_step(&mut self, input: &[f32], move_idx: usize, outcome: f32, lr: f32) {
        let (policy, value, cache) = self.forward_input(input);

        // Value gradient through sigmoid: d(MSE)/d(logit) = (v-y)*v*(1-v)
        let dv = (value - outcome) * value * (1.0 - value);

        // Policy gradient (cross-entropy + softmax fused): p[i] - indicator(i==move_idx)
        let mut dp = policy.clone();
        if move_idx < MAX_MOVES { dp[move_idx] -= 1.0; }

        // Compute dh2 before mutating weights
        let mut dh2 = vec![0.0f32; HIDDEN2];
        for i in 0..HIDDEN2 { dh2[i] += dv * self.wv[i]; }
        for j in 0..MAX_MOVES {
            let g = dp[j];
            if g.abs() < 1e-9 { continue; }
            let base = j * HIDDEN2;
            for i in 0..HIDDEN2 { dh2[i] += g * self.wp[base + i]; }
        }

        // Update value head
        for i in 0..HIDDEN2 { self.wv[i] -= lr * dv * cache.h2[i]; }
        self.bv -= lr * dv;

        // Update policy head
        for j in 0..MAX_MOVES {
            let g = dp[j];
            if g.abs() < 1e-9 { continue; }
            let base = j * HIDDEN2;
            for i in 0..HIDDEN2 { self.wp[base + i] -= lr * g * cache.h2[i]; }
            self.bp[j] -= lr * g;
        }

        // h2 ReLU backward, compute dh1 before mutating w2
        let dh2_pre: Vec<f32> = dh2.iter().zip(cache.h2_pre.iter())
            .map(|(&d, &pre)| if pre > 0.0 { d } else { 0.0 })
            .collect();

        let mut dh1 = vec![0.0f32; HIDDEN1];
        for j in 0..HIDDEN2 {
            let g = dh2_pre[j];
            if g.abs() < 1e-9 { continue; }
            let base = j * HIDDEN1;
            for i in 0..HIDDEN1 { dh1[i] += g * self.w2[base + i]; }
        }

        // Update layer 2
        for j in 0..HIDDEN2 {
            let g = dh2_pre[j];
            if g.abs() < 1e-9 { continue; }
            let base = j * HIDDEN1;
            for i in 0..HIDDEN1 { self.w2[base + i] -= lr * g * cache.h1[i]; }
            self.b2[j] -= lr * g;
        }

        // h1 ReLU backward, update layer 1
        let dh1_pre: Vec<f32> = dh1.iter().zip(cache.h1_pre.iter())
            .map(|(&d, &pre)| if pre > 0.0 { d } else { 0.0 })
            .collect();

        for j in 0..HIDDEN1 {
            let g = dh1_pre[j];
            if g.abs() < 1e-9 { continue; }
            let base = j * INPUT_SIZE;
            for i in 0..INPUT_SIZE { self.w1[base + i] -= lr * g * input[i]; }
            self.b1[j] -= lr * g;
        }
    }

    fn to_flat(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(
            self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len()
            + self.wp.len() + self.bp.len() + self.wv.len() + 1
        );
        v.extend_from_slice(&self.w1);
        v.extend_from_slice(&self.b1);
        v.extend_from_slice(&self.w2);
        v.extend_from_slice(&self.b2);
        v.extend_from_slice(&self.wp);
        v.extend_from_slice(&self.bp);
        v.extend_from_slice(&self.wv);
        v.push(self.bv);
        v
    }

    fn from_flat(v: &[f32]) -> Option<Self> {
        let mut net = Self::new();
        let sizes = [
            net.w1.len(), net.b1.len(), net.w2.len(), net.b2.len(),
            net.wp.len(), net.bp.len(), net.wv.len(), 1,
        ];
        let total: usize = sizes.iter().sum();
        if v.len() != total { return None; }
        let mut pos = 0;
        macro_rules! load {
            ($field:expr) => {
                let n = $field.len();
                $field.copy_from_slice(&v[pos..pos + n]);
                pos += n;
            };
        }
        load!(net.w1); load!(net.b1);
        load!(net.w2); load!(net.b2);
        load!(net.wp); load!(net.bp);
        load!(net.wv);
        net.bv = v[pos];
        Some(net)
    }

    fn save(&self, path: &str) {
        let flat = self.to_flat();
        let bytes: Vec<u8> = flat.iter().flat_map(|&f| f.to_le_bytes()).collect();
        if let Err(e) = std::fs::write(path, &bytes) {
            eprintln!("Warning: could not save weights to {}: {}", path, e);
        }
    }

    fn load(path: &str) -> Self {
        match std::fs::read(path) {
            Ok(bytes) => {
                let flat: Vec<f32> = bytes.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                Self::from_flat(&flat).unwrap_or_else(Self::new)
            }
            Err(_) => Self::new(),
        }
    }
}

// AlphaZero MCTS node.
// wins tracks the cumulative value from THIS NODE'S PARENT'S perspective,
// consistent with MctsEvalNode convention.
struct AzeroNode {
    board: Board,
    prior: f32,
    visits: u32,
    wins: f64,
    children: Vec<AzeroNode>,
}

impl AzeroNode {
    fn new(board: Board, prior: f32) -> Self {
        Self { board, prior, visits: 0, wins: 0.0, children: vec![] }
    }

    // UCT with policy prior: score = Q + C * P / (1 + N)
    fn select_best_child(&self) -> usize {
        const C: f64 = 1.0;
        let sqrt_n = (self.visits as f64).sqrt();
        let mut best = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (i, child) in self.children.iter().enumerate() {
            let q = if child.visits == 0 { 0.0 } else { child.wins / child.visits as f64 };
            let p = child.prior as f64;
            let score = q + C * p * sqrt_n / (1.0 + child.visits as f64);
            if score > best_score {
                best_score = score;
                best = i;
            }
        }
        best
    }

    // Returns value from THIS NODE'S PARENT'S perspective.
    fn iterate(&mut self, net: &PhutballNet) -> f64 {
        if let Some(winner) = self.board.check_winner() {
            let val = if winner == self.board.side_to_move.flip() { 1.0 } else { 0.0 };
            self.wins += val;
            self.visits += 1;
            return val;
        }

        // Leaf: expand with network evaluation
        if self.visits == 0 {
            let moves: Vec<Board> = self.board.get_nearby_boards()
                .into_iter().take(MAX_MOVES).collect();
            let (policy, net_value) = net.forward(&self.board);
            let n = moves.len().min(MAX_MOVES);
            let prior_sum: f32 = policy[..n].iter().sum();
            let scale = if prior_sum > 1e-9 { 1.0 / prior_sum } else { 1.0 };
            self.children = moves.into_iter().enumerate().map(|(i, b)| {
                let p = if i < n { policy[i] * scale } else { 1.0 / n as f32 };
                AzeroNode::new(b, p)
            }).collect();
            // net_value is from self.board.side_to_move; invert for parent's perspective
            let parent_val = 1.0 - net_value as f64;
            self.wins += parent_val;
            self.visits += 1;
            return parent_val;
        }

        if self.children.is_empty() {
            self.visits += 1;
            self.wins += 0.5;
            return 0.5;
        }

        let best = self.select_best_child();
        let child_result = self.children[best].iterate(net);
        let our_result = 1.0 - child_result;
        self.wins += our_result;
        self.visits += 1;
        our_result
    }
}

struct AzeroPlayer {
    net: PhutballNet,
    time_ms: u64,
    max_sims: u32,
}

impl AzeroPlayer {
    fn new(time_ms: u64) -> Self {
        Self { net: PhutballNet::load("weights.bin"), time_ms, max_sims: u32::MAX }
    }

    fn with_net(net: PhutballNet, time_ms: u64) -> Self {
        Self { net, time_ms, max_sims: 20 }
    }
}

impl Player for AzeroPlayer {
    fn make_move(&self, board: &Board) -> String {
        let deadline = Instant::now() + Duration::from_millis(self.time_ms);

        let mut named: Vec<(String, Board)> = board.get_all_nearby_moves().into_iter().collect();
        named.sort_by(|a, b| a.0.cmp(&b.0));
        named.truncate(MAX_MOVES);

        if named.is_empty() {
            return String::new();
        }

        let (policy, _) = self.net.forward(board);
        let n = named.len().min(MAX_MOVES);
        let prior_sum: f32 = policy[..n].iter().sum();
        let scale = if prior_sum > 1e-9 { 1.0 / prior_sum } else { 1.0 };

        // Build root with pre-seeded children (visits=1 skips the leaf-expansion path)
        let mut root = AzeroNode {
            board: board.clone(),
            prior: 1.0,
            visits: 1,
            wins: 0.0,
            children: named.iter().enumerate().map(|(i, (_, b))| {
                let p = if i < n { policy[i] * scale } else { 1.0 / n as f32 };
                AzeroNode::new(b.clone(), p)
            }).collect(),
        };

        let mut sims = 0u32;
        while sims < self.max_sims && Instant::now() < deadline {
            root.iterate(&self.net);
            sims += 1;
        }

        let best_idx = root.children.iter()
            .enumerate()
            .max_by_key(|(_, c)| c.visits)
            .map(|(i, _)| i)
            .unwrap_or(0);

        named[best_idx].0.clone()
    }
}

// ============================================================================
// Imitation Learning: generate-data, pretrain, train
// ============================================================================

struct ImitationSample {
    input:    Vec<f32>, // INPUT_SIZE floats
    move_idx: usize,    // index in sorted get_all_nearby_moves()
    outcome:  f32,      // 1.0=win, 0.0=loss, 0.5=draw from side_to_move's perspective
}

// Play one game using the given engine for both sides, return imitation samples.
// Only positions where the chosen move is in the nearby-move list are recorded.
fn generate_imitation_game(engine_spec: &str, rng: &mut Xorshift64) -> Vec<ImitationSample> {
    let engine1 = parse_player(engine_spec, false);
    let engine2 = parse_player(engine_spec, false);
    let engines: [Box<dyn Player>; 2] = [engine1, engine2];

    let mut board = Board::new();
    let mut placed = 0;
    while placed < 4 {
        let row = (rng.next() % WIDTH as u64) as usize;
        let col = (rng.next() % LENGTH as u64) as usize;
        let pos = Position::new(row, col);
        if board.get(pos) == Piece::Empty {
            board.set(pos, Piece::Man);
            placed += 1;
        }
    }

    let mut positions: Vec<(Vec<f32>, usize, Side)> = Vec::new();

    for _ in 0..100 {
        if board.check_winner().is_some() { break; }

        let idx = board.moves_made as usize % 2;
        let mv = engines[idx].make_move(&board);
        let input = PhutballNet::encode_board(&board);

        let mut named: Vec<(String, Board)> = board.get_all_nearby_moves().into_iter().collect();
        named.sort_by(|a, b| a.0.cmp(&b.0));

        if let Some(move_idx) = named.iter().position(|(name, _)| name == &mv) {
            if move_idx < MAX_MOVES {
                positions.push((input, move_idx, board.side_to_move));
            }
        }

        let all_moves = board.get_all_moves();
        board = match all_moves.get(&mv) {
            Some(b) => b.clone(),
            None => match all_moves.values().next() {
                Some(b) => b.clone(),
                None => break,
            },
        };
    }

    let winner = board.check_winner();
    positions.into_iter().map(|(input, move_idx, side)| {
        let outcome = match winner {
            None    => 0.5,
            Some(w) => if w == side { 1.0 } else { 0.0 },
        };
        ImitationSample { input, move_idx, outcome }
    }).collect()
}

// Binary format: magic(4) + count(4 LE u32) + samples(count × (INPUT_SIZE×4 + 4 + 4))
fn save_imitation_data(samples: &[ImitationSample], path: &str) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).expect("cannot create data file");
    f.write_all(b"PHTB").unwrap();
    f.write_all(&(samples.len() as u32).to_le_bytes()).unwrap();
    for s in samples {
        for &x in &s.input { f.write_all(&x.to_le_bytes()).unwrap(); }
        f.write_all(&(s.move_idx as u32).to_le_bytes()).unwrap();
        f.write_all(&s.outcome.to_le_bytes()).unwrap();
    }
}

fn load_imitation_data(path: &str) -> Vec<ImitationSample> {
    let bytes = std::fs::read(path).expect("cannot read data file");
    if bytes.len() < 8 || &bytes[..4] != b"PHTB" {
        eprintln!("Error: invalid data file");
        std::process::exit(1);
    }
    let count = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let sample_bytes = INPUT_SIZE * 4 + 4 + 4;
    let mut samples = Vec::with_capacity(count);
    let mut pos = 8;
    for _ in 0..count {
        if pos + sample_bytes > bytes.len() { break; }
        let mut input = vec![0.0f32; INPUT_SIZE];
        for i in 0..INPUT_SIZE {
            let o = pos + i * 4;
            input[i] = f32::from_le_bytes([bytes[o], bytes[o+1], bytes[o+2], bytes[o+3]]);
        }
        pos += INPUT_SIZE * 4;
        let move_idx = u32::from_le_bytes([bytes[pos], bytes[pos+1], bytes[pos+2], bytes[pos+3]]) as usize;
        pos += 4;
        let outcome = f32::from_le_bytes([bytes[pos], bytes[pos+1], bytes[pos+2], bytes[pos+3]]);
        pos += 4;
        samples.push(ImitationSample { input, move_idx, outcome });
    }
    samples
}

fn shuffle_samples(samples: &mut Vec<ImitationSample>, rng: &mut Xorshift64) {
    for i in (1..samples.len()).rev() {
        let j = (rng.next() as usize) % (i + 1);
        samples.swap(i, j);
    }
}

fn run_generate_data(games: usize, engine_spec: &str, out_path: &str) {
    let mut rng = Xorshift64::new();
    // Resume from existing partial data if present
    let (mut all_samples, games_done) = if std::path::Path::new(out_path).exists() {
        let existing = load_imitation_data(out_path);
        // Count completed games by reading the game-count trailer written alongside data
        let count_path = format!("{}.games", out_path);
        let done = if std::path::Path::new(&count_path).exists() {
            std::fs::read_to_string(&count_path)
                .ok()
                .and_then(|s| s.trim().parse::<usize>().ok())
                .unwrap_or(0)
        } else { 0 };
        println!("Resuming: {} samples from {} games already done", existing.len(), done);
        (existing, done)
    } else {
        (Vec::new(), 0)
    };
    if games_done >= games {
        println!("Already complete: {} games done, {} samples", games_done, all_samples.len());
        return;
    }
    for g in (games_done + 1)..=games {
        let samples = generate_imitation_game(engine_spec, &mut rng);
        println!("Game {:4}/{}: {} samples (total {})", g, games, samples.len(), all_samples.len() + samples.len());
        all_samples.extend(samples);
        // Save every 20 games (or on the last game) to allow resume if interrupted
        if g % 20 == 0 || g == games {
            save_imitation_data(&all_samples, out_path);
            let count_path = format!("{}.games", out_path);
            std::fs::write(&count_path, g.to_string()).ok();
        }
    }
    println!("Done: {} samples in {} games saved to {}", all_samples.len(), games, out_path);
}

fn run_pretrain(data_path: &str, epochs: usize, save_path: &str) {
    let mut samples = load_imitation_data(data_path);
    println!("Loaded {} samples from {}", samples.len(), data_path);
    let mut net = PhutballNet::new();
    let mut rng = Xorshift64::new();
    const LR: f32 = 0.001;

    for epoch in 1..=epochs {
        shuffle_samples(&mut samples, &mut rng);
        let mut policy_loss = 0.0f64;
        let mut value_loss  = 0.0f64;
        for s in &samples {
            let (policy, value, _) = net.forward_input(&s.input);
            let p = policy.get(s.move_idx).cloned().unwrap_or(1e-9).max(1e-9);
            policy_loss += -p.ln() as f64;
            value_loss  += 0.5 * (value - s.outcome).powi(2) as f64;
            net.train_step(&s.input, s.move_idx, s.outcome, LR);
        }
        let n = samples.len().max(1) as f64;
        println!("Epoch {:3}/{}: policy_loss={:.4} value_loss={:.4}",
            epoch, epochs, policy_loss / n, value_loss / n);
    }
    net.save(save_path);
    println!("Saved weights to {}", save_path);
}

fn run_train(iterations: usize, games_per_iter: usize, save_path: &str, replay_path: Option<&str>) {
    let mut net = PhutballNet::load(save_path);
    let mut rng = Xorshift64::new();
    const LR: f32 = 0.0005;
    const TIME_MS: u64 = 200;

    let replay_pool: Vec<ImitationSample> = match replay_path {
        Some(path) => {
            let data = load_imitation_data(path);
            println!("Loaded {} replay samples from {}", data.len(), path);
            data
        }
        None => Vec::new(),
    };

    for iter in 1..=iterations {
        let mut samples: Vec<ImitationSample> = Vec::new();

        for _ in 0..games_per_iter {
            let player  = AzeroPlayer::with_net(net.clone(), TIME_MS);
            let player2 = AzeroPlayer::with_net(net.clone(), TIME_MS);
            let engines: [&dyn Player; 2] = [&player, &player2];

            let mut board = Board::new();
            let mut placed = 0;
            while placed < 4 {
                let row = (rng.next() % WIDTH as u64) as usize;
                let col = (rng.next() % LENGTH as u64) as usize;
                let pos = Position::new(row, col);
                if board.get(pos) == Piece::Empty {
                    board.set(pos, Piece::Man);
                    placed += 1;
                }
            }

            let mut positions: Vec<(Vec<f32>, usize, Side)> = Vec::new();
            for _ in 0..100 {
                if board.check_winner().is_some() { break; }
                let idx = board.moves_made as usize % 2;
                let mv = engines[idx].make_move(&board);
                let input = PhutballNet::encode_board(&board);
                let mut named: Vec<(String, Board)> = board.get_all_nearby_moves().into_iter().collect();
                named.sort_by(|a, b| a.0.cmp(&b.0));
                if let Some(move_idx) = named.iter().position(|(name, _)| name == &mv) {
                    if move_idx < MAX_MOVES {
                        positions.push((input, move_idx, board.side_to_move));
                    }
                }
                let all_moves = board.get_all_moves();
                board = match all_moves.get(&mv) {
                    Some(b) => b.clone(),
                    None => match all_moves.values().next() {
                        Some(b) => b.clone(),
                        None => break,
                    },
                };
            }
            let winner = board.check_winner();
            for (input, move_idx, side) in positions {
                let outcome = match winner {
                    None    => 0.5,
                    Some(w) => if w == side { 1.0 } else { 0.0 },
                };
                samples.push(ImitationSample { input, move_idx, outcome });
            }
        }

        // Mix in replay samples (50% of self-play count) to prevent mode collapse
        let replay_count = (samples.len() / 2).min(replay_pool.len());
        if replay_count > 0 {
            let start = (rng.next() as usize) % (replay_pool.len() - replay_count + 1);
            for s in &replay_pool[start..start + replay_count] {
                samples.push(ImitationSample {
                    input: s.input.clone(),
                    move_idx: s.move_idx,
                    outcome: s.outcome,
                });
            }
        }

        shuffle_samples(&mut samples, &mut rng);
        let mut policy_loss = 0.0f64;
        let mut value_loss  = 0.0f64;
        for s in &samples {
            let (policy, value, _) = net.forward_input(&s.input);
            let p = policy.get(s.move_idx).cloned().unwrap_or(1e-9).max(1e-9);
            policy_loss += -p.ln() as f64;
            value_loss  += 0.5 * (value - s.outcome).powi(2) as f64;
            net.train_step(&s.input, s.move_idx, s.outcome, LR);
        }
        let n = samples.len().max(1) as f64;
        println!("Iter {:3}/{}: {} sp + {} replay samples, policy_loss={:.4} value_loss={:.4}",
            iter, iterations, samples.len() - replay_count, replay_count, policy_loss / n, value_loss / n);
        net.save(save_path);
    }
    println!("Training complete. Weights saved to {}", save_path);
}

// ============================================================================
// NNUE Value Network — supervised, raw-Rust inference
// ============================================================================
//
// Architecture: 571 inputs → 32 ReLU → 1 sigmoid (value only, no policy).
// Input planes: 285 man-plane + 285 ball-plane + 1 side-to-move.
//
// Weight layout for L1: w1[feature * NNUE_L1 + neuron] (feature-major).
// This means each active feature contributes a contiguous 32-element slice,
// making the sparse first-layer forward pass cache-friendly.
//
// Typical board has ~20 occupied cells, so L1 costs ~20×32 = 640 mults
// instead of 571×32 = 18,272 — ~28× speedup over dense eval.

const NNUE_MAN:   usize = WIDTH * LENGTH;              // 285
const NNUE_BALL_F: usize = WIDTH * LENGTH;             // 285
const NNUE_INPUT: usize = NNUE_MAN + NNUE_BALL_F + 1; // 571
const NNUE_L1:    usize = 32;

#[derive(Clone)]
struct NnueNet {
    w1: Vec<f32>, // [NNUE_INPUT * NNUE_L1] = 18_272 — feature-major layout
    b1: Vec<f32>, // [NNUE_L1]
    w2: Vec<f32>, // [NNUE_L1] — value head
    b2: f32,
}

impl NnueNet {
    fn new() -> Self {
        let mut rng = Xorshift64::new_with_seed(0xdeadbeef_cafef00du64);
        let s1 = (2.0_f32 / NNUE_INPUT as f32).sqrt();
        let s2 = (2.0_f32 / NNUE_L1 as f32).sqrt();
        let w1 = (0..NNUE_INPUT * NNUE_L1)
            .map(|_| { let u = rng.next() as f32 / u64::MAX as f32; (u*2.0-1.0)*s1 })
            .collect();
        let w2 = (0..NNUE_L1)
            .map(|_| { let u = rng.next() as f32 / u64::MAX as f32; (u*2.0-1.0)*s2 })
            .collect();
        Self { w1, b1: vec![0.0; NNUE_L1], w2, b2: 0.0 }
    }

    // Sparse forward: accumulate only for occupied cells and side feature.
    fn score(&self, board: &Board) -> f32 {
        let mut h = self.b1.clone();
        for row in 0..WIDTH {
            for col in 0..LENGTH {
                if board.array[row][col] == Piece::Man {
                    let base = (row * LENGTH + col) * NNUE_L1;
                    for j in 0..NNUE_L1 { h[j] += self.w1[base + j]; }
                }
            }
        }
        let ball = board.ball_at;
        if ball.is_on_board() {
            let base = (NNUE_MAN + ball.row * LENGTH + ball.col) * NNUE_L1;
            for j in 0..NNUE_L1 { h[j] += self.w1[base + j]; }
        }
        if board.side_to_move == Side::Left {
            let base = (NNUE_MAN + NNUE_BALL_F) * NNUE_L1;
            for j in 0..NNUE_L1 { h[j] += self.w1[base + j]; }
        }
        for x in h.iter_mut() { *x = x.max(0.0); }
        let mut v = self.b2;
        for j in 0..NNUE_L1 { v += self.w2[j] * h[j]; }
        net_sigmoid(v)
    }

    fn save(&self, path: &str) {
        let mut flat = Vec::with_capacity(self.w1.len() + self.b1.len() + self.w2.len() + 1);
        flat.extend_from_slice(&self.w1);
        flat.extend_from_slice(&self.b1);
        flat.extend_from_slice(&self.w2);
        flat.push(self.b2);
        let bytes: Vec<u8> = flat.iter().flat_map(|&f| f.to_le_bytes()).collect();
        if let Err(e) = std::fs::write(path, &bytes) {
            eprintln!("Warning: could not save NNUE weights to {}: {}", path, e);
        }
    }

    fn load(path: &str) -> Self {
        let expected = NNUE_INPUT * NNUE_L1 + NNUE_L1 + NNUE_L1 + 1;
        match std::fs::read(path) {
            Ok(bytes) if bytes.len() == expected * 4 => {
                let flat: Vec<f32> = bytes.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let mut net = Self::new();
                let mut pos = 0;
                net.w1.copy_from_slice(&flat[pos..pos + NNUE_INPUT * NNUE_L1]);
                pos += NNUE_INPUT * NNUE_L1;
                net.b1.copy_from_slice(&flat[pos..pos + NNUE_L1]);
                pos += NNUE_L1;
                net.w2.copy_from_slice(&flat[pos..pos + NNUE_L1]);
                pos += NNUE_L1;
                net.b2 = flat[pos];
                net
            }
            Ok(_) => {
                eprintln!("Warning: NNUE weight size mismatch in {}, using random init", path);
                Self::new()
            }
            Err(_) => {
                eprintln!("Warning: {} not found, using random NNUE weights", path);
                Self::new()
            }
        }
    }
}

static NNUE_NET: std::sync::OnceLock<NnueNet> = std::sync::OnceLock::new();

fn nnue_eval(board: &Board) -> f64 {
    NNUE_NET.get_or_init(|| NnueNet::load("nnue.bin")).score(board) as f64
}

// Dense board encoding for data files: 571 floats.
fn nnue_encode_board(board: &Board) -> Vec<f32> {
    let mut enc = vec![0.0f32; NNUE_INPUT];
    for row in 0..WIDTH {
        for col in 0..LENGTH {
            match board.array[row][col] {
                Piece::Man  => enc[row * LENGTH + col] = 1.0,
                Piece::Ball => enc[NNUE_MAN + row * LENGTH + col] = 1.0,
                Piece::Empty => {}
            }
        }
    }
    if board.side_to_move == Side::Left {
        enc[NNUE_MAN + NNUE_BALL_F] = 1.0;
    }
    enc
}

// Binary format: magic "NNUV" + count(u32 LE) + samples(enc[571 f32] + outcome f32).
fn run_nnue_gen_data(games: usize, engine_spec: &str, out_path: &str) {
    let mut rng = Xorshift64::new();
    let mut all_enc: Vec<Vec<f32>> = Vec::new();
    let mut all_out: Vec<f32>      = Vec::new();

    for g in 1..=games {
        let engine1 = parse_player(engine_spec, false);
        let engine2 = parse_player(engine_spec, false);
        let engines: [Box<dyn Player>; 2] = [engine1, engine2];

        let mut board = Board::new();
        let mut placed = 0;
        while placed < 4 {
            let row = (rng.next() % WIDTH as u64) as usize;
            let col = (rng.next() % LENGTH as u64) as usize;
            let pos = Position::new(row, col);
            if board.get(pos) == Piece::Empty {
                board.set(pos, Piece::Man);
                placed += 1;
            }
        }

        let mut pos_boards: Vec<(Vec<f32>, Side)> = Vec::new();
        for _ in 0..100 {
            if board.check_winner().is_some() { break; }
            let idx = board.moves_made as usize % 2;
            let mv  = engines[idx].make_move(&board);
            pos_boards.push((nnue_encode_board(&board), board.side_to_move));
            let all_moves = board.get_all_moves();
            board = match all_moves.get(&mv) {
                Some(b) => b.clone(),
                None    => match all_moves.values().next() {
                    Some(b) => b.clone(),
                    None    => break,
                },
            };
        }

        let winner  = board.check_winner();
        let n_poses = pos_boards.len();
        for (enc, side) in pos_boards {
            let outcome = match winner {
                None    => 0.5,
                Some(w) => if w == side { 1.0 } else { 0.0 },
            };
            all_enc.push(enc);
            all_out.push(outcome);
        }
        if g % 10 == 0 || g == games {
            println!("Game {:4}/{}: {} positions (total {})", g, games, n_poses, all_enc.len());
        }
    }

    let count = all_enc.len() as u32;
    let mut f = std::fs::File::create(out_path).expect("cannot create NNUE data file");
    use std::io::Write as _;
    f.write_all(b"NNUV").unwrap();
    f.write_all(&count.to_le_bytes()).unwrap();
    for (enc, &out) in all_enc.iter().zip(all_out.iter()) {
        for &x in enc { f.write_all(&x.to_le_bytes()).unwrap(); }
        f.write_all(&out.to_le_bytes()).unwrap();
    }
    println!("Saved {} positions from {} games to {}", count, games, out_path);
}

fn run_nnue_train(data_path: &str, epochs: usize, save_path: &str) {
    let bytes = std::fs::read(data_path).expect("cannot read NNUE data file");
    if bytes.len() < 8 || &bytes[..4] != b"NNUV" {
        eprintln!("Error: {} is not a valid NNUE data file", data_path);
        std::process::exit(1);
    }
    let count = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let sample_bytes = NNUE_INPUT * 4 + 4;

    let mut samples: Vec<(Vec<f32>, f32)> = Vec::with_capacity(count);
    let mut pos = 8usize;
    for _ in 0..count {
        if pos + sample_bytes > bytes.len() { break; }
        let mut enc = vec![0.0f32; NNUE_INPUT];
        for i in 0..NNUE_INPUT {
            let o = pos + i * 4;
            enc[i] = f32::from_le_bytes([bytes[o], bytes[o+1], bytes[o+2], bytes[o+3]]);
        }
        pos += NNUE_INPUT * 4;
        let outcome = f32::from_le_bytes([bytes[pos], bytes[pos+1], bytes[pos+2], bytes[pos+3]]);
        pos += 4;
        samples.push((enc, outcome));
    }
    println!("Loaded {} samples from {}", samples.len(), data_path);

    let mut net = NnueNet::new();
    let mut rng  = Xorshift64::new();
    const LR: f32 = 0.001;

    for epoch in 1..=epochs {
        for i in (1..samples.len()).rev() {
            let j = (rng.next() as usize) % (i + 1);
            samples.swap(i, j);
        }
        let mut total_loss = 0.0f64;
        for (enc, outcome) in &samples {
            let outcome = *outcome;
            // Sparse forward
            let mut h = net.b1.clone();
            for (feat, &val) in enc.iter().enumerate() {
                if val != 0.0 {
                    let base = feat * NNUE_L1;
                    for j in 0..NNUE_L1 { h[j] += val * net.w1[base + j]; }
                }
            }
            let h_pre = h.clone();
            for x in h.iter_mut() { *x = x.max(0.0); }
            let mut vlogit = net.b2;
            for j in 0..NNUE_L1 { vlogit += net.w2[j] * h[j]; }
            let value = net_sigmoid(vlogit);
            total_loss += 0.5 * ((value - outcome) as f64).powi(2);

            // Backward: MSE + sigmoid
            let dv = (value - outcome) * value * (1.0 - value);
            let mut dh = vec![0.0f32; NNUE_L1];
            for j in 0..NNUE_L1 {
                dh[j] = dv * net.w2[j];
                net.w2[j] -= LR * dv * h[j];
            }
            net.b2 -= LR * dv;
            for j in 0..NNUE_L1 {
                if h_pre[j] <= 0.0 { dh[j] = 0.0; }
                net.b1[j] -= LR * dh[j];
            }
            for (feat, &val) in enc.iter().enumerate() {
                if val != 0.0 {
                    let base = feat * NNUE_L1;
                    for j in 0..NNUE_L1 { net.w1[base + j] -= LR * val * dh[j]; }
                }
            }
        }
        let n = samples.len().max(1) as f64;
        println!("Epoch {:3}/{}: value_loss={:.5}", epoch, epochs, total_loss / n);
    }
    net.save(save_path);
    println!("Saved NNUE weights to {}", save_path);
}

fn run_nnue_gen_random(positions: usize, max_men: usize, engine_spec: &str, out_path: &str) {
    let parts: Vec<&str> = engine_spec.splitn(2, ':').collect();
    let engine_type = parts[0];
    let ms: u64 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(200);

    let scorer = match engine_type {
        "eval5" => TimedPlayer5::with_eval(ms, false, Eval4Evaluator::score),
        other => {
            eprintln!("nnue-gen-random: unsupported engine '{}', use eval5[:ms]", other);
            std::process::exit(1);
        }
    };

    let mut rng = Xorshift64::new();
    let mut all_enc: Vec<Vec<f32>> = Vec::new();
    let mut all_out: Vec<f32>      = Vec::new();
    let mut generated = 0usize;
    let mut attempts  = 0usize;

    while generated < positions {
        attempts += 1;
        // Ball at row 1-13, col 2-16 (avoids goal columns 0,1 and 17,18)
        let ball_row = 1 + (rng.next() % 13) as usize; // 1..=13
        let ball_col = 2 + (rng.next() % 15) as usize; // 2..=16

        let mut board = Board::new();
        // Relocate ball from starting position to random position
        board.array[board.ball_at.row][board.ball_at.col] = Piece::Empty;
        board.array[ball_row][ball_col] = Piece::Ball;
        board.ball_at = Position::new(ball_row, ball_col);

        // Place K random men (K uniform in 1..=max_men)
        let k = 1 + (rng.next() % max_men.max(1) as u64) as usize;
        let mut placed = 0;
        let mut tries  = 0usize;
        while placed < k && tries < 1000 {
            tries += 1;
            let row = (rng.next() % WIDTH  as u64) as usize;
            let col = (rng.next() % LENGTH as u64) as usize;
            let pos = Position::new(row, col);
            if board.get(pos) == Piece::Empty {
                board.set(pos, Piece::Man);
                placed += 1;
            }
        }

        // Skip positions that are already game-over
        if board.check_winner().is_some() { continue; }

        let score = scorer.score_position(&board) as f32;
        all_enc.push(nnue_encode_board(&board));
        all_out.push(score);
        generated += 1;

        if generated % 100 == 0 || generated == positions {
            println!("Position {:4}/{} (attempts {})", generated, positions, attempts);
        }
    }

    let count = all_enc.len() as u32;
    let mut f = std::fs::File::create(out_path).expect("cannot create NNUE random data file");
    use std::io::Write as _;
    f.write_all(b"NNUV").unwrap();
    f.write_all(&count.to_le_bytes()).unwrap();
    for (enc, &out) in all_enc.iter().zip(all_out.iter()) {
        for &x in enc { f.write_all(&x.to_le_bytes()).unwrap(); }
        f.write_all(&out.to_le_bytes()).unwrap();
    }
    println!("Saved {} random positions to {}", count, out_path);
}

fn run_nnue_merge(inputs: &[String], out_path: &str) {
    let mut combined: Vec<u8> = Vec::new();
    let mut total_count: u32  = 0;

    for input in inputs {
        let bytes = std::fs::read(input)
            .unwrap_or_else(|e| { eprintln!("cannot read {}: {}", input, e); std::process::exit(1); });
        if bytes.len() < 8 || &bytes[..4] != b"NNUV" {
            eprintln!("Error: {} is not a valid NNUE data file", input);
            std::process::exit(1);
        }
        let count = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        println!("  {}: {} samples", input, count);
        total_count += count;
        combined.extend_from_slice(&bytes[8..]);
    }

    use std::io::Write as _;
    let mut f = std::fs::File::create(out_path)
        .unwrap_or_else(|e| { eprintln!("cannot create {}: {}", out_path, e); std::process::exit(1); });
    f.write_all(b"NNUV").unwrap();
    f.write_all(&total_count.to_le_bytes()).unwrap();
    f.write_all(&combined).unwrap();
    println!("Merged {} total samples into {}", total_count, out_path);
}

// ============================================================================
// Tests
// ============================================================================

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
        assert_eq!(pos.checked_add((-10, 0)), None);
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
        assert_eq!(moves.len(), WIDTH * LENGTH - 1);

        let moved_board = moves.get("A1").expect("A1 should be a valid move");
        assert_eq!(moved_board.get(Position::new(0, 0)), Piece::Man);
        assert_eq!(moved_board.side_to_move, Side::Right);
        assert_eq!(moved_board.moves_made, 1);
    }

    #[test]
    fn test_no_ball_moves_initially() {
        let board = Board::new();
        assert_eq!(board.get_ball_moves().len(), 0);
    }

    #[test]
    fn test_simple_jump() {
        let mut board = Board::new();
        let man_pos = Position::new(START_ROW, START_COL + 1);
        board.set(man_pos, Piece::Man);

        let ball_moves = board.get_ball_moves();
        assert!(ball_moves.contains_key("E "));

        let jumped_board = &ball_moves["E "];
        assert_eq!(jumped_board.ball_at.col, START_COL + 2);
        assert_eq!(jumped_board.get(man_pos), Piece::Empty);
        assert_eq!(jumped_board.get(jumped_board.ball_at), Piece::Ball);
    }

    #[test]
    fn test_multi_jump() {
        let mut board = Board::new();
        board.set(Position::new(START_ROW, START_COL + 1), Piece::Man);
        board.set(Position::new(START_ROW, START_COL + 2), Piece::Man);

        let ball_moves = board.get_ball_moves();
        assert!(ball_moves.contains_key("E "));

        let jumped = &ball_moves["E "];
        assert_eq!(jumped.ball_at.col, START_COL + 3);
        assert_eq!(jumped.get(Position::new(START_ROW, START_COL + 1)), Piece::Empty);
        assert_eq!(jumped.get(Position::new(START_ROW, START_COL + 2)), Piece::Empty);
    }

    #[test]
    fn test_direction_deltas() {
        assert_eq!(Direction::N.delta(),  (-1,  0));
        assert_eq!(Direction::S.delta(),  ( 1,  0));
        assert_eq!(Direction::E.delta(),  ( 0,  1));
        assert_eq!(Direction::W.delta(),  ( 0, -1));
        assert_eq!(Direction::NE.delta(), (-1,  1));
        assert_eq!(Direction::NW.delta(), (-1, -1));
        assert_eq!(Direction::SE.delta(), ( 1,  1));
        assert_eq!(Direction::SW.delta(), ( 1, -1));
    }

    #[test]
    fn test_get_all_moves() {
        let board = Board::new();
        let all_moves  = board.get_all_moves();
        let man_moves  = board.get_man_moves();
        let ball_moves = board.get_ball_moves();
        assert_eq!(all_moves.len(), man_moves.len() + ball_moves.len());
    }

    #[test]
    fn test_diagonal_jump() {
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
        let mut board = Board::new();
        board.set(Position::new(START_ROW, START_COL + 1), Piece::Man);
        board.set(Position::new(START_ROW - 1, START_COL + 3), Piece::Man);

        let ball_moves = board.get_ball_moves();
        assert!(ball_moves.contains_key("E "));
        assert!(ball_moves.contains_key("E NE "));

        let double_jump = &ball_moves["E NE "];
        assert_eq!(double_jump.ball_at.row, START_ROW - 2);
        assert_eq!(double_jump.ball_at.col, START_COL + 4);
    }

    #[test]
    fn test_jump_into_goal() {
        let mut board = Board::new();
        board.set(board.ball_at, Piece::Empty);

        let ball_pos = Position::new(0, 1);
        board.ball_at = ball_pos;
        board.set(ball_pos, Piece::Ball);

        let man_pos = Position::new(0, 0);
        board.set(man_pos, Piece::Man);

        let ball_moves = board.get_ball_moves();
        assert!(ball_moves.contains_key("W "), "Should have W jump");

        let jumped = &ball_moves["W "];
        assert!(jumped.ball_at.col <= 0 || !jumped.ball_at.is_on_board());
        assert_eq!(jumped.get(man_pos), Piece::Empty);

        assert!(!ball_moves.contains_key("E "));
        assert!(!ball_moves.contains_key("N "));
        assert!(!ball_moves.contains_key("S "));
        assert!(!ball_moves.contains_key("NE "));
        assert!(!ball_moves.contains_key("NW "));
        assert!(!ball_moves.contains_key("SE "));
        assert!(!ball_moves.contains_key("SW "));
        assert_eq!(ball_moves.len(), 1);
    }

    #[test]
    fn test_multijump_stops_at_goal_col() {
        let mut board = Board::new();
        board.set(board.ball_at, Piece::Empty);

        let ball_pos = Position::new(7, 3);
        board.ball_at = ball_pos;
        board.set(ball_pos, Piece::Ball);
        board.set(Position::new(7, 2), Piece::Man);
        board.set(Position::new(7, 0), Piece::Man);

        let ball_moves = board.get_ball_moves();
        assert!(ball_moves.contains_key("W "));
        let after_w = &ball_moves["W "];
        assert_eq!(after_w.ball_at.col, 1);

        let mut board2 = Board::new();
        board2.set(board2.ball_at, Piece::Empty);
        let ball_pos2 = Position::new(7, 2);
        board2.ball_at = ball_pos2;
        board2.set(ball_pos2, Piece::Ball);
        board2.set(Position::new(7, 1), Piece::Man);
        board2.set(Position::new(6, 0), Piece::Man);

        let ball_moves2 = board2.get_ball_moves();
        assert!(ball_moves2.contains_key("W "));
        let after_w2 = &ball_moves2["W "];
        assert_eq!(after_w2.ball_at.col, 0);

        assert!(!ball_moves2.contains_key("W NE "));
        assert!(!ball_moves2.contains_key("W SE "));
        assert!(!ball_moves2.contains_key("W NW "));
        assert!(!ball_moves2.contains_key("W SW "));
    }

    #[test]
    fn test_diagonal_jump_into_goal() {
        let mut board = Board::new();
        board.set(board.ball_at, Piece::Empty);

        let ball_pos = Position::new(1, 1);
        board.ball_at = ball_pos;
        board.set(ball_pos, Piece::Ball);
        board.set(Position::new(0, 0), Piece::Man);

        let ball_moves = board.get_ball_moves();
        assert!(ball_moves.contains_key("NW "));

        let jumped = &ball_moves["NW "];
        assert_eq!(jumped.ball_at.col, 0);
        assert_eq!(jumped.get(Position::new(0, 0)), Piece::Empty);
    }

    #[test]
    fn test_no_jump_off_row_only() {
        let mut board = Board::new();
        board.set(board.ball_at, Piece::Empty);

        let ball_pos = Position::new(1, 5);
        board.ball_at = ball_pos;
        board.set(ball_pos, Piece::Ball);
        board.set(Position::new(0, 5), Piece::Man);

        let ball_moves = board.get_ball_moves();
        assert!(!ball_moves.contains_key("N "));
    }

    #[test]
    fn test_no_jump_sequence_through_goal() {
        let mut board = Board::new();
        board.set(board.ball_at, Piece::Empty);

        let ball_pos = Position::new(7, 2);
        board.ball_at = ball_pos;
        board.set(ball_pos, Piece::Ball);
        board.set(Position::new(7, 1), Piece::Man);
        board.set(Position::new(7, 0), Piece::Man);

        let ball_moves = board.get_ball_moves();
        assert!(ball_moves.contains_key("W "));

        let after_w = &ball_moves["W "];
        assert_eq!(after_w.ball_at.col, 0);
        assert_eq!(after_w.get(Position::new(7, 1)), Piece::Empty);
        assert_eq!(after_w.get(Position::new(7, 0)), Piece::Empty);
        assert_eq!(after_w.get(Position::new(7, 0)), Piece::Empty);
    }

    #[test]
    fn test_full_board_with_ball_at_b2() {
        let mut board = Board::new();
        board.set(board.ball_at, Piece::Empty);

        let ball_pos = Position::new(1, 1);
        board.ball_at = ball_pos;
        board.set(ball_pos, Piece::Ball);

        for row in 0..WIDTH {
            for col in 0..LENGTH {
                let pos = Position::new(row, col);
                if pos.row != ball_pos.row || pos.col != ball_pos.col {
                    board.set(pos, Piece::Man);
                }
            }
        }

        let all_moves  = board.get_all_moves();
        let ball_moves = board.get_ball_moves();
        let man_moves  = board.get_man_moves();

        assert_eq!(man_moves.len(), 0);
        assert_eq!(ball_moves.len(), 4);
        assert!(ball_moves.contains_key("NW "));
        assert!(ball_moves.contains_key("SW "));
        assert!(ball_moves.contains_key("E "));
        assert!(ball_moves.contains_key("W "));
        assert_eq!(all_moves.len(), ball_moves.len());
        assert!(ball_moves.len() > 0);
    }

    #[test]
    fn test_alphabeta_equals_minimax_symmetric() {
        let mut board = Board::new();
        board.set(board.ball_at, Piece::Empty);

        let ball_pos = Position::new(7, 9);
        board.ball_at = ball_pos;
        board.set(ball_pos, Piece::Ball);

        for dr in -1..=1i32 {
            for dc in -1..=1i32 {
                if dr == 0 && dc == 0 { continue; }
                if let Some(pos) = ball_pos.checked_add((dr, dc)) {
                    if pos.is_on_board() { board.set(pos, Piece::Man); }
                }
            }
        }

        let minimax_d2  = MinimaxPlayer::new(2, false);
        let alphabeta_d2 = AlphaBetaPlayer::new(2, false);
        assert_eq!(minimax_d2.evaluate(&board), alphabeta_d2.evaluate(&board),
                   "Depth 2: Minimax and AlphaBeta should evaluate position identically");

        let minimax_d3  = MinimaxPlayer::new(3, false);
        let alphabeta_d3 = AlphaBetaPlayer::new(3, false);
        assert_eq!(minimax_d3.evaluate(&board), alphabeta_d3.evaluate(&board),
                   "Depth 3: Minimax and AlphaBeta should evaluate position identically");
    }

    #[test]
    fn test_alphabeta_equals_minimax_asymmetric() {
        let mut board = Board::new();
        board.set(board.ball_at, Piece::Empty);

        let ball_pos = Position::new(7, 9);
        board.ball_at = ball_pos;
        board.set(ball_pos, Piece::Ball);

        for opt in &[
            ball_pos.checked_add((-1, 0)),
            ball_pos.checked_add((0, 1)),
            ball_pos.checked_add((1, 0)),
            ball_pos.checked_add((-1, 1)),
            ball_pos.checked_add((1, -1)),
        ] {
            if let Some(pos) = *opt {
                if pos.is_on_board() { board.set(pos, Piece::Man); }
            }
        }

        let minimax_d2  = MinimaxPlayer::new(2, false);
        let alphabeta_d2 = AlphaBetaPlayer::new(2, false);
        assert_eq!(minimax_d2.evaluate(&board), alphabeta_d2.evaluate(&board),
                   "Depth 2: asymmetric");

        let minimax_d3  = MinimaxPlayer::new(3, false);
        let alphabeta_d3 = AlphaBetaPlayer::new(3, false);
        assert_eq!(minimax_d3.evaluate(&board), alphabeta_d3.evaluate(&board),
                   "Depth 3: asymmetric");
    }

    #[test]
    fn test_parallel_alphabeta_equals_alphabeta_symmetric() {
        let mut board = Board::new();
        board.set(board.ball_at, Piece::Empty);

        let ball_pos = Position::new(7, 9);
        board.ball_at = ball_pos;
        board.set(ball_pos, Piece::Ball);

        for dr in -1..=1i32 {
            for dc in -1..=1i32 {
                if dr == 0 && dc == 0 { continue; }
                if let Some(pos) = ball_pos.checked_add((dr, dc)) {
                    if pos.is_on_board() { board.set(pos, Piece::Man); }
                }
            }
        }

        for depth in 2..=4 {
            let alphabeta = AlphaBetaPlayer::new(depth, false);
            let parallel  = ParallelAlphaBetaPlayer::new(depth);
            assert_eq!(alphabeta.evaluate(&board), parallel.evaluate(&board),
                       "Depth {}: Parallel should match sequential AlphaBeta", depth);
        }
    }

    #[test]
    fn test_parallel_alphabeta_equals_alphabeta_asymmetric() {
        let mut board = Board::new();
        board.set(board.ball_at, Piece::Empty);

        let ball_pos = Position::new(7, 9);
        board.ball_at = ball_pos;
        board.set(ball_pos, Piece::Ball);

        for opt in &[
            ball_pos.checked_add((-1, 0)),
            ball_pos.checked_add((0, 1)),
            ball_pos.checked_add((1, 0)),
            ball_pos.checked_add((-1, 1)),
            ball_pos.checked_add((1, -1)),
        ] {
            if let Some(pos) = *opt {
                if pos.is_on_board() { board.set(pos, Piece::Man); }
            }
        }

        for depth in 2..=4 {
            let alphabeta = AlphaBetaPlayer::new(depth, false);
            let parallel  = ParallelAlphaBetaPlayer::new(depth);
            assert_eq!(alphabeta.evaluate(&board), parallel.evaluate(&board),
                       "Depth {}: Parallel should match sequential for asymmetric", depth);
        }
    }

    #[test]
    fn test_parallel_deterministic_move_selection() {
        let mut board = Board::new();
        board.set(board.ball_at, Piece::Empty);

        let ball_pos = Position::new(7, 9);
        board.ball_at = ball_pos;
        board.set(ball_pos, Piece::Ball);

        for dr in -1..=1i32 {
            for dc in -1..=1i32 {
                if dr == 0 && dc == 0 { continue; }
                if let Some(pos) = ball_pos.checked_add((dr, dc)) {
                    if pos.is_on_board() { board.set(pos, Piece::Man); }
                }
            }
        }

        let parallel = ParallelAlphaBetaPlayer::new(3);
        let moves: Vec<String> = (0..5).map(|_| parallel.make_move(&board)).collect();

        for i in 1..moves.len() {
            assert_eq!(moves[0], moves[i], "Parallel player should make same move consistently");
        }
    }

    // -------------------------------------------------------------------------
    // New tests for timed engine and tournament
    // -------------------------------------------------------------------------

    #[test]
    fn test_timed_engine_returns_valid_move() {
        let board = Board::new();
        let player = TimedPlayer::new(100, false);
        let start = Instant::now();
        let mv = player.make_move(&board);
        let elapsed = start.elapsed();

        // Should finish within 3× budget (generous for CI)
        assert!(elapsed.as_millis() < 500, "Timed engine took {}ms", elapsed.as_millis());

        // Move must be in the nearby-moves set the engine uses
        let nearby = board.get_all_nearby_moves();
        assert!(nearby.contains_key(&mv), "Timed engine returned invalid move: {:?}", mv);
    }

    #[test]
    fn test_tournament_game_completes() {
        // A single game between two plodding engines should complete without panicking.
        let mut rng = Xorshift64::new_with_seed(42);
        for _ in 0..3 {
            let _result = play_tournament_game("plodding", "plodding", &mut rng, 4);
            // Any result (win/lose/tie) is acceptable — just must not panic.
        }
    }

    #[test]
    fn test_ibeta_uniform() {
        // Beta(1,1) = Uniform(0,1); P(θ > 0.5) should be exactly 0.5.
        let p = prob_engine1_better(0, 0);
        assert!((p - 0.5).abs() < 1e-6, "P={}", p);
    }

    #[test]
    fn test_ibeta_one_win() {
        // Beta(2,1): P(θ > 0.5) = 1 - I_0.5(2,1) = 0.75
        let p = prob_engine1_better(1, 0);
        assert!((p - 0.75).abs() < 1e-6, "P={}", p);
    }

    #[test]
    fn test_ibeta_symmetry() {
        // prob_engine1_better(w, w) should be 0.5 for any w.
        for w in [0u32, 1, 5, 10, 50] {
            let p = prob_engine1_better(w, w);
            assert!((p - 0.5).abs() < 1e-6, "w={} P={}", w, p);
        }
    }

    #[test]
    fn test_mcts_returns_valid_move() {
        let board = Board::new();
        let player = MctsPlayer::new(100);
        let start = Instant::now();
        let mv = player.make_move(&board);
        let elapsed = start.elapsed();

        assert!(elapsed.as_millis() < 500, "MCTS took {}ms", elapsed.as_millis());
        let nearby = board.get_all_nearby_moves();
        assert!(nearby.contains_key(&mv), "MCTS returned invalid move: {:?}", mv);
    }

    #[test]
    fn test_mcts_eval_returns_valid_move() {
        let board = Board::new();
        let player = MctsEvalPlayer::new(100);
        let start = Instant::now();
        let mv = player.make_move(&board);
        let elapsed = start.elapsed();

        assert!(elapsed.as_millis() < 500, "MCTS-eval took {}ms", elapsed.as_millis());
        let nearby = board.get_all_nearby_moves();
        assert!(nearby.contains_key(&mv), "MCTS-eval returned invalid move: {:?}", mv);
    }

    #[test]
    fn test_mcts2_returns_valid_move() {
        let board = Board::new();
        let player = Mcts2Player::new(100);
        let start = Instant::now();
        let mv = player.make_move(&board);
        let elapsed = start.elapsed();

        assert!(elapsed.as_millis() < 500, "mcts2 took {}ms", elapsed.as_millis());
        let nearby = board.get_all_nearby_moves();
        assert!(nearby.contains_key(&mv), "mcts2 returned invalid move: {:?}", mv);
    }

    #[test]
    fn test_beam_mcts_returns_valid_move() {
        let board = Board::new();
        let player = BeamMctsPlayer::new(100);
        let start = Instant::now();
        let mv = player.make_move(&board);
        let elapsed = start.elapsed();

        assert!(elapsed.as_millis() < 500, "beam-mcts took {}ms", elapsed.as_millis());
        let nearby = board.get_all_nearby_moves();
        assert!(nearby.contains_key(&mv), "beam-mcts returned invalid move: {:?}", mv);
    }

    #[test]
    fn test_azero_returns_valid_move() {
        let board = Board::new();
        let player = AzeroPlayer::new(100);
        let start = Instant::now();
        let mv = player.make_move(&board);
        let elapsed = start.elapsed();
        assert!(elapsed.as_millis() < 500, "Azero took {}ms", elapsed.as_millis());
        let nearby = board.get_all_nearby_moves();
        assert!(nearby.contains_key(&mv), "Azero returned invalid move: {:?}", mv);
    }

    #[test]
    fn test_rich_evaluator_differs_from_location_evaluator() {
        let mut board = Board::new();
        // Place several men within Manhattan distance ≤ 3 of the ball.
        for dc in 1..=2i32 {
            if let Some(pos) = board.ball_at.checked_add((0, dc)) {
                if pos.is_on_board() {
                    board.set(pos, Piece::Man);
                }
            }
        }

        let location_score = LocationEvaluator::score(&board);
        let rich_score     = RichEvaluator::score(&board);

        assert_ne!(
            location_score, rich_score,
            "RichEvaluator should score differently from LocationEvaluator when men are near ball"
        );
    }

    #[test]
    fn test_generate_imitation_data() {
        let mut rng = Xorshift64::new_with_seed(42);
        let samples = generate_imitation_game("eval:10", &mut rng);
        for s in &samples {
            assert_eq!(s.input.len(), INPUT_SIZE, "input must be INPUT_SIZE floats");
            assert!(s.move_idx < MAX_MOVES, "move_idx={} >= MAX_MOVES={}", s.move_idx, MAX_MOVES);
            assert!(s.outcome >= 0.0 && s.outcome <= 1.0, "outcome={} out of [0,1]", s.outcome);
        }
    }

    #[test]
    fn test_ttable_store_and_retrieve() {
        let mut tt = TTable::new();
        let board = Board::new();
        let zt = get_zobrist();
        let hash = zobrist_hash(&board, zt);

        // Nothing stored yet
        assert!(tt.probe(hash).is_none());

        tt.store(hash, 3, 0.75, NodeType::Exact, 0xdeadbeef);
        let entry = tt.probe(hash).expect("entry must be present after store");
        assert_eq!(entry.hash, hash);
        assert_eq!(entry.depth, 3);
        assert!((entry.score - 0.75).abs() < 1e-9);
        assert!(matches!(entry.node_type, NodeType::Exact));
        assert_eq!(entry.best_next_hash, 0xdeadbeef);

        // Different hash should miss
        assert!(tt.probe(hash ^ 1).is_none());
    }

    #[test]
    fn test_eval5_aspiration_score_in_range() {
        // Verify eval5 returns a valid move and internally scores stay in [0,1].
        // We exercise via make_move on the starting board with a short budget.
        let board = Board::new();
        let player = TimedPlayer5::with_eval(200, false, Eval4Evaluator::score);
        let mv = player.make_move(&board);
        let nearby = board.get_all_nearby_moves();
        assert!(
            nearby.contains_key(&mv),
            "eval5 returned a move not in nearby-moves set: {:?}",
            mv
        );

        // Confirm the leaf evaluator never returns outside [0,1] on this board.
        let score = Eval4Evaluator::score(&board);
        assert!(
            score >= 0.0 && score <= 1.0,
            "Eval4Evaluator score out of [0,1]: {}",
            score
        );
    }

    #[test]
    fn test_eval6_beam_search_returns_valid_move() {
        // eval6 (beam search) must return a valid move; beam pruning fires at depth>=2.
        // We run with enough time to reach depth 2 so the pruning code is exercised.
        let board = Board::new();
        let player = TimedPlayer6::with_eval(300, false, Eval4Evaluator::score);
        let mv = player.make_move(&board);
        let nearby = board.get_all_nearby_moves();
        assert!(
            nearby.contains_key(&mv),
            "eval6 returned a move not in nearby-moves set: {:?}",
            mv
        );
    }

    #[test]
    fn test_eval4_jump_chain_scoring() {
        // Board with 3 consecutive men east of ball
        let mut board3 = Board::new();
        for dc in 1..=3i32 {
            if let Some(pos) = board3.ball_at.checked_add((0, dc)) {
                if pos.is_on_board() {
                    board3.set(pos, Piece::Man);
                }
            }
        }

        // Board with only 1 man east of ball
        let mut board1 = Board::new();
        if let Some(pos) = board1.ball_at.checked_add((0, 1)) {
            if pos.is_on_board() {
                board1.set(pos, Piece::Man);
            }
        }

        let score3 = Eval4Evaluator::score(&board3);
        let score1 = Eval4Evaluator::score(&board1);
        assert!(
            score3 > score1,
            "Eval4 should score 3-chain ({:.4}) higher than 1-chain ({:.4})",
            score3, score1
        );
    }

    #[test]
    fn test_nnue_score_in_range() {
        let board = Board::new();
        let net   = NnueNet::new();
        let score = net.score(&board);
        assert!(
            score >= 0.0 && score <= 1.0,
            "NnueNet::score out of [0,1]: {}",
            score
        );
    }

    #[test]
    fn test_nnue_encode_board_active_features() {
        let board = Board::new();
        let enc   = nnue_encode_board(&board);
        assert_eq!(enc.len(), NNUE_INPUT);
        let nonzero = enc.iter().filter(|&&x| x != 0.0).count();
        // Starting board: 1 ball active in ball plane + 1 side_left feature = 2
        assert_eq!(nonzero, 2,
            "Starting board should have 2 active features (ball + side_left), got {}", nonzero);
    }

    #[test]
    fn test_nnue_save_load_roundtrip() {
        let net  = NnueNet::new();
        let board = Board::new();
        let score_before = net.score(&board);

        let tmp = "/tmp/test_nnue_roundtrip.bin";
        net.save(tmp);
        let loaded = NnueNet::load(tmp);
        let score_after = loaded.score(&board);

        assert!(
            (score_before - score_after).abs() < 1e-6,
            "score changed after save/load: {} vs {}",
            score_before, score_after
        );
        let _ = std::fs::remove_file(tmp);
    }

    #[test]
    fn test_nnue_training_reduces_loss() {
        // One epoch of training on a trivial single sample should reduce loss.
        let board   = Board::new();
        let enc     = nnue_encode_board(&board);
        let outcome = 1.0f32;

        let mut net = NnueNet::new();
        let initial_value = net.score(&board);
        let initial_loss  = 0.5 * (initial_value - outcome).powi(2);

        // 20 gradient steps toward outcome=1.0
        const LR: f32 = 0.01;
        for _ in 0..20 {
            let mut h = net.b1.clone();
            for (feat, &val) in enc.iter().enumerate() {
                if val != 0.0 {
                    let base = feat * NNUE_L1;
                    for j in 0..NNUE_L1 { h[j] += val * net.w1[base + j]; }
                }
            }
            let h_pre = h.clone();
            for x in h.iter_mut() { *x = x.max(0.0); }
            let mut vlogit = net.b2;
            for j in 0..NNUE_L1 { vlogit += net.w2[j] * h[j]; }
            let value = net_sigmoid(vlogit);
            let dv = (value - outcome) * value * (1.0 - value);
            let mut dh = vec![0.0f32; NNUE_L1];
            for j in 0..NNUE_L1 {
                dh[j] = dv * net.w2[j];
                net.w2[j] -= LR * dv * h[j];
            }
            net.b2 -= LR * dv;
            for j in 0..NNUE_L1 {
                if h_pre[j] <= 0.0 { dh[j] = 0.0; }
                net.b1[j] -= LR * dh[j];
            }
            for (feat, &val) in enc.iter().enumerate() {
                if val != 0.0 {
                    let base = feat * NNUE_L1;
                    for j in 0..NNUE_L1 { net.w1[base + j] -= LR * val * dh[j]; }
                }
            }
        }

        let final_value = net.score(&board);
        let final_loss  = 0.5 * (final_value - outcome).powi(2);
        assert!(
            final_loss < initial_loss,
            "Training did not reduce loss: {:.5} → {:.5}",
            initial_loss, final_loss
        );
    }

    #[test]
    fn test_eval5q_qsearch_returns_score() {
        // Board with a man adjacent to the ball so get_ball_boards() is non-empty.
        let mut board = Board::new();
        if let Some(pos) = board.ball_at.checked_add((0, 1)) {
            if pos.is_on_board() {
                board.set(pos, Piece::Man);
            }
        }
        let player = TimedPlayer5Q::new(200, false);
        let mut nodes = 0u64;
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(100);
        let score = player.qsearch(&board, 0.0, 1.0, Q_DEPTH, deadline, &mut nodes);
        assert!(score.is_some(), "qsearch must return Some(score)");
        let s = score.unwrap();
        assert!(s >= 0.0 && s <= 1.0, "qsearch score out of [0,1]: {}", s);
    }
}
