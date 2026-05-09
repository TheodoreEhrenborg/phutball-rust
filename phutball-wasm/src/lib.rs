#![allow(dead_code)]

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use yew::prelude::*;
use gloo::timers::callback::Timeout;
use std::collections::HashMap;

// ============================================================================
// Constants
// ============================================================================

const LENGTH: usize = 19;
const WIDTH: usize = 15;
const START_ROW: usize = 7;
const START_COL: usize = 9;

const WEIGHT_PROGRESS: f64 = 0.6;
const WEIGHT_MEN_NEAR: f64 = 0.3;
const WEIGHT_DIRECTIONAL: f64 = 0.1;

// ============================================================================
// Timing (uses js_sys::Date::now() for milliseconds since epoch)
// ============================================================================

fn now_ms() -> f64 {
    js_sys::Date::now()
}

// ============================================================================
// Board types
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    Left,
    Right,
}

impl Side {
    fn flip(self) -> Self {
        match self {
            Side::Left => Side::Right,
            Side::Right => Side::Left,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Side::Left => "Left",
            Side::Right => "Right",
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

    fn is_on_board(self) -> bool {
        self.row < WIDTH && self.col < LENGTH
    }

    fn checked_add(self, delta: (i32, i32)) -> Option<Self> {
        let nr = self.row as i32 + delta.0;
        let nc = self.col as i32 + delta.1;
        if nr >= 0 && nc >= 0 {
            Some(Position::new(nr as usize, nc as usize))
        } else {
            None
        }
    }

    fn to_notation(self) -> String {
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

    fn delta(self) -> (i32, i32) {
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

impl std::fmt::Display for Direction {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Clone, PartialEq)]
struct Board {
    side_to_move: Side,
    moves_made: u32,
    array: [[Piece; LENGTH]; WIDTH],
    ball_at: Position,
}

impl Board {
    fn new() -> Self {
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

    fn get_all_moves(&self) -> HashMap<String, Board> {
        let mut moves = self.get_man_moves();
        moves.extend(self.get_ball_moves());
        moves
    }

    fn get_all_nearby_moves(&self) -> HashMap<String, Board> {
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

    fn check_winner(&self) -> Option<Side> {
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
                None => continue,
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

    fn get_ball_moves_recursive(&self, moves: &mut HashMap<String, Board>, prefix: String) {
        if !self.ball_at.is_on_board() {
            return;
        }

        for direction in Direction::ALL {
            let delta = direction.delta();

            let first_pos = match self.ball_at.checked_add(delta) {
                Some(p) => p,
                None => continue,
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
}

// ============================================================================
// Zobrist Hashing + Transposition Table
// ============================================================================

struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
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

struct ZobristTable {
    man_at: [[u64; LENGTH]; WIDTH],
    ball_at: [[u64; LENGTH]; WIDTH],
    side_left: u64,
}

impl ZobristTable {
    fn new() -> Self {
        let mut rng = Xorshift64::new_with_seed(0x9e3779b97f4a7c15);
        let mut man_at = [[0u64; LENGTH]; WIDTH];
        let mut ball_at = [[0u64; LENGTH]; WIDTH];
        for row in 0..WIDTH {
            for col in 0..LENGTH {
                man_at[row][col] = rng.next();
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
                Piece::Man => h ^= zt.man_at[row][col],
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
    hash: u64,
    depth: u32,
    score: f64,
    node_type: NodeType,
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
// Evaluators
// ============================================================================

struct LocationEvaluator;

impl LocationEvaluator {
    fn score(board: &Board) -> f64 {
        let mut location = board.ball_at.col as i32;
        if location <= 0 { location = 0; }
        if location >= LENGTH as i32 - 1 { location = LENGTH as i32 - 1; }
        let value = location as f64 / (LENGTH - 1) as f64;
        match board.side_to_move {
            Side::Left => value,
            Side::Right => 1.0 - value,
        }
    }
}

struct RichEvaluator;

impl RichEvaluator {
    fn score(board: &Board) -> f64 {
        let mut location = board.ball_at.col as i32;
        if location <= 0 { location = 0; }
        if location >= LENGTH as i32 - 1 { location = LENGTH as i32 - 1; }
        let raw_progress = location as f64 / (LENGTH - 1) as f64;
        let progress = match board.side_to_move {
            Side::Left => raw_progress,
            Side::Right => 1.0 - raw_progress,
        };

        let ball = board.ball_at;
        let ball_col = ball.col as i32;
        let mut total_men = 0u32;
        let mut men_near = 0u32;
        let mut forward_men = 0u32;
        for row in 0..WIDTH {
            for col in 0..LENGTH {
                let pos = Position::new(row, col);
                if board.get(pos) == Piece::Man {
                    total_men += 1;
                    let dist = (row as i32 - ball.row as i32).unsigned_abs()
                        + (col as i32 - ball_col).unsigned_abs();
                    if dist <= 3 { men_near += 1; }
                    let is_forward = match board.side_to_move {
                        Side::Left => (col as i32) > ball_col,
                        Side::Right => (col as i32) < ball_col,
                    };
                    if is_forward { forward_men += 1; }
                }
            }
        }
        let denom = total_men as f64 + 1.0;
        WEIGHT_PROGRESS * progress
            + WEIGHT_MEN_NEAR * (men_near as f64 / denom)
            + WEIGHT_DIRECTIONAL * (forward_men as f64 / denom)
    }
}

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

    fn score(board: &Board) -> f64 {
        let mut location = board.ball_at.col as i32;
        if location <= 0 { location = 0; }
        if location >= LENGTH as i32 - 1 { location = LENGTH as i32 - 1; }
        let raw_progress = location as f64 / (LENGTH - 1) as f64;
        let progress = match board.side_to_move {
            Side::Left => raw_progress,
            Side::Right => 1.0 - raw_progress,
        };

        let chain = Self::max_jump_chain(board) as f64;
        let chain_score = (chain / 10.0).min(1.0);

        let ball = board.ball_at;
        let ball_col = ball.col as i32;
        let goal_col: i32 = match board.side_to_move {
            Side::Left => LENGTH as i32 - 1,
            Side::Right => 0,
        };
        let mut total_men = 0u32;
        let mut men_near = 0u32;
        let mut goal_side_men = 0u32;
        for row in 0..WIDTH {
            for col in 0..LENGTH {
                let pos = Position::new(row, col);
                if board.get(pos) == Piece::Man {
                    total_men += 1;
                    let dist = (row as i32 - ball.row as i32).unsigned_abs()
                        + (col as i32 - ball_col).unsigned_abs();
                    if dist <= 3 { men_near += 1; }
                    if (col as i32 - goal_col).abs() <= 3 { goal_side_men += 1; }
                }
            }
        }
        let denom = total_men as f64 + 1.0;
        0.4 * progress
            + 0.3 * chain_score
            + 0.2 * (men_near as f64 / denom)
            + 0.1 * (goal_side_men as f64 / denom)
    }
}

// ============================================================================
// Engine implementations
// ============================================================================

struct PloddingEngine;

impl PloddingEngine {
    fn make_move(board: &Board) -> String {
        let moves = board.get_all_moves();
        let preferred_dir = match board.side_to_move {
            Side::Left => "E ",
            Side::Right => "W ",
        };
        if moves.contains_key(preferred_dir) {
            return preferred_dir.to_string();
        }
        let offset = match board.side_to_move {
            Side::Left => 1,
            Side::Right => -1,
        };
        if let Some(target) = board.ball_at.checked_add((0, offset)) {
            if target.is_on_board() {
                let mv = target.to_notation();
                if moves.contains_key(&mv) {
                    return mv;
                }
            }
        }
        moves.keys().next().unwrap().clone()
    }
}

struct AlphaBetaEngine {
    depth: u32,
}

impl AlphaBetaEngine {
    fn new(depth: u32) -> Self {
        Self { depth }
    }

    fn negamax_ab(board: &Board, depth: u32, mut alpha: f64, beta: f64) -> f64 {
        if let Some(winner) = board.check_winner() {
            return if winner == board.side_to_move { 1.0 } else { 0.0 };
        }
        if depth == 0 {
            return LocationEvaluator::score(board);
        }
        let moves = board.get_all_nearby_moves();
        let mut max_score = 0.0_f64;
        for (_, next_board) in moves {
            let score = 1.0 - Self::negamax_ab(&next_board, depth - 1, 1.0 - beta, 1.0 - alpha);
            if score > max_score { max_score = score; }
            alpha = alpha.max(score);
            if alpha >= beta { break; }
        }
        max_score
    }

    fn make_move(&self, board: &Board) -> String {
        let moves = board.get_all_nearby_moves();
        let mut best_move = moves.keys().next().unwrap().clone();
        let mut best_score = 0.0_f64;
        for (move_name, next_board) in &moves {
            let score = 1.0 - Self::negamax_ab(next_board, self.depth - 1, 0.0, 1.0);
            if score > best_score || (score == best_score && *move_name < best_move) {
                best_score = score;
                best_move = move_name.clone();
            }
        }
        best_move
    }
}

struct Eval2Engine {
    budget_ms: u64,
    eval_fn: fn(&Board) -> f64,
}

impl Eval2Engine {
    fn new(budget_ms: u64) -> Self {
        Self { budget_ms, eval_fn: RichEvaluator::score }
    }

    fn negamax(
        &self,
        board: &Board,
        depth: u32,
        mut alpha: f64,
        mut beta: f64,
        deadline: f64,
        nodes: &mut u64,
        tt: &mut TTable,
        zt: &ZobristTable,
    ) -> Option<f64> {
        *nodes += 1;
        if *nodes % 1000 == 0 && now_ms() >= deadline {
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
                    NodeType::Exact => return Some(entry.score),
                    NodeType::LowerBound => alpha = alpha.max(entry.score),
                    NodeType::UpperBound => beta = beta.min(entry.score),
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
                next_board, depth - 1, 1.0 - beta, 1.0 - alpha,
                deadline, nodes, tt, zt,
            ) {
                Some(s) => 1.0 - s,
                None => return None,
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

    fn make_move(&self, board: &Board) -> String {
        let deadline = now_ms() + self.budget_ms as f64;
        let zt = get_zobrist();
        let moves_map = board.get_all_nearby_moves();
        if moves_map.is_empty() { return String::new(); }

        let mut moves: Vec<(String, Board)> = moves_map.into_iter().collect();
        let mut best_move = moves[0].0.clone();
        let mut depth = 1u32;
        let mut tt = TTable::new();

        loop {
            let mut nodes: u64 = 0;
            let mut depth_best_idx = 0usize;
            let mut depth_best_score = f64::NEG_INFINITY;
            let mut timed_out = false;

            for (idx, (move_name, next_board)) in moves.iter().enumerate() {
                match self.negamax(next_board, depth - 1, 0.0, 1.0, deadline, &mut nodes, &mut tt, zt) {
                    Some(s) => {
                        let score = 1.0 - s;
                        if score > depth_best_score
                            || (score == depth_best_score && *move_name < moves[depth_best_idx].0)
                        {
                            depth_best_score = score;
                            depth_best_idx = idx;
                        }
                    }
                    None => { timed_out = true; break; }
                }
            }

            if !timed_out {
                best_move = moves[depth_best_idx].0.clone();
                moves.swap(0, depth_best_idx);
                depth += 1;
            } else {
                break;
            }

            if now_ms() >= deadline { break; }
        }

        best_move
    }
}

struct Eval5Engine {
    budget_ms: u64,
    eval_fn: fn(&Board) -> f64,
}

impl Eval5Engine {
    fn new(budget_ms: u64) -> Self {
        Self { budget_ms, eval_fn: Eval4Evaluator::score }
    }

    fn negamax(
        &self,
        board: &Board,
        depth: u32,
        mut alpha: f64,
        mut beta: f64,
        deadline: f64,
        nodes: &mut u64,
        tt: &mut TTable,
        zt: &ZobristTable,
    ) -> Option<f64> {
        *nodes += 1;
        if *nodes % 1000 == 0 && now_ms() >= deadline {
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
                    NodeType::Exact => return Some(entry.score),
                    NodeType::LowerBound => alpha = alpha.max(entry.score),
                    NodeType::UpperBound => beta = beta.min(entry.score),
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
                next_board, depth - 1, 1.0 - beta, 1.0 - alpha,
                deadline, nodes, tt, zt,
            ) {
                Some(s) => 1.0 - s,
                None => return None,
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

    fn make_move(&self, board: &Board) -> String {
        let deadline = now_ms() + self.budget_ms as f64;
        let zt = get_zobrist();
        let moves_map = board.get_all_nearby_moves();
        if moves_map.is_empty() { return String::new(); }

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
                let mut depth_best_idx = 0usize;
                let mut depth_best_score = f64::NEG_INFINITY;
                let mut timed_out = false;

                for (idx, (_mv, next_board)) in moves.iter().enumerate() {
                    match self.negamax(
                        next_board, depth - 1, 1.0 - beta, 1.0 - alpha,
                        deadline, &mut nodes, &mut tt, zt,
                    ) {
                        Some(s) => {
                            let score = 1.0 - s;
                            if score > depth_best_score
                                || (score == depth_best_score && moves[idx].0 < moves[depth_best_idx].0)
                            {
                                depth_best_score = score;
                                depth_best_idx = idx;
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
                    depth += 1;
                    break;
                }

                if now_ms() >= deadline { break 'depth; }
            }

            if now_ms() >= deadline { break; }
        }

        best_move
    }
}

// ============================================================================
// Eval6: beam search forward pruning (best engine)
// ============================================================================

const BEAM_K: usize = 8;

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

struct Eval6Engine {
    budget_ms: u64,
    eval_fn: fn(&Board) -> f64,
}

impl Eval6Engine {
    fn new(budget_ms: u64) -> Self {
        Self { budget_ms, eval_fn: Eval4Evaluator::score }
    }

    fn negamax(
        &self,
        board: &Board,
        depth: u32,
        mut alpha: f64,
        mut beta: f64,
        deadline: f64,
        nodes: &mut u64,
        tt: &mut TTable,
        zt: &ZobristTable,
    ) -> Option<f64> {
        *nodes += 1;
        if *nodes % 1000 == 0 && now_ms() >= deadline {
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
                    NodeType::Exact => return Some(entry.score),
                    NodeType::LowerBound => alpha = alpha.max(entry.score),
                    NodeType::UpperBound => beta = beta.min(entry.score),
                }
                if alpha >= beta { return Some(entry.score); }
            }
        }
        let ball_pos = board.ball_at;
        let all_children = board.get_nearby_boards();
        let (jumps, mut placements): (Vec<Board>, Vec<Board>) =
            all_children.into_iter().partition(|b| b.ball_at != ball_pos);
        if depth >= 2 && placements.len() > BEAM_K {
            let tt_best = if tt_best_hash != 0 {
                placements.iter().position(|b| zobrist_hash(b, zt) == tt_best_hash)
                    .map(|pos| placements.swap_remove(pos))
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
                next_board, depth - 1, 1.0 - beta, 1.0 - alpha, deadline, nodes, tt, zt,
            ) {
                Some(s) => 1.0 - s,
                None => return None,
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

    fn make_move(&self, board: &Board) -> String {
        let deadline = now_ms() + self.budget_ms as f64;
        let zt = get_zobrist();
        let moves_map = board.get_all_nearby_moves();
        if moves_map.is_empty() { return String::new(); }
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
                let mut depth_best_idx = 0usize;
                let mut depth_best_score = f64::NEG_INFINITY;
                let mut timed_out = false;
                for (idx, (_mv, next_board)) in moves.iter().enumerate() {
                    match self.negamax(
                        next_board, depth - 1, 1.0 - beta, 1.0 - alpha,
                        deadline, &mut nodes, &mut tt, zt,
                    ) {
                        Some(s) => {
                            let score = 1.0 - s;
                            if score > depth_best_score
                                || (score == depth_best_score && moves[idx].0 < moves[depth_best_idx].0)
                            {
                                depth_best_score = score;
                                depth_best_idx = idx;
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
                    depth += 1;
                    break;
                }
                if now_ms() >= deadline { break 'depth; }
            }
            if now_ms() >= deadline { break; }
        }
        best_move
    }
}

// ============================================================================
// Engine enum (dispatcher)
// ============================================================================

enum Engine {
    Human,
    Plodding,
    Eval6(Eval6Engine),
}

impl Engine {
    fn from_spec(spec: &str, budget_ms: u64) -> Engine {
        match spec {
            "human" => Engine::Human,
            "plodding" => Engine::Plodding,
            "eval6" => Engine::Eval6(Eval6Engine::new(budget_ms)),
            _ => Engine::Human,
        }
    }

    fn is_human(&self) -> bool {
        matches!(self, Engine::Human)
    }

    fn make_move(&self, board: &Board) -> String {
        match self {
            Engine::Human => String::new(),
            Engine::Plodding => PloddingEngine::make_move(board),
            Engine::Eval6(e) => e.make_move(board),
        }
    }
}

// ============================================================================
// GameState: Yew-managed application state
// ============================================================================

#[derive(Clone, PartialEq)]
struct GameState {
    board: Board,
    left_spec: String,
    right_spec: String,
    budget_ms: u64,
    ball_selected: bool,
    // (move_name, dest_row, dest_col) for available ball jumps
    jump_dests: Vec<(String, usize, usize)>,
}

impl GameState {
    fn new(left: &str, right: &str, budget_ms: u64) -> Self {
        Self {
            board: Board::new(),
            left_spec: left.to_string(),
            right_spec: right.to_string(),
            budget_ms,
            ball_selected: false,
            jump_dests: vec![],
        }
    }

    fn current_engine(&self) -> Engine {
        match self.board.side_to_move {
            Side::Left => Engine::from_spec(&self.left_spec, self.budget_ms),
            Side::Right => Engine::from_spec(&self.right_spec, self.budget_ms),
        }
    }

    fn is_human_turn(&self) -> bool {
        self.current_engine().is_human()
    }

    fn winner(&self) -> Option<Side> {
        self.board.check_winner()
    }

    fn play_engine_move(&mut self) -> bool {
        if self.winner().is_some() { return false; }
        let engine = self.current_engine();
        if engine.is_human() { return false; }
        let mv = engine.make_move(&self.board);
        if mv.is_empty() { return false; }
        let moves = self.board.get_all_moves();
        if let Some(new_board) = moves.get(&mv) {
            self.board = new_board.clone();
            true
        } else {
            false
        }
    }

    fn play_human_move(&mut self, mv: &str) -> bool {
        if self.winner().is_some() { return false; }
        let moves = self.board.get_all_moves();
        if let Some(new_board) = moves.get(mv) {
            self.board = new_board.clone();
            self.ball_selected = false;
            self.jump_dests.clear();
            return true;
        }
        let with_space = format!("{} ", mv);
        if let Some(new_board) = moves.get(&with_space) {
            self.board = new_board.clone();
            self.ball_selected = false;
            self.jump_dests.clear();
            return true;
        }
        false
    }

    fn toggle_ball(&mut self) {
        if self.ball_selected {
            self.ball_selected = false;
            self.jump_dests.clear();
        } else {
            self.ball_selected = true;
            let ball_moves = self.board.get_ball_moves();
            self.jump_dests = ball_moves
                .into_iter()
                .map(|(mv, b)| (mv, b.ball_at.row, b.ball_at.col))
                .collect();
        }
    }

    fn status_text(&self) -> String {
        if let Some(winner) = self.winner() {
            return format!("Game over — {} wins!", winner.as_str());
        }
        let side = self.board.side_to_move.as_str();
        if self.is_human_turn() {
            format!("{} to move — type a move below and click Preview", side)
        } else {
            format!("{} to move (engine thinking\u{2026})", side)
        }
    }
}

// ============================================================================
// SVG board layout constants
// ============================================================================

const CELL: i32 = 32;
const PAD_L: i32 = 40;
const PAD_T: i32 = 30;

fn svg_w() -> i32 { PAD_L + (LENGTH as i32 - 1) * CELL + 20 }
fn svg_h() -> i32 { PAD_T + (WIDTH as i32 - 1) * CELL + 20 }

fn board_svg(board: &Board, cell: i32, pad_l: i32, pad_t: i32) -> Html {
    let w = pad_l + (LENGTH as i32 - 1) * cell + 20;
    let h = pad_t + (WIDTH as i32 - 1) * cell + 20;

    let vert_lines = (0..LENGTH as i32).map(|c| {
        let x = pad_l + c * cell;
        html! { <line x1={x.to_string()} y1={pad_t.to_string()}
                      x2={x.to_string()} y2={(pad_t + (WIDTH as i32 - 1) * cell).to_string()}
                      stroke="#aaa" stroke-width="0.8"/> }
    });

    let horiz_lines = (0..WIDTH as i32).map(|r| {
        let y = pad_t + r * cell;
        html! { <line x1={pad_l.to_string()} y1={y.to_string()}
                      x2={(pad_l + (LENGTH as i32 - 1) * cell).to_string()} y2={y.to_string()}
                      stroke="#aaa" stroke-width="0.8"/> }
    });

    let col_labels = (0..LENGTH as i32).map(|c| {
        let x = pad_l + c * cell;
        html! { <text x={x.to_string()} y={(pad_t - 6).to_string()}
                      text-anchor="middle" font-size="11" fill="#333">
                    {(c + 1).to_string()}
                </text> }
    });

    let row_labels = (0..WIDTH as i32).map(|r| {
        let y = pad_t + r * cell;
        let label = (b'A' + r as u8) as char;
        html! { <text x={(pad_l - 6).to_string()} y={y.to_string()}
                      text-anchor="end" dy="0.35em" font-size="11" fill="#333">
                    {label.to_string()}
                </text> }
    });

    let men: Vec<Html> = (0..WIDTH).flat_map(|row| {
        (0..LENGTH).filter_map(move |col| {
            if board.array[row][col] == Piece::Man {
                let cx = pad_l + col as i32 * cell;
                let cy = pad_t + row as i32 * cell;
                let r = (cell * 5 / 16).max(4);
                Some(html! {
                    <circle cx={cx.to_string()} cy={cy.to_string()} r={r.to_string()}
                            fill="white" stroke="#333" stroke-width="1.5"/>
                })
            } else {
                None
            }
        })
    }).collect();

    let ball = board.ball_at;
    let bcx = pad_l + ball.col as i32 * cell;
    let bcy = pad_t + ball.row as i32 * cell;
    let ball_r = (cell * 3 / 8).max(5);

    let goal_l_x = pad_l - cell / 2;
    let goal_r_x = pad_l + (LENGTH as i32 - 1) * cell;
    let goal_y = pad_t - cell / 2;
    let goal_zone_h = (WIDTH as i32 - 1) * cell + cell;

    html! {
        <svg width={w.to_string()} height={h.to_string()}
             style="max-width:100%;height:auto;border:1px solid #999;background:white;display:block;">
            <rect x={goal_l_x.to_string()} y={goal_y.to_string()}
                  width={(cell / 2).to_string()} height={goal_zone_h.to_string()}
                  fill="rgba(255,150,150,0.25)"/>
            <rect x={goal_r_x.to_string()} y={goal_y.to_string()}
                  width={(cell / 2).to_string()} height={goal_zone_h.to_string()}
                  fill="rgba(150,200,255,0.25)"/>
            { for vert_lines }
            { for horiz_lines }
            { for col_labels }
            { for row_labels }
            { for men }
            <circle cx={bcx.to_string()} cy={bcy.to_string()} r={ball_r.to_string()}
                    fill="#111" stroke="white" stroke-width="1.5"/>
        </svg>
    }
}

// ============================================================================
// Yew App component
// ============================================================================

#[function_component(App)]
fn app() -> Html {
    let left_spec = use_state(|| "eval6".to_string());
    let right_spec = use_state(|| "human".to_string());
    let budget_secs = use_state(|| 5u64);
    let game = use_state(|| GameState::new("eval6", "human", 5000));
    let move_input: UseStateHandle<String> = use_state(|| String::new());
    let preview_board: UseStateHandle<Option<Board>> = use_state(|| None);
    let move_error: UseStateHandle<Option<String>> = use_state(|| None);

    // Schedule engine moves whenever game state changes
    {
        let game = game.clone();
        use_effect_with((*game).clone(), move |g| {
            let is_engine_turn = !g.is_human_turn() && g.winner().is_none();
            if is_engine_turn {
                let game = game.clone();
                Timeout::new(50, move || {
                    let mut next = (*game).clone();
                    next.play_engine_move();
                    game.set(next);
                })
                .forget();
            }
            || ()
        });
    }

    // New game
    let on_new_game = {
        let game = game.clone();
        let left_spec = left_spec.clone();
        let right_spec = right_spec.clone();
        let budget_secs = budget_secs.clone();
        let move_input = move_input.clone();
        let preview_board = preview_board.clone();
        let move_error = move_error.clone();
        Callback::from(move |_: MouseEvent| {
            game.set(GameState::new(&left_spec, &right_spec, *budget_secs * 1000));
            move_input.set(String::new());
            preview_board.set(None);
            move_error.set(None);
        })
    };

    // Left player select
    let on_left_change = {
        let left_spec = left_spec.clone();
        Callback::from(move |e: Event| {
            if let Some(el) = e.target().and_then(|t| t.dyn_into::<web_sys::HtmlSelectElement>().ok()) {
                left_spec.set(el.value());
            }
        })
    };

    // Right player select
    let on_right_change = {
        let right_spec = right_spec.clone();
        Callback::from(move |e: Event| {
            if let Some(el) = e.target().and_then(|t| t.dyn_into::<web_sys::HtmlSelectElement>().ok()) {
                right_spec.set(el.value());
            }
        })
    };

    // Budget slider
    let on_budget_input = {
        let budget_secs = budget_secs.clone();
        Callback::from(move |e: web_sys::InputEvent| {
            if let Some(el) = e.target().and_then(|t| t.dyn_into::<web_sys::HtmlInputElement>().ok()) {
                if let Ok(v) = el.value().parse::<u64>() {
                    budget_secs.set(v);
                }
            }
        })
    };

    // Move text input: update state and auto-preview on each keystroke
    let on_move_input = {
        let game = game.clone();
        let move_input = move_input.clone();
        let preview_board = preview_board.clone();
        let move_error = move_error.clone();
        Callback::from(move |e: web_sys::InputEvent| {
            if let Some(el) = e.target().and_then(|t| t.dyn_into::<web_sys::HtmlInputElement>().ok()) {
                let val = el.value();
                move_input.set(val.clone());
                if val.is_empty() {
                    preview_board.set(None);
                    move_error.set(None);
                } else {
                    let all_moves = (*game).board.get_all_moves();
                    if let Some(b) = all_moves.get(&val).or_else(|| all_moves.get(&format!("{} ", val))) {
                        preview_board.set(Some(b.clone()));
                        move_error.set(None);
                    } else {
                        preview_board.set(None);
                        move_error.set(Some("Invalid move".to_string()));
                    }
                }
            }
        })
    };

    // Preview button
    let on_preview = {
        let game = game.clone();
        let move_input = move_input.clone();
        let preview_board = preview_board.clone();
        let move_error = move_error.clone();
        Callback::from(move |_: MouseEvent| {
            let val = (*move_input).clone();
            if val.is_empty() {
                move_error.set(Some("Enter a move first".to_string()));
                return;
            }
            let all_moves = (*game).board.get_all_moves();
            if let Some(b) = all_moves.get(&val).or_else(|| all_moves.get(&format!("{} ", val))) {
                preview_board.set(Some(b.clone()));
                move_error.set(None);
            } else {
                preview_board.set(None);
                move_error.set(Some("Invalid move".to_string()));
            }
        })
    };

    // Confirm button: execute the previewed move
    let on_confirm = {
        let game = game.clone();
        let move_input = move_input.clone();
        let preview_board = preview_board.clone();
        let move_error = move_error.clone();
        Callback::from(move |_: MouseEvent| {
            if (*preview_board).is_none() { return; }
            let input = (*move_input).clone();
            let mut new_state = (*game).clone();
            if new_state.play_human_move(&input) {
                game.set(new_state);
                move_input.set(String::new());
                preview_board.set(None);
                move_error.set(None);
            }
        })
    };

    let g = &*game;
    let status = g.status_text();
    let lv = (*left_spec).clone();
    let rv = (*right_spec).clone();
    let bv = *budget_secs;
    let cur_input = (*move_input).clone();
    let has_preview = (*preview_board).is_some();
    let is_human_active = g.is_human_turn() && g.winner().is_none();
    let preview_svg = (*preview_board).as_ref().map(|b| board_svg(b, 18, 28, 22));
    let error_msg = (*move_error).clone();

    html! {
        <div style="font-family:sans-serif;margin:10px;background:#f4f4f4;min-height:100vh;">
            <h1 style="margin:0 0 8px;font-size:18px;">{"Phutball (Philosopher\u{2019}s Football)"}</h1>

            <div style="margin-bottom:8px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:13px;">
                <label>{"Left: "}
                    <select onchange={on_left_change}>
                        <option value="human"    selected={lv == "human"}>{"Human"}</option>
                        <option value="plodding" selected={lv == "plodding"}>{"Plodding"}</option>
                        <option value="eval6"    selected={lv == "eval6"}>{"Beam Search"}</option>
                    </select>
                </label>
                <label>{"Right: "}
                    <select onchange={on_right_change}>
                        <option value="human"    selected={rv == "human"}>{"Human"}</option>
                        <option value="plodding" selected={rv == "plodding"}>{"Plodding"}</option>
                        <option value="eval6"    selected={rv == "eval6"}>{"Beam Search"}</option>
                    </select>
                </label>
                <label>
                    {format!("{}s per move", bv)}
                    <input type="range" min="1" max="60" step="1"
                           value={bv.to_string()}
                           oninput={on_budget_input}/>
                </label>
                <button onclick={on_new_game} style="padding:6px 12px;cursor:pointer;font-size:13px;">
                    {"New Game"}
                </button>
            </div>

            <div style="width:100%;overflow-x:auto;-webkit-overflow-scrolling:touch;">
                { board_svg(&g.board, CELL, PAD_L, PAD_T) }
            </div>

            <div style="margin-top:6px;font-size:15px;font-weight:bold;min-height:22px;">
                {status}
            </div>

            if is_human_active {
                <div style="margin-top:8px;display:flex;gap:6px;align-items:center;flex-wrap:wrap;">
                    <input type="text"
                           value={cur_input}
                           oninput={on_move_input}
                           placeholder="Enter move (e.g. A5, E , E NE )"
                           style="padding:4px 8px;font-size:14px;width:220px;"/>
                    <button onclick={on_preview}
                            style="padding:4px 10px;cursor:pointer;font-size:13px;">
                        {"Preview"}
                    </button>
                    <button onclick={on_confirm}
                            disabled={!has_preview}
                            style="padding:4px 10px;cursor:pointer;font-size:13px;">
                        {"Confirm"}
                    </button>
                    if let Some(ref err) = error_msg {
                        <span style="color:red;font-size:13px;">{err}</span>
                    }
                </div>
            }

            if let Some(svg) = preview_svg {
                <div style="margin-top:8px;">
                    <div style="font-size:13px;font-weight:bold;margin-bottom:4px;">{"After this move:"}</div>
                    <div style="width:100%;overflow-x:auto;">
                        {svg}
                    </div>
                </div>
            }

            <div style="margin-top:8px;font-size:12px;color:#555;max-width:620px;">
                <b>{"Rules: "}</b>
                {"Left tries to move the ball to the right edge (column 19); Right to the left edge (column 1). "}
                {"On your turn: type a move and click Preview, then Confirm. Man placement: row letter + column (e.g. A5). "}
                {"Ball jump: direction + space (e.g. E ). Chain jumps with spaces (e.g. E NE )."}
            </div>
        </div>
    }
}

// ============================================================================
// Entry point
// ============================================================================

#[wasm_bindgen(start)]
pub fn run() {
    yew::Renderer::<App>::new().render();
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yew_app_compiles() {
        // Verifies core game logic works; full WASM test is: wasm-pack build phutball-wasm --target web
        let gs = GameState::new("human", "human", 500);
        assert_eq!(gs.board.moves_made, 0);
        assert!(gs.winner().is_none());
        assert!(!gs.ball_selected);

        // Plodding engine doesn't use js_sys timing, safe to call in native tests
        let mut gs2 = GameState::new("plodding", "plodding", 100);
        gs2.play_engine_move();
        assert_eq!(gs2.board.moves_made, 1);
    }
}
