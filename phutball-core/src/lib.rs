#![allow(dead_code)]

use std::collections::HashMap;

// ============================================================================
// Constants
// ============================================================================

pub const LENGTH: usize = 19;
pub const WIDTH: usize = 15;
pub const START_ROW: usize = 7;
pub const START_COL: usize = 9;

pub const WEIGHT_PROGRESS: f64 = 0.6;
pub const WEIGHT_MEN_NEAR: f64 = 0.3;
pub const WEIGHT_DIRECTIONAL: f64 = 0.1;

pub const BEAM_K: usize = 8;

// ============================================================================
// Timing — platform-specific
// ============================================================================

#[cfg(target_arch = "wasm32")]
pub fn now_ms() -> f64 {
    js_sys::Date::now()
}

#[cfg(not(target_arch = "wasm32"))]
pub fn now_ms() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as f64)
        .unwrap_or(0.0)
}

// ============================================================================
// Board types
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}

impl Side {
    pub fn flip(self) -> Self {
        match self {
            Side::Left => Side::Right,
            Side::Right => Side::Left,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Side::Left => "Left",
            Side::Right => "Right",
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Piece {
    Man,
    Empty,
    Ball,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Position {
    pub row: usize,
    pub col: usize,
}

impl Position {
    pub fn new(row: usize, col: usize) -> Self {
        Self { row, col }
    }

    pub fn is_on_board(self) -> bool {
        self.row < WIDTH && self.col < LENGTH
    }

    pub fn checked_add(self, delta: (i32, i32)) -> Option<Self> {
        let new_row = self.row as i32 + delta.0;
        let new_col = self.col as i32 + delta.1;
        if new_row >= 0 && new_col >= 0 {
            Some(Position::new(new_row as usize, new_col as usize))
        } else {
            None
        }
    }

    pub fn to_notation(self) -> String {
        let row_char = (b'A' + self.row as u8) as char;
        format!("{}{}", row_char, self.col + 1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    N, S, E, W, NE, NW, SE, SW,
}

impl Direction {
    pub const ALL: [Direction; 8] = [
        Direction::N, Direction::S, Direction::E, Direction::W,
        Direction::NE, Direction::NW, Direction::SE, Direction::SW,
    ];

    pub fn delta(self) -> (i32, i32) {
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

#[derive(Clone)]
pub struct Board {
    pub side_to_move: Side,
    pub moves_made: u32,
    pub array: [[Piece; LENGTH]; WIDTH],
    pub ball_at: Position,
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

    pub fn get(&self, pos: Position) -> Piece {
        self.array[pos.row][pos.col]
    }

    pub fn set(&mut self, pos: Position, piece: Piece) {
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

    pub fn get_ball_moves(&self) -> HashMap<String, Board> {
        let mut moves = HashMap::new();
        self.get_ball_moves_recursive(&mut moves, String::new());
        moves
    }

    pub fn get_ball_boards(&self) -> Vec<Board> {
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

    pub fn get_nearby_boards(&self) -> Vec<Board> {
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

    pub fn increment(&mut self) {
        self.side_to_move = self.side_to_move.flip();
        self.moves_made += 1;
    }

    pub fn clone_with_side_flip(&self) -> Board {
        let mut b = self.clone();
        b.side_to_move = b.side_to_move.flip();
        b
    }

    pub fn pretty_string_details(&self) -> String {
        let white_circle: char = '\u{25CB}';
        let black_circle: char = '\u{25CF}';
        let mut output = String::new();
        output.push_str("          1111111111\n");
        output.push_str(" 1234567890123456789\n");

        for row in 0..WIDTH {
            output.push((b'A' + row as u8) as char);
            for col in 0..LENGTH {
                let pos = Position::new(row, col);
                match self.get(pos) {
                    Piece::Man   => output.push(white_circle),
                    Piece::Ball  => output.push(black_circle),
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
        let white_circle: char = '\u{25CB}';
        let black_circle: char = '\u{25CF}';
        let mut output = String::new();
        for row in 0..WIDTH {
            for col in 0..LENGTH {
                let pos = Position::new(row, col);
                match self.get(pos) {
                    Piece::Man   => output.push(white_circle),
                    Piece::Ball  => output.push(black_circle),
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

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for Board {
    fn eq(&self, other: &Self) -> bool {
        self.side_to_move == other.side_to_move
            && self.moves_made == other.moves_made
            && self.ball_at == other.ball_at
            && self.array == other.array
    }
}

// ============================================================================
// Zobrist Hashing + Transposition Table
// ============================================================================

pub struct Xorshift64 {
    pub state: u64,
}

impl Xorshift64 {
    pub fn new_with_seed(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    pub fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

pub struct ZobristTable {
    pub man_at: [[u64; LENGTH]; WIDTH],
    pub ball_at: [[u64; LENGTH]; WIDTH],
    pub side_left: u64,
}

impl ZobristTable {
    pub fn new() -> Self {
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

impl Default for ZobristTable {
    fn default() -> Self {
        Self::new()
    }
}

pub fn zobrist_hash(board: &Board, zt: &ZobristTable) -> u64 {
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

pub static ZOBRIST: std::sync::OnceLock<ZobristTable> = std::sync::OnceLock::new();

pub fn get_zobrist() -> &'static ZobristTable {
    ZOBRIST.get_or_init(ZobristTable::new)
}

pub const TT_SIZE: usize = 1 << 17;

#[derive(Clone, Copy)]
pub enum NodeType { Exact, LowerBound, UpperBound }

#[derive(Clone)]
pub struct TTEntry {
    pub hash: u64,
    pub depth: u32,
    pub score: f64,
    pub node_type: NodeType,
    pub best_next_hash: u64,
}

pub struct TTable {
    pub entries: Vec<Option<TTEntry>>,
}

impl TTable {
    pub fn new() -> Self {
        Self { entries: vec![None; TT_SIZE] }
    }

    pub fn probe(&self, hash: u64) -> Option<&TTEntry> {
        let idx = (hash as usize) & (TT_SIZE - 1);
        self.entries[idx].as_ref().filter(|e| e.hash == hash)
    }

    pub fn store(&mut self, hash: u64, depth: u32, score: f64, node_type: NodeType, best_next_hash: u64) {
        let idx = (hash as usize) & (TT_SIZE - 1);
        self.entries[idx] = Some(TTEntry { hash, depth, score, node_type, best_next_hash });
    }
}

impl Default for TTable {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Evaluators
// ============================================================================

pub struct LocationEvaluator;

impl LocationEvaluator {
    pub fn score(board: &Board) -> f64 {
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

pub struct RichEvaluator;

impl RichEvaluator {
    pub fn score(board: &Board) -> f64 {
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

pub struct Eval4Evaluator;

impl Eval4Evaluator {
    pub fn count_consecutive_men(board: &Board, from: Position, dir: Direction) -> usize {
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

    pub fn max_jump_chain(board: &Board) -> usize {
        let mut best = 0;
        for dir in Direction::ALL {
            let chain = Self::count_consecutive_men(board, board.ball_at, dir);
            best = best.max(chain);
        }
        best
    }

    pub fn score(board: &Board) -> f64 {
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

pub struct JumpChainEvaluator;

impl JumpChainEvaluator {
    pub fn count_chain(board: &Board, from: Position, dir: Direction) -> usize {
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

    pub fn score(board: &Board) -> f64 {
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

// ============================================================================
// PloddingEngine
// ============================================================================

pub struct PloddingEngine;

impl PloddingEngine {
    pub fn make_move(board: &Board) -> String {
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
                let mv = target.to_notation();
                if moves.contains_key(&mv) {
                    return mv;
                }
            }
        }
        moves.keys().next().unwrap().clone()
    }
}

// ============================================================================
// Eval6Engine: beam-search iterative-deepening alpha-beta
// Authoritative algorithm from main.rs TimedPlayer6, adapted to use now_ms()
// ============================================================================

pub struct Eval6Engine {
    pub budget_ms: u64,
    pub eval_fn: fn(&Board) -> f64,
}

impl Eval6Engine {
    pub fn new(budget_ms: u64) -> Self {
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

    pub fn make_move(&self, board: &Board) -> String {
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
