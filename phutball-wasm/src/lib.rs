#![allow(dead_code)]

use phutball_core::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use yew::prelude::*;
use gloo::timers::callback::Timeout;

// Hash format: #<left>/<right>/<left_secs>/<right_secs>/<move1>/<move2>/...
// e.g.  #human/eval6/1/5/A5/E/B7/E-NE
// Spaces inside a move (chain jumps) are encoded as '-'.

fn valid_spec(s: &str) -> bool {
    matches!(s, "human" | "plodding" | "eval6")
}

fn parse_hash(hash: &str) -> (String, String, u64, u64, Vec<String>) {
    let h = hash.trim_start_matches('#');
    let mut parts = h.split('/');
    let left       = parts.next().unwrap_or("");
    let right      = parts.next().unwrap_or("");
    let left_secs  = parts.next().unwrap_or("").parse::<u64>().unwrap_or(0);
    let right_secs = parts.next().unwrap_or("").parse::<u64>().unwrap_or(0);
    let moves: Vec<String> = parts
        .filter(|s| !s.is_empty())
        .map(|s| s.replace('-', " "))
        .collect();
    if valid_spec(left) && valid_spec(right) && left_secs >= 1 && right_secs >= 1 {
        (left.to_string(), right.to_string(), left_secs, right_secs, moves)
    } else {
        ("eval6".to_string(), "human".to_string(), 1, 1, vec![])
    }
}

fn build_hash(left: &str, right: &str, ls: u64, rs: u64, history: &[String]) -> String {
    let mut parts = vec![left.to_string(), right.to_string(), ls.to_string(), rs.to_string()];
    parts.extend(history.iter().map(|m| m.replace(' ', "-")));
    parts.join("/")
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
    left_budget_ms: u64,
    right_budget_ms: u64,
    ball_selected: bool,
    jump_dests: Vec<(String, usize, usize)>,
    history: Vec<String>,
}

impl GameState {
    fn new(left: &str, right: &str, left_ms: u64, right_ms: u64) -> Self {
        Self {
            board: Board::new(),
            left_spec: left.to_string(),
            right_spec: right.to_string(),
            left_budget_ms: left_ms,
            right_budget_ms: right_ms,
            ball_selected: false,
            jump_dests: vec![],
            history: vec![],
        }
    }

    fn current_engine(&self) -> Engine {
        match self.board.side_to_move {
            Side::Left  => Engine::from_spec(&self.left_spec,  self.left_budget_ms),
            Side::Right => Engine::from_spec(&self.right_spec, self.right_budget_ms),
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
            self.history.push(mv.trim().to_string());
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
            self.history.push(mv.trim().to_string());
            self.board = new_board.clone();
            self.ball_selected = false;
            self.jump_dests.clear();
            return true;
        }
        let with_space = format!("{} ", mv);
        if let Some(new_board) = moves.get(&with_space) {
            self.history.push(mv.trim().to_string());
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
    let init_hash = web_sys::window()
        .and_then(|w| w.location().hash().ok())
        .unwrap_or_default();
    let (init_left, init_right, init_ls, init_rs, init_moves) = parse_hash(&init_hash);

    let left_spec         = use_state(|| init_left.clone());
    let right_spec        = use_state(|| init_right.clone());
    let left_budget_secs  = use_state(|| init_ls);
    let right_budget_secs = use_state(|| init_rs);
    let game = use_state(|| {
        let mut gs = GameState::new(&init_left, &init_right, init_ls * 1000, init_rs * 1000);
        for mv in init_moves {
            if !gs.play_human_move(&mv) { break; }
        }
        gs
    });
    let move_input: UseStateHandle<String> = use_state(|| String::new());
    let preview_board: UseStateHandle<Option<Board>> = use_state(|| None);
    let move_error: UseStateHandle<Option<String>> = use_state(|| None);

    // Keep URL hash in sync with full game state
    {
        let game              = game.clone();
        let left_spec         = left_spec.clone();
        let right_spec        = right_spec.clone();
        let left_budget_secs  = left_budget_secs.clone();
        let right_budget_secs = right_budget_secs.clone();
        use_effect_with(
            ((*game).history.clone(), (*left_spec).clone(), (*right_spec).clone(),
             *left_budget_secs, *right_budget_secs),
            move |(history, left, right, ls, rs)| {
                if let Some(window) = web_sys::window() {
                    let _ = window.location().set_hash(&build_hash(left, right, *ls, *rs, history));
                }
                || ()
            },
        );
    }

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
        let game              = game.clone();
        let left_spec         = left_spec.clone();
        let right_spec        = right_spec.clone();
        let left_budget_secs  = left_budget_secs.clone();
        let right_budget_secs = right_budget_secs.clone();
        let move_input    = move_input.clone();
        let preview_board = preview_board.clone();
        let move_error    = move_error.clone();
        Callback::from(move |_: MouseEvent| {
            game.set(GameState::new(&left_spec, &right_spec,
                                    *left_budget_secs * 1000, *right_budget_secs * 1000));
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

    // Budget text inputs
    let on_left_budget_change = {
        let left_budget_secs = left_budget_secs.clone();
        Callback::from(move |e: web_sys::Event| {
            let el = e.target_unchecked_into::<web_sys::HtmlInputElement>();
            if let Ok(v) = el.value().parse::<u64>() {
                if v >= 1 { left_budget_secs.set(v); }
            }
        })
    };
    let on_right_budget_change = {
        let right_budget_secs = right_budget_secs.clone();
        Callback::from(move |e: web_sys::Event| {
            let el = e.target_unchecked_into::<web_sys::HtmlInputElement>();
            if let Ok(v) = el.value().parse::<u64>() {
                if v >= 1 { right_budget_secs.set(v); }
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

    let g   = &*game;
    let status = g.status_text();
    let lv  = (*left_spec).clone();
    let rv  = (*right_spec).clone();
    let lbs = *left_budget_secs;
    let rbs = *right_budget_secs;
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
                if lv == "eval6" {
                    <label>{"Left secs: "}
                        <input type="number" min="1" max="3600" step="1"
                               value={lbs.to_string()}
                               onchange={on_left_budget_change}
                               style="width:60px;"/>
                    </label>
                }
                <label>{"Right: "}
                    <select onchange={on_right_change}>
                        <option value="human"    selected={rv == "human"}>{"Human"}</option>
                        <option value="plodding" selected={rv == "plodding"}>{"Plodding"}</option>
                        <option value="eval6"    selected={rv == "eval6"}>{"Beam Search"}</option>
                    </select>
                </label>
                if rv == "eval6" {
                    <label>{"Right secs: "}
                        <input type="number" min="1" max="3600" step="1"
                               value={rbs.to_string()}
                               onchange={on_right_budget_change}
                               style="width:60px;"/>
                    </label>
                }
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
        let gs = GameState::new("human", "human", 500, 500);
        assert_eq!(gs.board.moves_made, 0);
        assert!(gs.winner().is_none());
        assert!(!gs.ball_selected);

        // Plodding engine doesn't use js_sys timing, safe to call in native tests
        let mut gs2 = GameState::new("plodding", "plodding", 100, 100);
        gs2.play_engine_move();
        assert_eq!(gs2.board.moves_made, 1);
    }
}
