Overall goal: Make a phutball engine that's as strong as possible, given 1s of walltime, single-cored.

Currently the 1s of walltime isn't coded anywhere; you'll have to code that.

Remember to control for the fact that other processes might be using the laptop, although I'll try to not use it heavily. Or if you assign tasks to multiple polecats, they might be using up CPU---but the single-cored requirement should help with that.

Process: Make a tournament subcommand that compares two engines. It randomly assigns them to a side, then places 4 pieces at random on the board, then runs the game (after 100 moves, declare tie). Then repeat. Assume uniform prior and hence beta posterior, and the tournament continues until there's a 90% change one engine is stronger.

That's how you compare a new engine to the current best engine.

Now, the current best engine is likely minimax (remember the parallel minimax isn't allowed). Can we do better? e.g. right now it doesn't handle the 1s cutoff. What about MCTS? A handcoded eval function for board state? AlphaZero? (which will require some ML---use burn.rs for that)

Like, the end goal is something like AlphaZero. I want you to make improvement towards that, picking the low-hanging fruit first
