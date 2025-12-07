

Stage 1: The rules of phutball. Preferably with the code well-organized in anticipation of future stages.

Stage 2: A minimax agent that plays phutball. Evaluation function is how close the ball is to the goal.

Stage 3: Data collection: For each board state, the eventual score after running depth-d minimax. Hence also a way to randomize the first few moves to create variety and hence a lot of data.

Stage 4: A neural net, trained with burn.dev, that predicts the score after running depth-d minimax. Maybe just a CNN? Hence bootstrapping where we play games with the neural net, generate more accurate data on the value of board positions, and hence train a better 2nd-generation neural net. Also worth thinking about optimization at this point---do we collect multiple boards and pass them to the net in a batch?

Stage 5: Something more ambitious like AlphaGo's policy network?

Stage 6: Running on wasm in the browser with yew.
