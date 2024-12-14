import chess
import chess.engine

#Code for Move prediction

#Enter FEN of board
board = chess.Board("rnbqk2r/ppp1ppbp/5np1/3p4/3P4/2PQ1N2/PP2PPPP/RNB1KB1R w - - 0 1")

engine_path = r"Your ENGine Path"

engine = chess.engine.SimpleEngine.popen_uci(engine_path)

result = engine.play(board, chess.engine.Limit(time=1.0))  
print("Best move:", result.move)

engine.quit()
