from stockfish import Stockfish

# Stockfish motorunun sistemdeki yolu
stockfish = Stockfish("/usr/games/stockfish")  # Linux için
# Windows'ta örnek: "C:\\Users\\...\\stockfish\\stockfish.exe"

# Motorun çalıştığını test et
print(stockfish._is_ready())

# Bir pozisyon verelim (FEN formatında)
stockfish.set_fen_position("2k3rr/p1pq4/1pb1p3/4Np2/3P4/2P1Q1bP/PP4P1/RN3R1K b - - 1 19")

# En iyi hamleyi bul
print(stockfish.get_best_move())

# Alternatif: en iyi 3 hamleyi iste
print(stockfish.get_top_moves(3))