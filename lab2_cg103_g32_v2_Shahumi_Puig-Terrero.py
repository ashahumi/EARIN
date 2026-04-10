import copy
import math

EMPTY = ' '
PLAYER_X = 'X'
PLAYER_O = 'O'
ROWS = 6
COLS = 7


class ConnectFour:
    """
    Class for game Connect 4
    """
    def __init__(self):
        self.board = [[EMPTY] * COLS for _ in range(ROWS)]
        self.current_player = PLAYER_X

    def print_board(self):
        print('\n')
        for row in self.board:
            print('|' + '|'.join(row) + '|')
        print('-' * (COLS * 2 + 1))
        print(' ' + ' '.join(str(i) for i in range(COLS)))
        print('\n')

    # --- Additional Helper Functions ---

    def is_valid_location(self, board, col):
        """Check if the top row of the given column is empty."""
        return board[0][col] == EMPTY

    def get_valid_locations(self, board):
        """Return a list of columns that are not full."""
        return [col for col in range(COLS) if self.is_valid_location(board, col)]

    def get_next_open_row(self, board, col):
        """Find the lowest available row in a specific column."""
        for r in range(ROWS - 1, -1, -1):
            if board[r][col] == EMPTY:
                return r
        return None

    def drop_piece(self, board, row, col, piece):
        """Place a piece on the board."""
        board[row][col] = piece

    def winning_move(self, board, piece):
        """Check the board for a 4-in-a-row horizontal, vertical, or diagonal."""
        # Horizontal
        for c in range(COLS - 3):
            for r in range(ROWS):
                if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                    return True
        # Vertical
        for c in range(COLS):
            for r in range(ROWS - 3):
                if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                    return True
        # Positive Slope Diagonal
        for c in range(COLS - 3):
            for r in range(ROWS - 3):
                if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                    return True
        # Negative SlopeDiagonal
        for c in range(COLS - 3):
            for r in range(3, ROWS):
                if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                    return True
        return False

    def is_terminal_node(self, board):
        """Check if the game is over (win or draw)."""
        return self.winning_move(board, PLAYER_X) or self.winning_move(board, PLAYER_O) or len(self.get_valid_locations(board)) == 0

    # --- Evaluation and AI Functions ---

    def evaluate_window(self, window, piece):
        """
        Evaluation of given window. Helper function to evaluate the separate parts of the board called windows
        """
        score = 0
        opp_piece = PLAYER_O if piece == PLAYER_X else PLAYER_X

        if window.count(piece) == 4:
            score += 100
        elif window.count(piece) == 3 and window.count(EMPTY) == 1:
            score += 5
        elif window.count(piece) == 2 and window.count(EMPTY) == 2:
            score += 2

        # Block the opponent
        if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
            score -= 4

        return score

    def evaluate_position(self, board, piece):
        """
        Evaluation of position
        """
        score = 0

        # Score center column (prioritize playing in the middle)
        center_array = [board[i][COLS//2] for i in range(ROWS)]
        center_count = center_array.count(piece)
        score += center_count * 3

        # Score Horizontal
        for r in range(ROWS):
            row_array = board[r]
            for c in range(COLS - 3):
                window = row_array[c:c+4]
                score += self.evaluate_window(window, piece)

        # Score Vertical
        for c in range(COLS):
            col_array = [board[r][c] for r in range(ROWS)]
            for r in range(ROWS - 3):
                window = col_array[r:r+4]
                score += self.evaluate_window(window, piece)

        # Score Positive Diagonal
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                window = [board[r+i][c+i] for i in range(4)]
                score += self.evaluate_window(window, piece)

        # Score Negative Diagonal
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                window = [board[r+3-i][c+i] for i in range(4)]
                score += self.evaluate_window(window, piece)

        return score

    def minimax(self, board, depth, maximizing_player, alpha, beta, ai_piece):
        """
        Minimax with alpha-beta pruning algorithm
        """
        valid_locations = self.get_valid_locations(board)
        is_terminal = self.is_terminal_node(board)
        player_piece = PLAYER_O if ai_piece == PLAYER_X else PLAYER_X

        if depth == 0 or is_terminal:
            if is_terminal:
                if self.winning_move(board, ai_piece):
                    return (math.inf, None)
                elif self.winning_move(board, player_piece):
                    return (-math.inf, None)
                else:
                    return (0, None)
            else:
                return (self.evaluate_position(board, ai_piece), None)

        if maximizing_player:
            value = -math.inf
            best_col = valid_locations[0] if valid_locations else None
            for col in valid_locations:
                row = self.get_next_open_row(board, col)
                b_copy = copy.deepcopy(board)
                self.drop_piece(b_copy, row, col, ai_piece)
                new_score = self.minimax(b_copy, depth - 1, False, alpha, beta, ai_piece)[0]
                
                if new_score > value:
                    value = new_score
                    best_col = col
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value, best_col

        else: # Minimizing player
            value = math.inf
            best_col = valid_locations[0] if valid_locations else None
            for col in valid_locations:
                row = self.get_next_open_row(board, col)
                b_copy = copy.deepcopy(board)
                self.drop_piece(b_copy, row, col, player_piece)
                new_score = self.minimax(b_copy, depth - 1, True, alpha, beta, ai_piece)[0]
                
                if new_score < value:
                    value = new_score
                    best_col = col
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value, best_col


def main():
    """
    Main game loop implementation. Player1 should play first with 'X', player2 plays second with 'O'
    """
    print("Welcome to Connect 4!")
    print("Do you want to play first or second?")
    
    choice = ''
    while choice not in ['1', '2']:
        choice = input("Enter '1' to play first (X) or '2' to play second (O): ").strip()
    
    if choice == '1':
        human_piece = PLAYER_X
        ai_piece = PLAYER_O
        print("You are playing as 'X'. You go first.")
    else:
        human_piece = PLAYER_O
        ai_piece = PLAYER_X
        print("You are playing as 'O'. The computer goes first as 'X'.")

    game = ConnectFour()
    game_over = False
    turn_piece = PLAYER_X # 'X' always goes first

    game.print_board()

    while not game_over:
        # HUMAN TURN
        if turn_piece == human_piece:
            valid_input = False
            while not valid_input:
                try:
                    col = int(input(f"Your turn ({human_piece}). Choose a column (0-6): "))
                    if 0 <= col <= 6:
                        if game.is_valid_location(game.board, col):
                            row = game.get_next_open_row(game.board, col)
                            game.drop_piece(game.board, row, col, human_piece)
                            valid_input = True
                        else:
                            print("Column is full! Choose another one.")
                    else:
                        print("Invalid column. Please choose a number between 0 and 6.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            
            if game.winning_move(game.board, human_piece):
                game.print_board()
                print("Congratulations! You won!")
                game_over = True

        # AI TURN
        else:
            print(f"Computer is thinking ({ai_piece})...")
            # Set depth to 4 or 5 for a good balance of speed and difficulty
            score, col = game.minimax(game.board, 5, True, -math.inf, math.inf, ai_piece)
            
            if col is not None:
                row = game.get_next_open_row(game.board, col)
                game.drop_piece(game.board, row, col, ai_piece)
                print(f"Computer drops a piece in column {col}")

            if game.winning_move(game.board, ai_piece):
                game.print_board()
                print("Game Over. The computer wins!")
                game_over = True

        if not game_over:
            game.print_board()
            
            # Check for draw
            if len(game.get_valid_locations(game.board)) == 0:
                print("Board is full. The game is a draw!")
                game_over = True
            
            # Switch turn
            turn_piece = PLAYER_O if turn_piece == PLAYER_X else PLAYER_X

def run_automated_tests():
    print("="*40)
    print("      RUNNING AUTOMATED AI TESTS")
    print("="*40)

    # Format of test cases: ("Test Name", [list of moves], Expected AI Column, AI's Piece)
    test_cases = [
        (
            "Test 1: AI must block an immediate horizontal win",
            [0, 0, 1, 1, 2], # X plays 0, O plays 0, X plays 1... X is threatening column 3
            3,               # AI MUST play 3 to block
            PLAYER_O         # AI is playing as O
        ),
        (
            "Test 2: AI must block an immediate vertical win",
            [4, 2, 4, 3, 4], # X plays 4 three times. X is threatening the top of column 4.
            4,               # AI MUST play 4 to block
            PLAYER_O
        ),
        (
            "Test 3: AI takes an immediate win instead of blocking",
            [0, 1, 0, 1, 0, 1, 6], # O has three in column 1. X has three in col 0. 
            1,                     # AI (O) should play 1 to win the game, ignoring X's threat.
            PLAYER_O
        ),
        (
            "Test 4: AI as Player 1 (X) builds a vertical win",
            [3, 2, 3, 2, 3, 4], # X has three in col 3. 
            3,                  # AI (X) should play 3 to win.
            PLAYER_X
        ),
        (
            "Test 5: Draw case (Board almost full, 2 moves left)",
            [
                # Fill columns 0 and 1 without winning
                0, 1, 0, 1, 0, 1,
                1, 0, 1, 0, 1, 0,
                # Fill columns 2 and 3 without winning
                2, 3, 2, 3, 2, 3,
                3, 2, 3, 2, 3, 2,
                # Fill columns 4 and 5 without winning
                4, 5, 4, 5, 4, 5,
                5, 4, 5, 4, 5, 4,
                # Fill 4 pieces into column 6, leaving only 2 empty spots on the entire board
                6, 6, 6, 6
            ],
            6,               # AI MUST play 6 because it's the only column with empty space
            PLAYER_X         # 40 moves have passed, so it's X's turn again
        )
    ]

    passed = 0
    for name, moves, expected_col, ai_piece in test_cases:
        print(f"\n" + "-"*50)
        print(f"Running {name}...")
        
        # 1. Set up a fresh game board
        game = ConnectFour()
        current_piece = PLAYER_X
        
        # 2. Apply the predetermined moves
        for col in moves:
            row = game.get_next_open_row(game.board, col)
            game.drop_piece(game.board, row, col, current_piece)
            # Swap turns
            current_piece = PLAYER_O if current_piece == PLAYER_X else PLAYER_X

        print("\nBoard state BEFORE AI move:")
        game.print_board()
        print(f"AI ({ai_piece}) is thinking...")

        # 3. Ask AI for its move
        score, chosen_col = game.minimax(game.board, 5, True, -math.inf, math.inf, ai_piece)
        
        # 4. Apply the AI's move and print the new board
        if chosen_col is not None:
            row = game.get_next_open_row(game.board, chosen_col)
            game.drop_piece(game.board, row, chosen_col, ai_piece)
            print(f"\nBoard state AFTER AI drops piece in column {chosen_col}:")
            game.print_board()

        # 5. Check if AI did what we expected
        if chosen_col == expected_col:
            print(f"✅ PASS: AI chose column {chosen_col} as expected.")
            passed += 1
        else:
            print(f"❌ FAIL: AI chose column {chosen_col}, but we expected {expected_col}.")

    print("\n" + "="*40)
    print(f"TEST RESULTS: {passed} / {len(test_cases)} tests passed.")
    print("="*40 + "\n")


if __name__ == "__main__":
    mode = input("Enter '1' to play the game, or '2' to run automated tests: ")
    if mode == '2':
        run_automated_tests()
    else:
        main()