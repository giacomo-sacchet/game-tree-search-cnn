import tkinter as tk
from tkinter import messagebox
import random
import numpy as np
import time


#################################################################################
# BLOCK 1 - Global variables and constants

# Initialize empty grid 6x7
Grid = np.zeros((6, 7), dtype=int)

# Scores and Game State
score_h = 0         # Human score
score_ia = 0        # IA score
score_ia_green = 0  # Green AI score

grid_color = "blue" # Default color
StartMatch = False  # State variable to manage resets

# 3-Player Mode Toggle
# False = Human vs 1 AI | True = Human vs Red AI vs Green AI
ThreePlayerMode = False

# Turn Tracking variable
# 1: Human, 2: Red AI, 3: Green AI 
current_turn = 1

# PLAYER TYPES: "Human", "1" (Random), "2" (Heuristic), "3" (Minimax)
player1_type = "Human" 
player2_type = "3"      # Default to Minimax


#################################################################################
# BLOCK 2 - Game logic

# Returns a list of column indices where a token can still be placed
def GetPossibleMoves():
    possible_moves = []
    for col in range(7):
        # If the top row (0) of a column is empty, it is not full yet
        if Grid[0][col] == 0:
            possible_moves.append(col)
            
    return possible_moves

# Checks if the specified player has met their specific winning criteria
def WinnerCheck(player):
    # Standard 4-in-a-row Check for Players 1 (Human) and 2 (Red AI)
    if player == 1 or player == 2:
        # 1. Horizontal Check 
        for r in range(6):
            for c in range(4): 
                if Grid[r][c] == player and Grid[r][c+1] == player and \
                   Grid[r][c+2] == player and Grid[r][c+3] == player:
                    return True

        # 2. Vertical Check 
        for r in range(3): 
            for c in range(7):
                if Grid[r][c] == player and Grid[r+1][c] == player and \
                   Grid[r+2][c] == player and Grid[r+3][c] == player:
                    return True

        # 3. Positively Sloped Diagonal (/) 
        for r in range(3, 6):
            for c in range(4):
                if Grid[r][c] == player and Grid[r-1][c+1] == player and \
                   Grid[r-2][c+2] == player and Grid[r-3][c+3] == player:
                    return True

        # 4. Negatively Sloped Diagonal (\) 
        for r in range(3):
            for c in range(4):
                if Grid[r][c] == player and Grid[r+1][c+1] == player and \
                   Grid[r+2][c+2] == player and Grid[r+3][c+3] == player:
                    return True

    # Square 2x2 check for Player 3 (Green AI) 
    elif player == 3:
        # A 6x7 grid allows squares starting up to row index 4 and col index 5
        for r in range(5): 
            for c in range(6): 
                if Grid[r][c] == 3 and Grid[r+1][c] == 3 and \
                   Grid[r][c+1] == 3 and Grid[r+1][c+1] == 3:
                    return True

    return False

# Determines the current state of the game
def CheckGameState():
    if WinnerCheck(1): return 1 # Human wins
    if WinnerCheck(2): return 2 # Red AI wins
    if ThreePlayerMode and WinnerCheck(3): return 3 # Green AI wins (2x2 square)
    if not 0 in Grid: return 4 # Draw
    
    return 0

# Handles score updates and grid color changes
def EndMatch(winner):
    global score_h, score_ia, score_ia_green, grid_color, StartMatch
    
    if winner == 1:
        score_h += 1
        grid_color = "yellow"
    elif winner == 2:
        score_ia += 1
        grid_color = "red"
    elif winner == 3:
        score_ia_green += 1
        grid_color = "green"
    elif winner == 4: # Draw
        grid_color = "white"
    
    StartMatch = True

def PlayHuman(x,y):
    Grid[x][y] = 1


#################################################################################
# BLOCK 3 - AI

# Helper function to find the lowest available row in a column
def FirstFreeRow(column):
    # Check from bottom (row 5) to top (row 0)
    for row in range(5, -1, -1):
        if Grid[row][column] == 0:
            return row
    return -1 # Column is full

# 1. RANDOM AI
def PlayAIRandom(player_id):
    possible_cols = GetPossibleMoves()
    if possible_cols:
        chosen_col = random.choice(possible_cols)
        row = FirstFreeRow(chosen_col)
        Grid[row][chosen_col] = player_id


# Evaluates the potential of a specific position for a given player
# Returns the max number of tokens the player would have in any window of four including its position
def GetMaxAlignment(row, col, player):
    # Temporarily place the token to evaluate the resulting board state
    Grid[row][col] = player
    max_count = 0
    
    # Directions to check: Horizontal, Vertical, Diagonal(/), Diagonal(\)
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    for dr, dc in directions:
        # For each direction, there are 4 possible windows of 4 that contain (row, col)
        # Check start offsets from -3 to 0
        for start_offset in range(-3, 1):
            window = []
            is_valid_window = True
            
            for i in range(4):
                r_idx = row + (start_offset + i) * dr
                c_idx = col + (start_offset + i) * dc
                
                # Check if the window is within the 6x7 grid boundaries
                if 0 <= r_idx < 6 and 0 <= c_idx < 7:
                    cell_value = Grid[r_idx][c_idx]
                    # A window is only useful if it doesn't contain the opponent's pieces
                    if cell_value != 0 and cell_value != player:
                        is_valid_window = False
                        break
                    window.append(cell_value)
                else:
                    is_valid_window = False
                    break
            
            if is_valid_window:
                # Count tokens in this potential 4-slot alignment
                # Note: non-contiguous tokens are counted 
                count = sum(1 for cell in window if cell == player)
                if count > max_count:
                    max_count = count

    # Remove the temporary token before returning
    Grid[row][col] = 0
    return max_count


# Helper to calculate the score for a specific move
def GetScoreForPosition(row, col, player):
    potential = GetMaxAlignment(row, col, player)
    opponent = 1 if player == 2 else 2
    opp_potential = GetMaxAlignment(row, col, opponent)
    
    if potential == 4: return 100
    if opp_potential == 4: return 50
    if potential == 3: return 30
    if opp_potential == 3: return 15
    if potential == 2: return 10
    return 0


# 2. HEURISTIC AI - choose the best move based on the point system
def PlayAIHeuristic(player_id):
    possible_cols = GetPossibleMoves()
    if not possible_cols:
        return
    
    best_score = -1
    best_moves = []
    
    for col in possible_cols:
        row = FirstFreeRow(col)
        # Calculate potential for the current player and their opponent
        current_move_score = GetScoreForPosition(row, col, player_id)
        
        if current_move_score > best_score:
            best_score = current_move_score
            best_moves = [col]
        elif current_move_score == best_score:
            best_moves.append(col)
    
    chosen_col = random.choice(best_moves)
    row = FirstFreeRow(chosen_col)
    Grid[row][chosen_col] = player_id


# Helper evaluation of the grid for minimax algo
def EvaluateGrid(p_id, opp_id):
    state = CheckGameState()
    
    # Immediate terminal states
    if state == p_id: return 500    # AI Wins
    if state == opp_id: return -500 # Human Wins
    if state == 4: return 0         # Draw
    
    p_scores = []
    opp_scores = []
    
    # Evaluate the board based on current alignments
    for col in range(7):
        row = FirstFreeRow(col)
        if row != -1:
            # Score for AI
            p_scores.append(GetScoreForPosition(row, col, p_id))
            # Score for human opponent
            opp_scores.append(GetScoreForPosition(row, col, opp_id))

    max_p = max(p_scores) if p_scores else 0
    max_opp = max(opp_scores) if opp_scores else 0
    
    # The heuristic returns the advantage of the AI over the human
    return max_p - max_opp


# Minimax algo
def Minimax(depth, is_maximizing, p_id, opp_id):
    state = CheckGameState()
    # Base case: depth reached or game ended
    if depth == 0 or state != 0:
        return EvaluateGrid(p_id, opp_id)

    possible_moves = GetPossibleMoves()
    
    if is_maximizing:
        best_val = -float('inf')
        for col in possible_moves:
            row = FirstFreeRow(col)
            Grid[row][col] = p_id 
            value = Minimax(depth - 1, False, p_id, opp_id)
            Grid[row][col] = 0 
            best_val = max(best_val, value)
        return best_val
    else:
        # Minimizing the opponent's advantage
        best_val = float('inf')
        for col in possible_moves:
            row = FirstFreeRow(col)
            Grid[row][col] = opp_id 
            value = Minimax(depth - 1, True, p_id, opp_id)
            Grid[row][col] = 0 
            best_val = min(best_val, value)
        return best_val


# 3. MINIMAX AI
def PlayAIMinimax(player_id, depth=3):
    # In 1v1 mode, the opponent is always Human (1)
    opp_id = 1

    best_score = -float('inf')
    best_moves = []
    possible_moves = GetPossibleMoves()
    
    if not possible_moves: return

    for col in possible_moves:
        row = FirstFreeRow(col)
        Grid[row][col] = player_id                              # Simulate move
        score = Minimax(depth - 1, False, player_id, opp_id)    # Evaluate this move using Minimax
        Grid[row][col] = 0                                      # Undo
        
        if score > best_score:
            best_score = score
            best_moves = [col]
        elif score == best_score:
            best_moves.append(col)
            
    # Choose randomly among the best evaluated moves
    chosen_col = random.choice(best_moves)
    row = FirstFreeRow(chosen_col)
    Grid[row][chosen_col] = player_id


# Helper to score a 2x2 window for the Green AI (Player 3)
# Evaluate potential to create a 2x2 square in the position (r_move, c_move)
def GetSquareScore(player_id, r_move, c_move):
    max_square_score = 0
    # A 2x2 square containing (r_move, c_move) can start from 4 top-left positions:
    # (r_move, c_move), (r_move-1, c_move), (r_move, c_move-1), (r_move-1, c_move-1)
    for dr in range(-1, 1):
        for dc in range(-1, 1):
            r_start, c_start = r_move + dr, c_move + dc
            
            # Ensure square boundaries are within the grid
            if 0 <= r_start < 5 and 0 <= c_start < 6:
                # Extract the 4 cells forming this specific square
                window = [
                    Grid[r_start][c_start], Grid[r_start+1][c_start],
                    Grid[r_start][c_start+1], Grid[r_start+1][c_start+1]
                ]
                
                # If any cell in the square is occupied by an opponent it's blocked
                if 1 in window or 2 in window:
                    continue
                
                # Count tokens already placed by the Green AI
                count = window.count(player_id)
                
                # Assign score based on proximity to completion
                if count == 4: current_score = 100  # Victory
                elif count == 3: current_score = 60 # One token away
                elif count == 2: current_score = 20 # Halfway
                else: current_score = 5             # Starting a square
                
                if current_score > max_square_score:
                    max_square_score = current_score
                
    return max_square_score


# 4. HEURISTIC AI PL3
def Play3AIHeuristic(player_id):
    possible_cols = GetPossibleMoves()
    if not possible_cols:
        return
    
    best_total_score = -1
    best_moves = []
    
    for col in possible_cols:
        row = FirstFreeRow(col)
        
        # 1. OFFENSE: Calculate progress toward the 2x2 square
        # Temporarily place token to check if it completes a square
        Grid[row][col] = 3
        off_score = GetSquareScore(3, row, col)
        Grid[row][col] = 0
        
        # 2. DEFENSE: Check if this move blocks the Human (Player 1)
        # Ignore Red AI (Player 2) threats
        def_score_human = GetScoreForPosition(row, col, 1)
        
        # 3. TOTAL EVALUATION
        # Priority 1: Winning (off_score == 100)
        # Priority 2: Blocking a human win (def_score_human >= 50)
        # Priority 3: Developing its own square
        if off_score == 100:
            total_move_score = 1000 # Immediate win priority
        elif def_score_human >= 50:
            total_move_score = 500  # Immediate block priority
        else:
            total_move_score = off_score
        
        # Track the best possible moves
        if total_move_score > best_total_score:
            best_total_score = total_move_score
            best_moves = [col]
        elif total_move_score == best_total_score:
            best_moves.append(col)
    
    # Execute the chosen move
    chosen_col = random.choice(best_moves)
    row = FirstFreeRow(chosen_col)
    Grid[row][chosen_col] = player_id


# Dictionary mapping
AI_FUNCTIONS = {
    "1": PlayAIRandom,
    "2": PlayAIHeuristic,
    "3": PlayAIMinimax,
    "4": Play3AIHeuristic,
    "5": lambda player_id: PlayAIDescent(player_id)
}


#################################################################################
# BLOCK 3b - Descent Minimax AI (Cohen-Solal & Cazenave, AAMAS 2023)
#
# Replaces the neural network with the existing heuristic EvaluateGrid().
# Replaces the terminal evaluator f_t with EvaluateGrid() on terminal states.
#
# Key data structures:
#   S  : dict mapping board_key -> {"value": float, "turn": int}
#        board_key = tuple(Grid.flatten()) - a hashable snapshot of the board
#   v  : the value table lives inside S (accessed via board_key)
#
# Because Connect 4 is a two-player alternating game, the turn of the player
# (whose move it is) must be part of the state key so that the same board
# position reached on different turns is treated correctly.
#
# The value stored in v(s) is always from the perspective of p_id (the AI),
# matching EvaluateGrid's sign convention.


# Return a hashable key for the current board state + whose turn it is
def _board_key(turn):
    return (tuple(Grid.flatten()), turn)


# Unified evaluation: works for both terminal and non-terminal states
def _heuristic_eval(p_id, opp_id):
    return EvaluateGrid(p_id, opp_id)


# Place a token; return the row used (or -1 if column is full)
def _apply_move(col, player):
    row = FirstFreeRow(col)
    if row != -1:
        Grid[row][col] = player
    return row


# Remove a token placed at (row, col)
def _undo_move(row, col):
    Grid[row][col] = 0


# One recursive iteration of Descent Minimax
# turn   : int  - player whose move it is right now (p_id or opp_id)
# S      : dict - partial game tree: board_key -> node dict with 'value'
# p_id   : int  - the AI player we are maximising for
# opp_id : int  - the opponent
# Returns the (updated) minimax value of the current state from p_id's view
def descent_iter(turn, S, p_id, opp_id):
    key = _board_key(turn)
    state = CheckGameState()

    # Terminal state
    if state != 0:
        val = _heuristic_eval(p_id, opp_id)   # f_t: exact terminal score
        S[key] = {"value": val, "turn": turn}
        return val

    # Non-terminal, first visit: expansion
    if key not in S:
        S[key] = {"value": None, "turn": turn, "edge": {}}
        next_turn = opp_id if turn == p_id else p_id
        possible_moves = GetPossibleMoves()

        for col in possible_moves:
            row = _apply_move(col, turn)
            child_key = _board_key(next_turn)
            child_state = CheckGameState()

            if child_state != 0:
                # Child is terminal: evaluate exactly (f_t)
                child_val = _heuristic_eval(p_id, opp_id)
                S[child_key] = {"value": child_val, "turn": next_turn}
                S[key]["edge"][col] = child_val
            else:
                # Child is non-terminal: estimate with heuristic
                S[key]["edge"][col] = _heuristic_eval(p_id, opp_id)

            _undo_move(row, col)

    # Choose best action (minimax selection)
    edges = S[key]["edge"]
    if turn == p_id:
        best_col = max(edges, key=lambda c: edges[c])
    else:
        best_col = min(edges, key=lambda c: edges[c])

    # Recurse down the best child (the "descent" step)
    next_turn = opp_id if turn == p_id else p_id
    row = _apply_move(best_col, turn)
    updated_val = descent_iter(next_turn, S, p_id, opp_id)
    _undo_move(row, best_col)

    # Update the edge value with the refined estimate from the recursion
    S[key]["edge"][best_col] = updated_val

    # Recompute best action after update and propagate to node value
    if turn == p_id:
        best_col = max(edges, key=lambda c: edges[c])
    else:
        best_col = min(edges, key=lambda c: edges[c])

    S[key]["value"] = edges[best_col]
    return S[key]["value"]


# Outer time-bounded loop: repeatedly call descent_iter for tau seconds
# Returns the accumulated partial game tree S
def descent(start_turn, S, p_id, opp_id, tau):
    t0 = time.time()
    while time.time() - t0 < tau:
        descent_iter(start_turn, S, p_id, opp_id)
    return S


# 5. DESCENT MINIMAX AI
# Entry point for the Descent AI
# Uses descent() to build a partial minimax game tree within the time
# budget tau seconds, then plays the action with the best edge value from the root
def PlayAIDescent(player_id, tau=1.0):
    opp_id = 1  # In 1v1 mode the opponent is always the Human (Player 1)
    possible_moves = GetPossibleMoves()
    if not possible_moves:
        return

    S = {}  # Fresh partial game tree for this move decision
    descent(player_id, S, player_id, opp_id, tau)

    # Action selection: pick the column with the highest edge value at the root
    root_key = _board_key(player_id)
    if root_key in S and S[root_key].get("edge"):
        edges = S[root_key]["edge"]
        best_score = max(edges.values())
        best_cols = [c for c, v in edges.items() if v == best_score]
        chosen_col = random.choice(best_cols)
    else:
        # Fallback: root not expanded (very short tau), use random choice
        chosen_col = random.choice(possible_moves)

    row = FirstFreeRow(chosen_col)
    if row != -1:
        Grid[row][chosen_col] = player_id

#################################################################################
# BLOCK 4 - Display and interface (events and view)

# Grid drawing updated for 3 players
def Dessine(PartieGagnee = False):
    canvas.delete("all")

    # Draw vertical and horizontal lines (remains blue by default)
    for i in range(8):
        canvas.create_line(i*100, 0, i*100, HAUT, fill=grid_color, width="4")
    for j in range(7):
        canvas.create_line(0, j*100, LARG, j*100, fill=grid_color, width="4")
            
    # Draw tokens based on the 6x7 grid state
    for row in range(6):
        for col in range(7):
            xc = col * 100 
            yc = row * 100 
            
            if Grid[row][col] == 1: # Human: Yellow
                canvas.create_oval(xc+10, yc+10, xc+90, yc+90, fill="yellow", outline="yellow")
            elif Grid[row][col] == 2: # AI Red: Red
                canvas.create_oval(xc+10, yc+10, xc+90, yc+90, fill="red", outline="red")
            elif Grid[row][col] == 3: # AI Green: Green
                canvas.create_oval(xc+10, yc+10, xc+90, yc+90, fill="green", outline="green")
    
    # Scoreboard with 3-player support
    score_text = f"Human: {score_h} | IA Red: {score_ia}"
    if ThreePlayerMode:
        score_text += f" | IA Green: {score_ia_green}"
        
    canvas.create_text(LARG//2, 15, text=score_text, fill="white", font=('Arial', 10, 'bold'))

# Mouse click handling the 3-player rotation
def MouseClick(event):
    global StartMatch, Grid, grid_color, current_turn
    
    # If a match just ended, clicking resets the board
    if StartMatch:
        Grid = np.zeros((6, 7), dtype=int)
        grid_color = "blue"
        StartMatch = False
        current_turn = 1 
        Dessine()
        return

    # 1. HUMAN TURN (Player 1)
    col = event.x // 100
    if not (0 <= col <= 6): return

    row = FirstFreeRow(col)
    if row == -1: return 

    Grid[row][col] = 1
    Dessine()
    
    winner = CheckGameState()
    if winner != 0:
        EndMatch(winner)
        Dessine()
        return

    # 2. RED AI TURN (Player 2)
    Window.update()
    time.sleep(0.1)
    
    if ThreePlayerMode:
        # In 3-player mode, use Heuristic for speed and stability
        PlayAIHeuristic(player_id=2)
    else:
        # In 1v1 mode, use the chosen opponent type
        AI_FUNCTIONS[player2_type](player_id=2)
        
    Dessine()
    
    winner = CheckGameState()
    if winner != 0:
        EndMatch(winner)
        Dessine()
        return

    # 3. GREEN AI TURN (Player 3)
    if ThreePlayerMode:
        Window.update()
        time.sleep(0.1)
        
        # Use the specific square-heuristic for the Green AI
        Play3AIHeuristic(player_id=3)
        Dessine()
        
        winner = CheckGameState()
        if winner != 0:
            EndMatch(winner)
            Dessine()
            return
    

#################################################################################
# BLOCK 5 - Simulation and execution control

# Simulation engine
def RunBackgroundGame(p1_type, p2_type):
    global Grid
    Grid = np.zeros((6, 7), dtype=int) # Reset for new match
    
    while True:
        # Player 1 Turn
        AI_FUNCTIONS[p1_type](player_id=1)
        state = CheckGameState()
        if state != 0: return state # Return 1 (P1 wins) or 3 (Draw)
        
        # Player 2 Turn
        AI_FUNCTIONS[p2_type](player_id=2)
        state = CheckGameState()
        if state != 0: return state # Return 2 (P2 wins) or 3 (Draw)


# Running the challenge
def StartSimulation(num_games, p1_type, p2_type):
    print(f"\n--- SIMULATION: AI {p1_type} vs AI {p2_type} ---")
    results = {1: 0, 2: 0, 3: 0, 4: 0}
    
    for i in range(num_games):
        winner = RunBackgroundGame(p1_type, p2_type)
        results[winner] += 1
        
    print(f"RESULTS: P1 Wins: {results[1]} | P2 Wins: {results[2]} | Draws: {results[4]}")
    print(f"Success Ratio (P2): {(results[2]/num_games)*100:.2f}%")


# Emprirically tests how long the minimax takes at different depths
def TestDepthPerformance():
    global Grid
    # Reset grid to a neutral state for testing
    Grid = np.zeros((6, 7), dtype=int) 
    
    print("\n--- MINIMAX DEPTH PERFORMANCE TEST ---")
    print("Goal: Find the max depth that computes in < 1 second.")
    
    # Test specifically for Player 2 (AI) logic
    p_id = 2
    
    for d in range(1, 10):  # Test depths 1 to 9
        start_time = time.time()
        
        # Simulate the AI
        AI_FUNCTIONS["3"](player_id=p_id, depth=d)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Clear grid for next depth test
        Grid = np.zeros((6, 7), dtype=int)
        
        print(f"Depth {d}: {duration:.4f} seconds")
        
        if duration > 1.0:
            print(f"\n>>> Depth {d} exceeded the 1-second limit.")
            print(f"RESULT: The recommended max depth for your hardware is {d-1}.")
            break
    print("---------------------------------------\n")


# Beginning of the program
# Ask for the choice to run the sim or to play the game
if __name__ == "__main__":
    print("WELCOME TO CONNECT 4")
    print("1: Play vs AI | 2: Simulation | 3: Depth Test | 4: 3-Player Mode")

    mode = input("Select Mode: ")
    if mode not in ['1', '2', '3', '4']:
        print("Input not accepted. Exit")
        exit()

    if mode in ['1', '4']:
        if mode == '4':
            ThreePlayerMode = True
        else:
            print("\nChoose your opponent: 1: Random | 2: Heuristic | 3: Minimax | 5: Descent")
            player2_type = input("Choice: ")
            if player2_type not in ['1', '2', '3', '5']:
                print("Input not accepted. Exit")
                exit()

        # BLOCK 0 - Creation of the screen (moved here to avoid the creation of the window in case the sim is selected)
        # Updated dimensions for a 6x7 grid
        LARG = 700
        HAUT = 600

        Window = tk.Tk()
        Window.geometry(str(LARG)+"x"+str(HAUT))   # taille de la fenetre
        Window.title("ESIEE - Connect 4")

        # création de la frame principale stockant toutes les pages
        F = tk.Frame(Window)
        F.pack(side="top", fill="both", expand=True)
        F.grid_rowconfigure(0, weight=1)
        F.grid_columnconfigure(0, weight=1)

        # gestion des différentes pages
        ListePages  = {}
        PageActive = 0

        def CreerUnePage(id):
            Frame = tk.Frame(F)
            ListePages[id] = Frame
            Frame.grid(row=0, column=0, sticky="nsew")
            return Frame

        def AfficherPage(id):
            global PageActive
            PageActive = id
            ListePages[id].tkraise()
    
        Frame0 = CreerUnePage(0)
        canvas = tk.Canvas(Frame0,width = LARG, height = HAUT, bg ="black" )
        canvas.place(x=0,y=0)
        canvas.bind('<ButtonPress-1>', MouseClick) #Triggers the activation of MouseClick when a click occurs

        # Interface
        AfficherPage(0)
        Dessine()
        Window.mainloop()

    elif mode == '2':
        print("\nSelect AI 1 (Yellow): 1: Random | 2: Heuristic | 3: Minimax | 5: Descent")
        p1 = input("Choice: ")
        if p1 not in ['1', '2', '3', '5']:
            print("Input not accepted. Exit")
            exit()
        print("Select AI 2 (Red): 1: Random | 2: Heuristic | 3: Minimax | 5: Descent")
        p2 = input("Choice: ")
        if p2 not in ['1', '2', '3', '5']:
            print("Input not accepted. Exit")
            exit()
        games = int(input("Number of games to simulate: "))
        StartSimulation(games, p1, p2)
        exit() # End after simulation
   
    elif mode == '3':
        TestDepthPerformance()
        exit()