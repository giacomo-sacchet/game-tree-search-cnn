import tkinter as tk
from tkinter import messagebox
import random
import numpy as np

###############################################################################
# BLOCK 0 - Creation of the screen (do not touch)

LARG = 300
HAUT = 300

Window = tk.Tk()
Window.geometry(str(LARG)+"x"+str(HAUT))   # taille de la fenetre
Window.title("ESIEE - Morpion")


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


#################################################################################
# BLOCK 1 - Global variables and constants (state)

# Initialize empty grid 3x3
Grille = np.zeros((3, 3), dtype=int)

# Scores and Game State
score_h = 0    # Human score
score_ia = 0   # IA score
grid_color = "blue"  # Default color
DebutDePartie = False # State variable to manage resets
           
  
#################################################################################
# BLOCK 2 - Game logic (rules)

# Checks the grid for a winner or a draw
# Returns: 1 (human wins), 2 (IA wins), 3 (draw), 0 (game continues)
def WinnerCheck():
    # Check rows and columns
    for i in range(3):
        if np.all(Grille[i, :] == 1) or np.all(Grille[:, i] == 1): return 1
        if np.all(Grille[i, :] == 2) or np.all(Grille[:, i] == 2): return 2

    # Check diagonals
    diag1 = np.diag(Grille)
    diag2 = np.diag(np.fliplr(Grille))
    if np.all(diag1 == 1) or np.all(diag2 == 1): return 1
    if np.all(diag1 == 2) or np.all(diag2 == 2): return 2

    # Check for Draw (no zeros left)
    if not 0 in Grille: return 3
    
    return 0

# Handles score updates and grid color changes
def EndMatch(winner):
    global score_h, score_ia, grid_color, DebutDePartie
    
    if winner == 1:
        score_h += 1
        grid_color = "red"
    elif winner == 2:
        score_ia += 1
        grid_color = "yellow"
    elif winner == 3:
        grid_color = "white"
    
    DebutDePartie = True

def PlayHuman(x,y):
    Grille[x][y] = 1


#################################################################################
# BLOCK 3 - AI (simulation)

def SimulIA(depth=0):
    # Simulates the IA, returns the best result found: IA (IA win), N (draw), H (human win) and the corresponding coordinates
    # 1. Check if the simulated game is over, returning depth as well
    res = WinnerCheck()
    if res == 2: return None, "IA", depth
    if res == 1: return None, "H", depth
    if res == 3: return None, "N", depth

    # 2. Get list of possible moves (empty cells)
    empty_cells = list(zip(*np.where(Grille == 0)))
    
    Results = []
    
    for (x, y) in empty_cells:
        Grille[x][y] = 2    # Play the move
        _, R, D = SimulHuman(depth + 1)   # Recursive call, increasing depth
        Results.append(((x, y), R, D))
        Grille[x][y] = 0    # Undo the move

    # 3. Return the most interesting result for the IA
    # Priority and Depth logic
    priority_ia = {"IA": 3, "N": 2, "H": 1}
    
    # Fallback
    best_cell, best_result, best_depth = Results[0]
    
    for cell, result, d in Results[1:]:
        # If the outcome is strictly better, take it
        if priority_ia[result] > priority_ia[best_result]:
            best_result = result
            best_cell = cell
            best_depth = d
        # If the outcome is the same, check the depth
        elif priority_ia[result] == priority_ia[best_result]:
            if result == "IA":
                # If it's a win, prefer the shortest path (lowest depth)
                if d < best_depth:
                    best_result = result
                    best_cell = cell
                    best_depth = d
            elif result == "H":
                # If it's a loss, prefer the longest path (highest depth) to delay it
                if d > best_depth:
                    best_result = result
                    best_cell = cell
                    best_depth = d
            elif result == "N":
                # If it's a draw, prefer reaching it faster
                if d < best_depth:
                    best_result = result
                    best_cell = cell
                    best_depth = d
            
    return best_cell, best_result, best_depth

def SimulHuman(depth=0):
    res = WinnerCheck()
    if res == 2: return None, "IA", depth
    if res == 1: return None, "H", depth
    if res == 3: return None, "N", depth

    empty_cells = list(zip(*np.where(Grille == 0)))
    
    Results = []

    for (x, y) in empty_cells:
        Grille[x][y] = 1    # Play move for human
        _, R, D = SimulIA(depth + 1)    # Recursive call to IA turn
        Results.append(((x, y), R, D))
        Grille[x][y] = 0    # Undo move

    priority_human = {"H": 3, "N": 2, "IA": 1}
    
    best_cell, best_result, best_depth = Results[0]
    
    for cell, result, d in Results[1:]:
        if priority_human[result] > priority_human[best_result]:
            best_result = result
            best_cell = cell
            best_depth = d
        elif priority_human[result] == priority_human[best_result]:
            if result == "H":
                # Human wins: prefer shorter paths
                if d < best_depth:
                    best_result = result
                    best_cell = cell
                    best_depth = d
            elif result == "IA":
                # Human loses: prefer longer paths
                if d > best_depth:
                    best_result = result
                    best_cell = cell
                    best_depth = d
            elif result == "N":
                if d < best_depth:
                    best_result = result
                    best_cell = cell
                    best_depth = d
            
    return best_cell, best_result, best_depth

def PlayAI():
    # Entry point for the IA
    # Calls the algorithm to determin the best strategy
    best_move, _ , _ = SimulIA()
    
    if best_move is not None:
        Grille[best_move[0]][best_move[1]] = 2


#################################################################################
# BLOCK 4 - Display and interface (events and view)

# 4.1 Grid drowing
def Dessine(PartieGagnee = False):
    ## DOC canvas : http://tkinter.fdex.eu/doc/caw.html
    canvas.delete("all")

    # Draw grid lines with the current state color
    for i in range(4):
        canvas.create_line(i*100, 0, i*100, 300, fill=grid_color, width="4")
        canvas.create_line(0, i*100, 300, i*100, fill=grid_color, width="4")
            
    for x in range(3):
        for y in range(3):
            xc = x * 100 
            yc = y * 100 
            if ( Grille[x][y] == 1):
                canvas.create_line(xc+10,yc+10,xc+90,yc+90,fill="red", width="4" )
                canvas.create_line(xc+90,yc+10,xc+10,yc+90,fill="red", width="4" )
            if ( Grille[x][y] == 2):
                canvas.create_oval(xc+10,yc+10,xc+90,yc+90,outline="yellow", width="4" )
    
    # Draw scores at the top in white
    canvas.create_text(150, 15, text=f"Human: {score_h}  |  IA: {score_ia}", fill="white", font=('Arial', 12, 'bold'))

# 4.2 Mouse click
def MouseClick(event):
    global DebutDePartie, Grille, grid_color
    
    # If a match just ended, clicking resets the board
    if DebutDePartie:
        Grille = np.zeros((3, 3), dtype=int)
        grid_color = "blue"
        DebutDePartie = False
        Dessine()
        return

    # Collects the point of click and convert it in coordinates
    Window.focus_set()
    x, y = event.x // 100, event.y // 100
    if not (0 <= x <= 2 and 0 <= y <= 2) or Grille[x][y] != 0: return
    
    # Debug
    print("clicked at", x,y)

    # Human plays, win check and AI response
    PlayHuman(x, y)

    # Check if human won
    winner = WinnerCheck()
    if winner != 0:
        EndMatch(winner)
    else:
        # AI plays
        PlayAI()
        # Check if AI won
        winner = WinnerCheck()
        if winner != 0:
            EndMatch(winner)

    Dessine()
    
canvas.bind('<ButtonPress-1>', MouseClick) #Triggers the activation of MouseClick when a click occurs

# 4.3 Interface
AfficherPage(0)
Dessine()
Window.mainloop()