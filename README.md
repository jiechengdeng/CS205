# CS205
Project1 

1. This project solves the 8-puzzle problem by implementing the following alogrithms:
  - uniform cost search 
  - A* with The Misplaced Tile Heuristic
  - A* with The Manhattan Distance Heuristic 

2. How to run the program?
  - use flags -d and -m
  - d means choose the puzzle with the optimal solution at depth d. d can only be [2,4,8,12,16,20,24]
  - m means select an algorithm to solve the problem
  - m can only be [0,1,2]  0 = uniform cost search 1 = A* with The Misplaced Tile Heuristic 2 = A* with The Manhattan Distance Heuristic 
  
  e.g: python n-puzzle.py -d 24 -m 2 
