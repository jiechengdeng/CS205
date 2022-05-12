from queue import PriorityQueue
import time
import argparse
import matplotlib.pyplot as plt

class Puzzle:
    def __init__(self, s):
        self.state = s
        self.h = 0
        self.f = 0
        self.g = 0
        self.len = 3
        self.depth = 0

    def __lt__(self,obj):
        return self.f <= obj.f

    def goal_test(self):
        for i in range(self.len):
            for j in range(self.len):
                if self.state[i][j] != goal[i][j]:
                    return False
        return True

    def misplacedTileCount(self):
        for i in range(self.len):
            for j in range(self.len):
                if goal[i][j] == 0:
                    continue
                if self.state[i][j] != goal[i][j]:
                    self.h += 1
        self.f += self.h

    def manhattanDistance(self):
        for i in range(self.len):
            for j in range(self.len):
                if self.state[i][j] == 0:
                    continue
                if self.state[i][j] != goal[i][j]:
                    p = pos[self.state[i][j]]
                    d = abs(p[0] - i) + abs(p[1] - j)
                    self.h += d
        self.f += self.h

    
class Stats:
    def __init__(self):
        self.depth = 0
        self.nodesVisited = 0
        self.numberOfexpansion = 0
        self.start_time = 0
        self.executionTime = 0
        self.method = ""
        self.maxQueueSize = 0
        self.depth_arr = [2,4,8,12,16,20,24]
    def report(self):
        print("Result of {}:\nFind a solution at depth {}\nNode visited: {}\nNode expanded: {}\nMax queue size: {}\n".format(self.method,self.depth,
        self.nodesVisited,self.numberOfexpansion,self.maxQueueSize))
        # print("Result of {}:\nFind a solution at depth {}\nNode visited: {}\nNode expanded: {}\nMax queue size: {}\nExecution time:{} seconds\n".format(self.method,self.depth,
        # self.nodesVisited,self.numberOfexpansion,self.maxQueueSize,round(self.executionTime,2)))
    def clear(self):
        self.depth = 0
        self.nodesVisited = 0
        self.numberOfexpansion = 0
        self.start_time = 0
        self.executionTime = 0
        self.maxQueueSize = 0

s = Stats()

def general_search(problem):
    nodes = PriorityQueue()
    p = Puzzle(problem)
    visited = {}

    nodes.put(p)
    while not nodes.empty():
        s.maxQueueSize = max(nodes.qsize(),s.maxQueueSize)
        current = nodes.get()
        
        if current.goal_test():
            s.depth = current.depth
            s.numberOfexpansion = len(visited)
            s.executionTime = time.time() - s.start_time
            s.report()
            return current
        s.nodesVisited += 1
        queue_function(nodes,current,visited)
    
    return None
        
# this function uses operator to expand current state 
def queue_function(nodes,current_state,visited):
    key = generate_key(current_state.state)
    visited[key] = True

    for i in range(current_state.len):
        for j in range(current_state.len):
            if current_state.state[i][j] == 0:
                if i - 1 >= 0:
                    move(current_state,i,j,i-1,j,visited,nodes)
                if i + 1 < current_state.len:
                    move(current_state,i,j,i+1,j,visited,nodes)
                if j - 1 >= 0:
                    move(current_state,i,j,i,j-1,visited,nodes)
                if j + 1 < current_state.len:
                    move(current_state,i,j,i,j+1,visited,nodes)
                break
    next = nodes.queue[0]
    print("The best state to expand with a g(n) = {} and h(n) = {} is:".format(next.g,next.h))
    printPuzzle(next.state)

# this function generates a possible state and check whether is a repeated state
# and calculate f(n)
def move(current,r1,c1,r2,c2,visited,nodes):
    next_state = [[i for i in row] for row in current.state]
    next_state[r1][c1] = next_state[r2][c2]
    next_state[r2][c2] = 0
    key = generate_key(next_state)
    if key not in visited:
        newP = Puzzle(next_state)
        newP.depth = current.depth + 1       
        newP.g = current.g + 1
        newP.f += newP.g
        if s.method == "misplacedTile":
            newP.misplacedTileCount()
        elif s.method == "the Manhattan Distance Heuristic":
            newP.manhattanDistance()
        nodes.put(newP)

def generate_key(state):
    key = ""
    for i in range(len(state)):
        for e in state[i]:
            key += str(e)
    return key 

def printPuzzle(p):
    for r in p:
        print(r)
    print('--------------------')

def execution_time_plot():
    execute_time_uniform = [0,0,0,0.03,0.24,1.52,4.91]
    execute_time_mpt = [0,0,0,0,0.01,0.07,0.35]
    execute_time_man = [0,0,0,0,0,0,0.05]
    fig,ax = plt.subplots()
    ax.set_ylabel("Time in Seconds",fontsize=20)
    ax.set_xlabel("Solution at Depth",fontsize=20)
    ax.plot(s.depth_arr,execute_time_uniform,label="Uniform-cost")
    ax.plot(s.depth_arr,execute_time_mpt,label = "Misplaced-tile")
    ax.plot(s.depth_arr,execute_time_man,label="Manhattan Distance")
    ax.set_xticks(s.depth_arr)
    ax.legend(fontsize=15)

    for i in range(4,len(execute_time_uniform)):
        ax.text(s.depth_arr[i],execute_time_uniform[i], execute_time_uniform[i],size=10)
        ax.text(s.depth_arr[i],execute_time_mpt[i], execute_time_mpt[i],size=10)
        
    for i in range(6,len(execute_time_uniform)):
        ax.text(s.depth_arr[i],execute_time_man[i], execute_time_man[i],size=10)

    plt.show()

def node_expand_plot():
    node_expand_uniform = [6,26,283,1537,10455,51422,121512]
    node_expand_mpt = [2,4,16,99,513,2767,14015]
    node_expand_man = [2,4,8,24,71,223,1383]
    fig,ax = plt.subplots()
    ax.set_ylabel("Nodes Expanded",fontsize=20)
    ax.set_xlabel("Solution at Depth",fontsize=20)
    ax.plot(s.depth_arr,node_expand_uniform,label="Uniform-cost")
    ax.plot(s.depth_arr,node_expand_mpt,label = "Misplaced-tile")
    ax.plot(s.depth_arr,node_expand_man,label="Manhattan Distance")
    ax.set_xticks(s.depth_arr)
    ax.legend(fontsize=15)
    for i in range(4,len(node_expand_uniform)):
        ax.text(s.depth_arr[i],node_expand_uniform[i], node_expand_uniform[i],size=10)
        ax.text(s.depth_arr[i],node_expand_mpt[i], node_expand_mpt[i],size=10)
        
    for i in range(6,len(node_expand_man)):
        ax.text(s.depth_arr[i],node_expand_man[i], node_expand_man[i],size=10)
    plt.show()

def node_visited_plot():
    node_visited_uniform = [6,26,290,1678,12390,74394,225957]
    node_visited_mpt = [2,4,16,99,530,2974,15592]
    node_visited_man = [2,4,8,24,71,225,1416]

    fig,ax = plt.subplots()
    ax.set_ylabel("Nodes Visited",fontsize=20)
    ax.set_xlabel("Solution at Depth",fontsize=20)
    ax.plot(s.depth_arr,node_visited_uniform,label="Uniform-cost")
    ax.plot(s.depth_arr,node_visited_mpt,label = "Misplaced-tile")
    ax.plot(s.depth_arr,node_visited_man,label="Manhattan Distance")
    ax.set_xticks(s.depth_arr)
    ax.legend(fontsize=15)
    for i in range(4,len(node_visited_uniform)):
        ax.text(s.depth_arr[i],node_visited_uniform[i], node_visited_uniform[i],size=10)
        ax.text(s.depth_arr[i],node_visited_mpt[i], node_visited_mpt[i],size=10)
        
    for i in range(6,len(node_visited_man)):
        ax.text(s.depth_arr[i],node_visited_man[i], node_visited_man[i],size=10)
    plt.show()

def max_queue_size_plot():
    max_queue_size_uniform = [8,22,185,1093,7977,33199,71269]
    max_queue_size_mpt = [3,6,15,76,353,1833,8672]
    max_queue_size_man = [3,6,10,20,52,148,841]

    fig,ax = plt.subplots()
    ax.set_ylabel("Max Queue Size",fontsize=20)
    ax.set_xlabel("Solution at Depth",fontsize=20)
    ax.plot(s.depth_arr,max_queue_size_uniform,label="Uniform-cost")
    ax.plot(s.depth_arr,max_queue_size_mpt,label = "Misplaced-tile")
    ax.plot(s.depth_arr,max_queue_size_man,label="Manhattan Distance")
    ax.set_xticks(s.depth_arr)
    ax.legend(fontsize=15)
    for i in range(4,len(max_queue_size_uniform)):
        ax.text(s.depth_arr[i],max_queue_size_uniform[i], max_queue_size_uniform[i],size=10)
        ax.text(s.depth_arr[i],max_queue_size_mpt[i], max_queue_size_mpt[i],size=10)
        
    for i in range(5,len(max_queue_size_man)):
        ax.text(s.depth_arr[i],max_queue_size_man[i], max_queue_size_man[i],size=10)
    plt.show()

if __name__ == "__main__":
    global goal 
    global pos


    parser = argparse.ArgumentParser(description='8-puzzle solver',allow_abbrev=False)

    parser.add_argument("-d",type=int,help="depth",choices=range(2, 25,2),default=24)
    parser.add_argument("-m",type=int,help="method",choices=[0,1,2],default=2)
    args = parser.parse_args()

    # test cases in project 1
    problemList = {}
    problemList[2] = [[1,2,3],[4,5,6],[0,7,8]]
    problemList[4] = [[1,2,3],[5,0,6],[4,7,8]]
    problemList[8] = [[1,3,6],[5,0,2],[4,7,8]]
    problemList[12] = [[1,3,6],[5,0,7],[4,8,2]]
    problemList[16] = [[1,6,7],[5,0,3],[4,8,2]]
    problemList[20] = [[7,1,2],[4,8,5],[6,3,0]]
    problemList[24] = [[0,7,2],[4,6,1],[3,5,8]] 

    methods = ["Uniform Cost Search","misplacedTile","the Manhattan Distance Heuristic"]
    print("You select the puzzle with the optimal solution at depth: ",args.d)
    printPuzzle(problemList[args.d])
    print("Use ",methods[args.m])

    goal = [
        [1,2,3],
        [4,5,6],
        [7,8,0]
    ]
    pos = {}
    count = 1
    for i in range(3):
        for j in range(3):
            if count == 9:
                break
            pos[count] = [i,j]
            count += 1
   
    s.start_time = time.time()
    s.method = methods[args.m]

    general_search(problemList[args.d])




    #execution_time_plot()
    #node_expand_plot()
    #node_visited_plot()
    #max_queue_size_plot()
    


    # for i,j in problemList.items():    
    #     s.start_time = time.time()
    #     general_search(j)
    #     s.clear()

    # print("Result of {}:\nFind a solution at depth {}\nNode visited: {}\nNode expanded: {}\nMax queue size: {}\nExecution time:{} seconds\n".format(self.method,self.depth,
        # self.nodesVisited,self.numberOfexpansion,self.maxQueueSize,round(self.executionTime,2)))