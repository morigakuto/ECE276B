import numpy as np

from astar import AStar

def segment_block_collision(p1, p2, block):
  
  d = p2 - p1

  x_min,y_min,z_min,x_max,y_max,z_max = block[:6]

  if d[0] == 0 and (p1[0] < x_min or p1[0] > x_max):
    return False
  if d[1] == 0 and (p1[1] < y_min or p1[1] > y_max):
    return False
  if d[2] == 0 and (p1[2] < z_min or p1[2] > z_max):
    return False

  tx,ty,tz = [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]
  
  if d[0] != 0:
    tx = sorted([(x_min - p1[0]) / d[0], (x_max - p1[0]) / d[0]])

  if d[1] != 0:
    ty = sorted([(y_min - p1[1]) / d[1], (y_max - p1[1]) / d[1]])

  if d[2] != 0:
    tz = sorted([(z_min - p1[2]) / d[2], (z_max - p1[2]) / d[2]])

  t_min = max(tx[0], ty[0], tz[0])
  t_max = min(tx[1], ty[1], tz[1])

  return True if t_min <= t_max and t_max >= 0 and t_min <= 1 else False

class MyPlanner:
  __slots__ = ['boundary', 'blocks']

  def __init__(self, boundary, blocks):
    self.boundary = boundary
    self.blocks = blocks

  def is_inside_boundary(self, point):
    return (self.boundary[0,0] <= point[0] <= self.boundary[0,3] and
            self.boundary[0,1] <= point[1] <= self.boundary[0,4] and
            self.boundary[0,2] <= point[2] <= self.boundary[0,5])

  def is_point_in_collision(self, point):
    for k in range(self.blocks.shape[0]):
      if (self.blocks[k,0] <= point[0] <= self.blocks[k,3] and
          self.blocks[k,1] <= point[1] <= self.blocks[k,4] and
          self.blocks[k,2] <= point[2] <= self.blocks[k,5]):
        return True
    return False

  def is_valid_point(self, point):
    return self.is_inside_boundary(point) and not self.is_point_in_collision(point)
  
  def _directional_vectors(self, step=None):
    [dX,dY,dZ] = np.meshgrid([-1,0,1],[-1,0,1],[-1,0,1])
    dR = np.vstack((dX.flatten(),dY.flatten(),dZ.flatten()))
    dR = np.delete(dR,13,axis=1)
    if step is None:
      step = 0.5
    norms = np.sqrt(np.sum(dR**2,axis=0))
    dR = dR / norms * step
    return dR
    
  def plan(self,start,goal):
    path = [start]
    dR = self._directional_vectors()
    numofdirs = dR.shape[1]
    
    for _ in range(2000):
      mindisttogoal = 1000000
      node = None
      for k in range(numofdirs):
        next = path[-1] + dR[:,k]

        if not self.is_valid_point(next):
          continue
        
        disttogoal = sum((next - goal)**2)
        if( disttogoal < mindisttogoal):
          mindisttogoal = disttogoal
          node = next
      
      if node is None:
        break
      
      path.append(node)
      
      # Check if done
      if sum((path[-1]-goal)**2) <= 0.1:
        break
      
    return np.array(path)

class MyPart1SamplingPlanner(MyPlanner):
  __slots__ = ['boundary', 'blocks']

  def __init__(self, boundary, blocks):
    super().__init__(boundary, blocks)

  def plan(self, start, goal):
    path = [start]
    dR = self._directional_vectors()
    numofdirs = dR.shape[1]
    
    for _ in range(2000):
      mindisttogoal = 1000000
      node = None
      for k in range(numofdirs):
        start_point = path[-1]
        next_vertical = dR[:,k]
        
        step = 0.1
        for t in np.arange(step,1.0+step,step):
          next = start_point + t*next_vertical
          collision = False
          for block in self.blocks:
            if segment_block_collision(start_point, next, block):
              collision = True
              break
            
          if collision:
            continue

          if not self.is_valid_point(next):
            continue
          
          disttogoal = sum((next - goal)**2)
          if( disttogoal < mindisttogoal):
            mindisttogoal = disttogoal
            node = next
        
      if node is None:
        break
      
      path.append(node)
      
      if sum((path[-1]-goal)**2) <= 0.1:
        break
    
    return np.array(path)
  
class MyPart1AABBPlanner(MyPlanner):
  __slots__ = ['boundary', 'blocks']

  def __init__(self, boundary, blocks):
    super().__init__(boundary, blocks)

  def plan(self,start,goal):
    path = [start]
    dR = self._directional_vectors(step=1.0)
    numofdirs = dR.shape[1]
    
    for _ in range(2000):
      mindisttogoal = 1000000
      node = None
      for k in range(numofdirs):
        next = path[-1] + dR[:,k]
        collision = False

        for block in self.blocks:
          if segment_block_collision(path[-1], next, block):
            collision = True
            break
          
        if collision:
          continue
          
        if not self.is_valid_point(next):
          continue
          
        disttogoal = sum((next - goal)**2)
        if( disttogoal < mindisttogoal):
          mindisttogoal = disttogoal
          node = next
      
      if node is None:
        break
      
      path.append(node)
      
      if sum((path[-1]-goal)**2) <= 0.1:
        break
      
    return np.array(path)
  
  
# Part 2: A* Planner
class MyAStarPlanner(MyPlanner):
  __slots__ = ['boundary', 'blocks', 'goal', 'step_size', 'stats']

  def __init__(self, boundary, blocks, step_size=0.275):
    super().__init__(boundary, blocks)
    self.goal = None
    self.step_size = step_size
    self.stats = {}  # Initialize stats
    
  def _directional_vectors(self, step=None):
    if step is None:
      step = self.step_size
    dirs = np.array(np.meshgrid([-1, 0, 1],
                                [-1, 0, 1],
                                [-1, 0, 1])).reshape(3,-1)
    dirs = np.delete(dirs,13,axis=1)
    return dirs * step

  def plan(self, start, goal):
    self.goal = goal
    result = AStar.plan(start, goal, self)

    # Handle both old and new return format
    if isinstance(result, tuple):
        path_coords, self.stats = result
    else:
        path_coords = result
        self.stats = {}

    return np.array(path_coords)

  def getHeuristic(self, coord):
    if self.goal is None:
        raise ValueError("Goal is not set. Call plan(start, goal) before querying the heuristic.")
    return np.linalg.norm(np.asarray(coord) - np.asarray(self.goal))
  
  def getNeighbors(self, coord):
    neighbors = []
    dirs = self._directional_vectors(self.step_size)
    coord = np.asarray(coord)
    for i in range(dirs.shape[1]):
      transition_cost = np.linalg.norm(dirs[:,i])
      neighbor = (coord + dirs[:,i], transition_cost)
      neighbors.append(neighbor)
    return neighbors

  def checkCollision(self, p1, p2):
    for block in self.blocks:
        if segment_block_collision(np.asarray(p1), np.asarray(p2), block):
            return True
    return False

  def isValidPoint(self, point):
    
    return self.is_valid_point(np.asarray(point))

  def coord_to_key(self, coord):
    
    return tuple(np.round(np.asarray(coord) / self.step_size).astype(int))

# Part 3: RRT Planner (将来的に追加)
class MyRRTPlanner:
  __slots__ = ['boundary', 'blocks']

  def __init__(self, boundary, blocks):
    self.boundary = boundary
    self.blocks = blocks
    # RRTに必要な情報をここで初期化

  def plan(self, start, goal):
    # RRTアルゴリズムをここに実装
    pass
