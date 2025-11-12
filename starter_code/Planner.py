import numpy as np
from scipy.spatial import KDTree
from astar import AStar
from rrt import RRT, RRTStar

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

# Part 3: RRT Planner
class MyRRTPlanner(MyPlanner):
  __slots__ = ['boundary', 'blocks', 'goal', 'step_size', 'stats']

  def __init__(self, boundary, blocks, step_size=0.275):
    super().__init__(boundary, blocks)
    self.step_size = step_size
    self.stats = {}  # Initialize stats

  def plan(self, start, goal):
    self.goal = goal
    result = RRT.plan(start, goal, self)

    # Handle both old and new return format
    if isinstance(result, tuple):
        path, self.stats = result
    else:
        path = result
        self.stats = {}

    path = np.array(path)
    return path
  
  def getRandomPoint(self):
    boundary = self.boundary
    while True:
        x = np.random.uniform(boundary[0, 0], boundary[0, 3])
        y = np.random.uniform(boundary[0, 1], boundary[0, 4])
        z = np.random.uniform(boundary[0, 2], boundary[0, 5])
        if self.is_valid_point(np.array([x,y,z])):
            return np.array([x,y,z])
        
  def getNearestNode(self, V, kdtree, point):
    dist, idx = kdtree.query(point, k=1)
    return V[idx]

  def Steer(self, from_node, to_point, step_size):
    direction = to_point - from_node.coord
    length = np.linalg.norm(direction)
    if length == 0.0:
        return from_node.coord
    if length < step_size:
        return to_point
    else:
      direction = direction / length
      new_point = from_node.coord + step_size * direction
      return new_point

  def checkCollision(self, p1, p2):
    for block in self.blocks:
        if segment_block_collision(np.asarray(p1), np.asarray(p2), block):
            return True
    return False
  
class MyRRTStarPlanner(MyRRTPlanner):
  __slots__ = ['boundary', 'blocks', 'goal', 'step_size', 'stats', 'vol_free', 'gamma_rrt_star', 'max_rewire_radius']

  def __init__(self, boundary, blocks, step_size=0.275, radius=None):
    super().__init__(boundary, blocks, step_size)

    boundary_vol = (self.boundary[0, 3] - self.boundary[0, 0]) * \
                   (self.boundary[0, 4] - self.boundary[0, 1]) * \
                   (self.boundary[0, 5] - self.boundary[0, 2])
    obstacle_vol = 0
    for block in self.blocks:
        obstacle_vol += (block[3] - block[0]) * \
                        (block[4] - block[1]) * \
                        (block[5] - block[2])
    self.vol_free = boundary_vol - obstacle_vol
    if self.vol_free <= 0: self.vol_free = 1e-6 

    d = 3.0 
    vol_unit_ball = (4.0/3.0) * np.pi
    self.gamma_rrt_star = 2.0 * ((1.0 + 1.0/d) ** (1.0/d)) * ((self.vol_free / vol_unit_ball) ** (1.0/d))

    if radius is None:
      extents = self.boundary[0, 3:6] - self.boundary[0, 0:3]
      diagonal = np.linalg.norm(extents)
      self.max_rewire_radius = max(self.step_size, diagonal)
    else:
      self.max_rewire_radius = max(self.step_size, radius)

  def plan(self, start, goal):
    self.goal = goal
    result = RRTStar.plan(start, goal, self)

    # Handle both old and new return format
    if isinstance(result, tuple):
        path, self.stats = result
    else:
        path = result
        self.stats = {}

    path = np.array(path)
    return path

  def get_dynamic_radius(self, num_nodes):
    d = 3.0
    if num_nodes < 2:
        return self.step_size 
    r_star = self.gamma_rrt_star * ((np.log(num_nodes) / num_nodes) ** (1.0/d))
  
    return min(r_star, self.max_rewire_radius)
  
  def getNeighborNodes(self, V, kdtree, point, radius):
    idxs = kdtree.query_ball_point( point, r=radius)
    neighbors = [V[idx] for idx in idxs]
    return neighbors

  def chooseParent(self, V, kdtree, new, radius):
    neighbors_nodes = self.getNeighborNodes(V, kdtree, new.coord, radius)
    current_parent = new.parent
    best_parent = current_parent
    best_edge_cost = np.linalg.norm(new.coord - current_parent.coord) if current_parent is not None else None
    min_cost = new.cost

    for neighbor in neighbors_nodes:
      if neighbor is new:
        continue
      edge_cost = np.linalg.norm(neighbor.coord - new.coord)
      potential_cost = neighbor.cost + edge_cost
      if potential_cost < min_cost:
        if not self.checkCollision(neighbor.coord, new.coord):
          min_cost = potential_cost
          best_parent = neighbor
          best_edge_cost = edge_cost

    if best_parent is not None:
      new.set_parent(best_parent, best_edge_cost)
    else:
      new.set_parent(None, min_cost)

    return new.parent, new.cost

  def _update_descendant_costs(self, root):
    stack = list(root.children)
    while stack:
      current = stack.pop()
      if current.parent is None:
        stack.extend(current.children)
        continue
      current.cost = current.parent.cost + np.linalg.norm(current.coord - current.parent.coord)
      stack.extend(current.children)

  def Rewiring(self, V, kdtree, new_node, radius):
    neighbor_nodes = self.getNeighborNodes(V, kdtree, new_node.coord, radius)
    for neighbor in neighbor_nodes:
      if neighbor == new_node.parent or neighbor is new_node:
        continue
      edge_cost = np.linalg.norm(new_node.coord - neighbor.coord)
      potential_cost = new_node.cost + edge_cost
      if potential_cost < neighbor.cost:
        if not self.checkCollision(new_node.coord, neighbor.coord):
          neighbor.set_parent(new_node, edge_cost)
          self._update_descendant_costs(neighbor)
