from pqdict import pqdict
import numpy as np
import math

class AStarNode:
  def __init__(self, pqkey, coord, hval):
    self.pqkey = pqkey
    self.coord = coord
    self.g = math.inf
    self.h = hval
    self.parent_node = None
    self.parent_action = None
    self.closed = False
  def __lt__(self, other):
    f_self = self.g + self.h
    f_other = other.g + other.h
    if f_self != f_other:
      return f_self < f_other
    return self.h < other.h


class AStar:
  @staticmethod
  def plan(start_coord, goal_coord, environment, epsilon=1.0):

    Graph = {}
    OPEN = pqdict()
    nodes_expanded = 0

    start_key = environment.coord_to_key(start_coord)
    start_node = AStarNode(start_key, start_coord, environment.getHeuristic(start_coord))
    start_node.g = 0.0
    Graph[start_key] = start_node
    OPEN[start_key] = start_node.g + start_node.h

    while OPEN:
      current_key, _ = OPEN.popitem()
      current_node = Graph[current_key]
      nodes_expanded += 1

      if np.linalg.norm(current_node.coord - goal_coord) <= np.sqrt(0.1):

        path = AStar.reconstruct_path(current_node)
        stats = {
            'nodes_considered': len(Graph),
            'nodes_expanded': nodes_expanded
        }
        return path, stats

      current_node.closed = True

      for neighbor_coord, cost in environment.getNeighbors(current_node.coord):
        if environment.isValidPoint(neighbor_coord) and \
           not environment.checkCollision(current_node.coord, neighbor_coord):
          next_cost = current_node.g + cost
          neighbor_key = environment.coord_to_key(neighbor_coord)
          if neighbor_key not in Graph:
            neighbor_node = AStarNode(neighbor_key, neighbor_coord, environment.getHeuristic(neighbor_coord))
            Graph[neighbor_key] = neighbor_node
            neighbor_node.g = next_cost
            neighbor_node.parent_node = current_node
            OPEN.additem(neighbor_key, next_cost + neighbor_node.h)
          elif Graph[neighbor_key].closed:
            continue
          else:
            neighbor_node = Graph[neighbor_key]
            if next_cost < Graph[neighbor_key].g:
              Graph[neighbor_key].g = next_cost
              Graph[neighbor_key].parent_node = current_node
              OPEN[neighbor_key] = next_cost + Graph[neighbor_key].h
        else:
          continue

    stats = {
        'nodes_considered': len(Graph),
        'nodes_expanded': nodes_expanded
    }
    raise RuntimeError(f"A* search failed to find a path to the goal. Explored {nodes_expanded} nodes.")
  
  @staticmethod
  def reconstruct_path(goal_node):
    """
    Utility for backtracking once the goal node is found.
    """
    path = []
    curr = goal_node
    while curr is not None:
      path.append(curr.coord)
      curr = curr.parent_node
    return list(reversed(path))

