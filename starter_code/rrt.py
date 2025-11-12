import numpy as np
from scipy.spatial import KDTree

class RRTNode:
    def __init__(self,coord,parent = None,cost = None):
        self.coord = coord
        self.parent = None
        self.children = []
        self.cost = 0.0

        if parent is not None:
            self.set_parent(parent, cost)
        else:
            self.cost = 0.0 if cost is None else cost

    def set_parent(self, new_parent, edge_cost=None):
        if self.parent is not new_parent:
            if self.parent is not None:
                try:
                    self.parent.children.remove(self)
                except ValueError:
                    pass
            self.parent = new_parent
            if new_parent is not None:
                new_parent.children.append(self)

        if new_parent is None:
            if edge_cost is not None:
                self.cost = edge_cost
            return

        if edge_cost is None:
            edge_cost = np.linalg.norm(self.coord - new_parent.coord)
        self.cost = new_parent.cost + edge_cost
    
class RRT:
    @staticmethod
    def plan(start, goal, environment):
        V = [RRTNode(start)]
        iterations = 0  # Track iterations

        while True:
            iterations += 1
            coordinates = [node.coord for node in V]
            kdtree = KDTree(coordinates)
            x_rand = environment.getRandomPoint()
            x_nearest = environment.getNearestNode(V,kdtree,x_rand)
            x_new_coord = environment.Steer(x_nearest, x_rand, environment.step_size)

            if not environment.checkCollision(x_nearest.coord, x_new_coord):
                x_new = RRTNode(x_new_coord, x_nearest)
                V.append(x_new)

                if np.linalg.norm(x_new.coord - goal) <= np.sqrt(0.1):
                    if not environment.checkCollision(x_new.coord, goal):
                        x_goal = RRTNode(goal, x_new)
                        path = RRT.reconstruct_path(x_goal)
                        stats = {
                            'nodes_considered': len(V),  # Nodes in tree
                            'iterations': iterations  # Total iterations
                        }
                        return path, stats
                    
    @staticmethod
    def reconstruct_path(goal_node):
        path = []
        curr = goal_node
        while curr is not None:
            path.append(curr.coord)
            curr = curr.parent
        return list(reversed(path))

class RRTStar:
    @staticmethod
    def plan(start, goal, environment):
        V = [RRTNode(start)]
        iterations = 0  # Track iterations
        rewire_count = 0  # Track rewiring operations

        while True:
            iterations += 1
            coordinates = [node.coord for node in V]
            kdtree = KDTree(coordinates)
            x_rand = environment.getRandomPoint()
            current_radius = environment.get_dynamic_radius(len(V))
            x_nearest = environment.getNearestNode(V,kdtree,x_rand)
            x_new_coord = environment.Steer(x_nearest, x_rand, environment.step_size)

            if not environment.checkCollision(x_nearest.coord, x_new_coord):
                cost = np.linalg.norm(x_new_coord - x_nearest.coord)
                x_new = RRTNode(x_new_coord, x_nearest, cost)
                environment.chooseParent(V, kdtree, x_new, current_radius)
                V.append(x_new)
                environment.Rewiring(V, kdtree, x_new, current_radius)
                rewire_count += 1

                if np.linalg.norm(x_new.coord - goal) <= np.sqrt(0.1):
                    if not environment.checkCollision(x_new.coord, goal):
                        x_goal = RRTNode(goal, x_new)
                        path = RRT.reconstruct_path(x_goal)
                        stats = {
                            'nodes_considered': len(V),  # Nodes in tree
                            'iterations': iterations,  # Total iterations
                            'rewire_operations': rewire_count  # RRT* specific
                        }
                        return path, stats