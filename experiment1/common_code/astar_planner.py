from dataclasses import dataclass
from math import sqrt
import heapq
from .data_structure import Point, GridPoint
from .collision import is_position_valid


class Astar:

    def __init__(self, resolution=0.01, env=None):

        self.resolution = resolution

        self.env = env
        self.bounds = getattr(env,"map_bounds",None)

        self.SQRT2_MINUS_1 = sqrt(2)-1.0


    def to_grid(self, point:Point)->GridPoint:

        return GridPoint(
            int(round(point.x/self.resolution)),
            int(round(point.y/self.resolution))
        )


    def from_grid(self, grid:GridPoint)->Point:

        return Point(
            round(grid.gx*self.resolution,4),
            round(grid.gy*self.resolution,4)
        )


    def heuristic(self,a:GridPoint,b:GridPoint)->float:
        dx = abs(a.gx-b.gx)
        dy = abs(a.gy-b.gy)

        h = max(dx,dy)+self.SQRT2_MINUS_1*min(dx,dy)

        # grid单位转换为实际距离
        return h*self.resolution


    def run_astar(self,start_point:Point,goal_point:Point):
        start = self.to_grid(start_point)
        goal = self.to_grid(goal_point)

        if start == goal:
            return [
                start_point,
                goal_point
            ]

        directions=[
            (1,0),
            (-1,0),
            (0,1),
            (0,-1),

            (1,1),
            (1,-1),
            (-1,1),
            (-1,-1)
        ]

        open_list=[]
        counter=0

        heapq.heappush(
            open_list,
            (
                self.heuristic(start,goal),
                counter,
                start,
                None
            )
        )


        came_from={}

        cost_so_far={
            start:0
        }



        final_node=None


        while open_list:
            _,_,current,parent=heapq.heappop(open_list)

            if current in came_from:
                continue

            came_from[current]=parent

            if current==goal:
                final_node=current
                break

            for dx,dy in directions:
                neighbor=GridPoint(
                    current.gx+dx,
                    current.gy+dy
                )


                neighbor_point=self.from_grid(neighbor)


                if not is_position_valid(neighbor_point, bounds=self.bounds):
                    continue



                move_cost = sqrt(dx*dx+dy*dy)*self.resolution


                new_cost = (
                    cost_so_far[current]
                    +
                    move_cost
                )

                if (
                    neighbor not in cost_so_far
                    or
                    new_cost < cost_so_far[neighbor]
                ):


                    cost_so_far[neighbor]=new_cost

                    counter+=1

                    priority=(
                        new_cost
                        +
                        self.heuristic(neighbor,goal)
                    )


                    heapq.heappush(
                        open_list,
                        (
                            priority,
                            counter,
                            neighbor,
                            current
                        )
                    )



        if final_node is None:
            return None
        # 回溯
        path=[]
        node=final_node

        while node is not None:
            path.append(node)
            node=came_from[node]


        path.reverse()

        full_path=[
            self.from_grid(p)
            for p in path
        ]

        # 简化路径
        new_path = self.remove_redundant_nodes(
            full_path,
            bounds=self.bounds
        )

        return new_path

    def is_obstacle_free(self, start: Point, end: Point, step_size: float | None = None, bounds: dict | None = None) -> bool:
        """检查从start到end的直线路径上是否有障碍物"""
        if bounds is None:
            bounds = self.bounds
        if step_size is None:
            step_size = self.resolution
        dist = ((end.x - start.x) ** 2 + (end.y - start.y) ** 2) ** 0.5
        steps = int(dist / step_size)
        if steps <= 0:
            return is_position_valid(end, bounds=bounds)
        for i in range(steps + 1):
            t = i / steps
            x = start.x + t * (end.x - start.x)
            y = start.y + t * (end.y - start.y)
            if not is_position_valid(Point(x, y), bounds=bounds):
                return False
        return True

    def remove_redundant_nodes(self, path: list[Point], bounds: dict | None = None) -> list[Point]:
        if bounds is None:
            bounds = self.bounds
        if len(path) < 2:
            return path
        if self.is_obstacle_free(path[0], path[-1], bounds=bounds):
            return [path[0], path[-1]]
        simplified_path = [path[0]]
        for i in range(1, len(path) - 1):
            start = simplified_path[-1]
            end = path[i + 1]
            if self.is_obstacle_free(start, end, bounds=bounds):
                continue
            else:
                simplified_path.append(path[i])
        simplified_path.append(path[-1])
        return simplified_path