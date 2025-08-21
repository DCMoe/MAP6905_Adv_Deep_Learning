# Week 1: Matrix Review

## Lecture 1: The Geometry of Linear Equations

Jake notes: In this section we need to communicate row and column pictures; intuition for solving systems of equations in both ways; introduce foundations for invertibility in high dim spaces (even if we don't use that language yet)

## Summary
This lecture introduces linear algebra through the lens of solving systems of equations. We can view matrices two ways, either through the "row picture" (geometric intersection of lines/planes defined by each row-equation) and the "column picture" (linear combinations of vectors corresponding to each column). For a system like Ax = b, we explore how to visualize and solve it.

## Key Concepts
- **Row Picture**: Each equation represents a line (in 2D) or plane (in 3D). Solutions are the intersections of these structures.
- **Column Picture**: Rewrite as x * column1 + y * column2 = b, finding coefficients for vector combinations.
- Basic solving via elimination: Subtract multiples of equations to isolate variables.

## Tutorial: Solving a 2x2 System
Consider the system:
- 2x + y = 5
- x - y = 1

**Step 1 (Elimination)**: Add the equations to eliminate y: 3x = 6 → x = 2.  
**Step 2**: Substitute: 2(2) + y = 5 → y = 1.  

In the row picture: Lines intersect at (2,1).  
In the column picture: 2 * [2,1]^T + 1 * [1,-1]^T = [5,1]^T.

### Python Example (Using NumPy)
```python
import numpy as np

# Define matrix A and vector b
A = np.array([[2, 1], [1, -1]])
b = np.array([5, 1])

# Solve Ax = b
x = np.linalg.solve(A, b)
print("Solution: x =", x[0], ", y =", x[1])
```
### Video Lecture
[![Video](youtube.png)](https://www.youtube.com/watch?v=ZK3O402wf1c)

## Lecture 2: Elimination with matrices