# Function to create wall mesh
import numpy as np
import trimesh

def create_wall_mesh(origin, u_vec, v_vec):
    width = np.linalg.norm(u_vec)
    height = np.linalg.norm(v_vec)
    
    wall = trimesh.creation.box(extents=(width, 0.01, height))  # Thickness of 0.01 in Y-axis
    
    # Calculate rotation to align wall
    normal = np.cross(u_vec, v_vec)
    normal = normal / np.linalg.norm(normal)
    y_axis = np.array([0, 1, 0])  
    rotation_axis = np.cross(y_axis, normal)
    rotation_angle = np.arccos(np.clip(np.dot(y_axis, normal), -1.0, 1.0))
    if np.linalg.norm(rotation_axis) > 0:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        rotation_matrix_wall = trimesh.transformations.rotation_matrix(rotation_angle, rotation_axis)
        wall.apply_transform(rotation_matrix_wall)
    
    # Position the wall in the scene
    wall.apply_translation(origin + 0.5 * (u_vec + v_vec))
    return wall