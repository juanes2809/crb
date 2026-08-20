import numpy as np
import trimesh

def create_sparse_wall(origin, width_vec, height_vec, color, sphere_radius, spacing):
    """

    
    :param origin: Wall origin vector (lower left corner).
    :param width_vec: Vector defining the width of the wall.
    :param height_vec: Vector defining the height of the wall.
    :param color: Color of the spheres.
    :param sphere_radius: Radius of each sphere.
    :param spacing: Spacing between the spheres.
    :return: List of spheres (Trimesh meshes).
    """
    wall_spheres = []
    
    # Calculate the number of spheres in each dimension
    num_x = int(np.linalg.norm(width_vec) // spacing)
    num_z = int(np.linalg.norm(height_vec) // spacing)
    
    # Unit vectors to iterate
    unit_width = width_vec / np.linalg.norm(width_vec)
    unit_height = height_vec / np.linalg.norm(height_vec)
    
    for i in range(num_x + 1):
        for j in range(num_z + 1):
            pos = origin + i * spacing * unit_width + j * spacing * unit_height
            sphere = trimesh.creation.icosphere(subdivisions=1, radius=sphere_radius)
            sphere.apply_translation(pos)
            wall_spheres.append(sphere)
    
    return wall_spheres